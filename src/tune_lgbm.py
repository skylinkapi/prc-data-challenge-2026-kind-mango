"""Optuna hyperparameter search over LightGBM. Features frozen from v4.

Builds features once, caches them, then runs N trials varying LGBM params.
Reports per-trial validation RMSE, saves best model as lgbm_v5.
"""
import glob
import os
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
CACHE = os.path.join(ROOT, "models", "features_cache")
HOLDOUT_MONTHS = {1, 7}
N_TRIALS = 60

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]
NUM_COLS_V1 = ["hour", "dow", "month", "sched_delay",
               "sknt", "gust_kt", "wind_cross_kt", "wind_head_kt",
               "vis_km", "low_vis", "very_low_vis",
               "ceiling_ft", "low_ceiling",
               "wx_precip", "wx_snow", "wx_thunder", "wx_freezing",
               "dep_load_prev_15m", "dep_load_prev_30m", "dep_load_prev_60m",
               "arr_load_prev_15m", "arr_load_prev_30m", "arr_load_prev_60m",
               "dep_same_rwy_prev_15m", "dep_same_rwy_prev_30m",
               "dep_same_rwy_prev_60m", "dep_queue_next_10m"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def build_and_cache():
    if os.path.exists(CACHE + ".train.parquet"):
        print("Loading cached features...")
        train = pd.read_parquet(CACHE + ".train.parquet")
        test = pd.read_parquet(CACHE + ".test.parquet")
        with open(CACHE + ".encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open(CACHE + ".feat.txt") as f:
            feat = f.read().splitlines()
        return train, test, feat, encoders

    t0 = time.time()
    print("Loading movements + building features (first run)...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep = dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)]
    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train, test = dep.loc[~hold].copy(), dep.loc[hold].copy()

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS

    # Save cache: keep only columns needed + label
    keep = feat + ["TAXITIME_SEC_mvt", "ADEP_mvt"]
    # ADEP is already in feat, so dedupe
    keep = list(dict.fromkeys(keep))
    train[keep].to_parquet(CACHE + ".train.parquet")
    test[keep].to_parquet(CACHE + ".test.parquet")
    with open(CACHE + ".encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(CACHE + ".feat.txt", "w") as f:
        f.write("\n".join(feat))
    print(f"  cache written ({time.time()-t0:.1f}s)")
    return train, test, feat, encoders


def objective(trial: optuna.Trial, train, test, feat) -> float:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 63, 511),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 5,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
        "verbosity": -1, "num_threads": -1,
    }

    y_tr = train["TAXITIME_SEC_mvt"].values
    y_te = test["TAXITIME_SEC_mvt"].values
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        params, dtrain, num_boost_round=2500,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    p = booster.predict(test[feat], num_iteration=booster.best_iteration)
    return rmse(y_te, p)


def main():
    train, test, feat, encoders = build_and_cache()
    print(f"Train {len(train):,}   Test {len(test):,}   Features {len(feat)}")

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction="minimize", storage=storage,
                                sampler=optuna.samplers.TPESampler(seed=0))
    t0 = time.time()
    study.optimize(lambda t: objective(t, train, test, feat),
                   n_trials=N_TRIALS, show_progress_bar=False)
    print(f"\nOptuna done in {time.time()-t0:.0f}s")
    print(f"Best RMSE: {study.best_value:.2f}s")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Refit best on train+test-worth-of-features NO — keep test for reporting
    best = {**study.best_params, "objective": "regression", "metric": "rmse",
            "bagging_freq": 5, "verbosity": -1, "num_threads": -1}
    y_tr = train["TAXITIME_SEC_mvt"].values
    y_te = test["TAXITIME_SEC_mvt"].values
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)
    print("\nRefitting best config with full early stopping...")
    booster = lgb.train(best, dtrain, num_boost_round=5000,
                        valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    p = booster.predict(test[feat], num_iteration=booster.best_iteration)
    print(f"\nlgbm_v5 hold-out RMSE: {rmse(y_te, p):.2f}s   (v4 = 297.8)")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].values, "y": y_te, "p": p}) \
        .groupby("apt", observed=True).apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(1)
    v4 = pd.Series({"LEMD":208.9,"LSZH":230.0,"LEBL":242.7,"EDDF":247.5,"EDDM":247.9,
                    "EHAM":251.5,"LTFM":296.3,"LFPG":333.4,"EGLL":364.4,"LIRF":466.6})
    tab["v4"] = v4
    tab["delta"] = (tab["rmse"] - tab["v4"]).round(1)
    print("\nPer-airport RMSE (v5 vs v4):")
    print(tab.sort_values("rmse").to_string())

    out_dir = os.path.join(ROOT, "models")
    booster.save_model(os.path.join(out_dir, "lgbm_v5.txt"))
    with open(os.path.join(out_dir, "lgbm_v5.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_v5.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(out_dir, "lgbm_v5.params.txt"), "w") as f:
        for k, v in best.items():
            f.write(f"{k}: {v}\n")
    print(f"\nSaved -> {out_dir}/lgbm_v5.*")


if __name__ == "__main__":
    main()
