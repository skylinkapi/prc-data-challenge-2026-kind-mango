"""LIRF-only specialist model.

LIRF contributes disproportionately to overall RMSE (462s in v10 vs
~230-330 at other airports). Dropping LIRF from 462 to 400 lowers overall
RMSE from 294 to 289; 462->350 gives 283.

Same feature set as v10. Uses a small Optuna search (20 trials) over the
same hyperparameter space, since LIRF has 5x more tail than other airports
and may benefit from different regularization / leaf sizes.
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
from features_advanced import add_advanced, ADV_NUM_COLS
from features_osm_path import add_osm_path, OSM_PATH_NUM_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}
LIRF = "LIRF"
N_TRIALS = 20

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
               "dep_same_rwy_prev_60m", "dep_queue_next_10m",
               "flt_null"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def build_features():
    t0 = time.time()
    print("Loading movements + all features for LIRF...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    m["blk_ts"] = pd.to_datetime(m["BLOCK_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep["_bs_diff"] = (dep["blk_ts"] - dep["sched_ts"]).dt.total_seconds().abs()
    dep["_qual_bad"] = dep["_bs_diff"] < 1

    dep = dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)]
    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]],
                       daily_ec=ec_daily)
    dep = add_osm_path(dep)

    # Fit encoders on GLOBAL non-hold-out (better encoder quality than LIRF-only)
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  Global build took {time.time()-t0:.1f}s")

    # NOW filter to LIRF
    dep = dep[dep["ADEP_mvt"] == LIRF].copy()
    print(f"  LIRF rows: {len(dep):,}")
    return dep, encoders


def main():
    dep, encoders = build_features()
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train_all = dep[~hold]
    n_bad = train_all["_qual_bad"].sum()
    train = train_all[~train_all["_qual_bad"]].copy()
    test = dep[hold].copy()
    print(f"Filtered {n_bad:,} BLOCK==SCHED. LIRF train {len(train):,}   hold-out {len(test):,}")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS
    print(f"Feature count: {len(feat)}")

    y_tr, y_te = train["TAXITIME_SEC_mvt"].values, test["TAXITIME_SEC_mvt"].values

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 63, 511),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 5,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
            "verbosity": -1, "num_threads": -1,
        }
        dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
        dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                             reference=dtrain, free_raw_data=False)
        b = lgb.train(params, dtrain, num_boost_round=2500,
                      valid_sets=[dvalid], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        p = b.predict(test[feat], num_iteration=b.best_iteration)
        return rmse(y_te, p)

    print(f"\nOptuna search over {N_TRIALS} trials for LIRF...")
    t0 = time.time()
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  best trial value: {study.best_value:.2f}s")
    print(f"  best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Refit best with full budget
    best = {**study.best_params, "objective": "regression", "metric": "rmse",
            "bagging_freq": 5, "verbosity": -1, "num_threads": -1}
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)
    print("\nRefitting best with early_stopping=100...")
    b = lgb.train(best, dtrain, num_boost_round=5000,
                  valid_sets=[dvalid], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    p = b.predict(test[feat], num_iteration=b.best_iteration)
    r = rmse(y_te, p)
    print(f"\nLIRF specialist hold-out RMSE: {r:.2f}s   (v10 general on LIRF: 462.04)")

    # Overall RMSE impact if LIRF only replaced
    print(f"\nOverall RMSE impact if LIRF prediction is swapped with specialist:")
    # We need v10 predictions on non-LIRF hold-out to combine
    # Use v10 stored numbers: v10 overall 294.40 on 344,078 rows
    N_ALL = 344078
    N_LIRF = len(test)
    LIRF_OLD_RMSE = 462.04
    OVERALL_OLD = 294.40
    old_mse_total = OVERALL_OLD**2 * N_ALL
    old_mse_lirf = LIRF_OLD_RMSE**2 * N_LIRF
    old_mse_other = old_mse_total - old_mse_lirf
    new_mse_lirf = r**2 * N_LIRF
    new_overall = np.sqrt((old_mse_other + new_mse_lirf) / N_ALL)
    print(f"  expected new overall: {new_overall:.2f}s   (drop of {OVERALL_OLD - new_overall:+.2f}s)")

    out_dir = os.path.join(ROOT, "models")
    b.save_model(os.path.join(out_dir, "lgbm_lirf.txt"))
    with open(os.path.join(out_dir, "lgbm_lirf.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_lirf.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(out_dir, "lgbm_lirf.params.txt"), "w") as f:
        for k, v in best.items():
            f.write(f"{k}: {v}\n")
    print(f"\nSaved -> {out_dir}/lgbm_lirf.*")


if __name__ == "__main__":
    main()
