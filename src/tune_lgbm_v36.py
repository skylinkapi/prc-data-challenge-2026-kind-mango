"""v36 tuner: Item 3 of the audit, purified protocol.

Fixes every leak in `tune_lgbm.py` (audit M1):
  - uncensored target: y > 0 only, no `sd` clip
  - feature list = the deployed R_all_v26 set (97 features, no ec_*/opdi_*)
  - train months {2..6,8..10}, blocked stop months {11,12}
  - hold-out months {1,7} excluded from training, stopping and selection;
    they serve one final side-by-side comparison of tuned vs BEST_PARAMS
  - search space adds `linear_tree` on/off and `linear_lambda`

Stages: `build` (build+cache features), `tune` (Optuna on cached frame),
`compare` (refit tuned vs BEST_PARAMS, one-shot hold-out report).
"""
import argparse
import glob
import os
import pickle
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from features_weather import add_weather, TARGET_ICAOS
from features_congestion_v2 import add_congestion_v2, CONG_V2_NUM_COLS
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS
from features_advanced import add_advanced, ADV_NUM_COLS
from features_osm_path import add_osm_path, OSM_PATH_NUM_COLS
from features_turnaround import add_turnaround, TURN_NUM_COLS
from features_disruption import add_disruption, DISR_NUM_COLS
from train_lgbm_v21 import add_obt_features, CAT_COLS, BEST_PARAMS
from train_lgbm_v23 import BASE_NUM_COLS
from train_r_all_v21 import SIGNED_LOG_BASE, SIGNED_LOG_COLS, TEMPERATURE_COLS, signed_log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
CACHE = os.path.join(ROOT, "models", "v36_tune_cache.parquet")
N_TRIALS = int(os.environ.get("TUNE_TRIALS", "24"))
HOLDOUT_MONTHS = {1, 7}
EARLYSTOP_MONTHS = {11, 12}
TRAIN_MONTHS = {2, 3, 4, 5, 6, 8, 9, 10}


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def build_and_cache():
    t0 = time.time()
    print("Building feature frame (purified v26 stack)...")
    frames = [pd.read_parquet(f) for f in sorted(
        glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS) | m["ADES_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[(m["PHASE_mvt"] == "DEP") & m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    sd = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds()
    dep["sched_delay"] = sd
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep = add_obt_features(dep)
    # Uncensored target: keep y > 0 (audit Item 3 / M1).
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]
    for base, col in zip(SIGNED_LOG_BASE, SIGNED_LOG_COLS):
        dep[col] = signed_log(dep[base])

    ctx = m[["ADEP_mvt", "ADES_mvt", "PHASE_mvt", "mvt_ts", "sched_ts",
             "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt",
             "BLOCK_TIME_UTC_mvt"]].copy()
    ctx_adv = m[m["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion_v2(dep, ctx[["ADEP_mvt", "ADES_mvt", "PHASE_mvt", "mvt_ts",
                                      "RUNWAY_mvt", "TAXITIME_SEC_mvt"]])
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol",
                                            "daily_features.parquet"))
    dep = add_advanced(dep, ctx_adv, daily_ec=ec_daily)
    del ctx_adv, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_turnaround(dep, ctx)
    dep = add_disruption(dep, ctx)
    del ctx; gc.collect()

    # Audited encoder recipe: fit on months {2..12} (all except hold-out {1,7}).
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)

    op_cols = operator_numeric_cols()
    feat = (CAT_COLS + BASE_NUM_COLS + SIGNED_LOG_COLS + TEMPERATURE_COLS +
            CONG_V2_NUM_COLS + op_cols +
            TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS +
            TURN_NUM_COLS + DISR_NUM_COLS)
    feat = [c for c in feat if c in dep.columns]

    keep = feat + ["TAXITIME_SEC_mvt", "month", "ADEP_mvt"]
    keep = list(dict.fromkeys(keep))
    out = dep[keep].copy()
    for c in CAT_COLS:
        out[c] = out[c].astype("category")
    out.to_parquet(CACHE)
    with open(CACHE + ".feat.txt", "w") as f:
        f.write("\n".join(feat))
    print(f"  cached {len(out):,} rows, {len(feat)} features "
          f"({time.time()-t0:.1f}s)")


def load_cached():
    dep = pd.read_parquet(CACHE)
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().splitlines()
    return dep, feat


def split_sets(dep):
    train = dep[dep["month"].isin(TRAIN_MONTHS)]
    stop = dep[dep["month"].isin(EARLYSTOP_MONTHS)]
    hold = dep[dep["month"].isin(HOLDOUT_MONTHS)]
    return train, stop, hold


def make_params(trial):
    linear_tree = trial.suggest_categorical("linear_tree", [True, False])
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 64, 440),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 3.0),
        "linear_tree": linear_tree,
        "bagging_freq": 5,
        "verbosity": -1,
        "num_threads": -1,
        "seed": 42, "bagging_seed": 42,
    }
    if linear_tree:
        params["linear_lambda"] = trial.suggest_float("linear_lambda", 1e-4, 10.0, log=True)
    return params


def objective(trial, train, stop, feat):
    params = make_params(trial)
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS, reference=dt, free_raw_data=True)
    b = lgb.train(params, dt, num_boost_round=3000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(60)])
    return b.best_score["valid_0"]["rmse"] if b.best_score else float("inf")


def run_tune(train, stop, feat):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, stop, feat), n_trials=N_TRIALS)
    best = {k: v for k, v in study.best_params.items()}
    fixed = {"objective": "regression", "metric": "rmse", "bagging_freq": 5,
             "verbosity": -1, "num_threads": -1, "seed": 42, "bagging_seed": 42}
    tuned = {**best, **fixed}
    print("\nBest study params:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"  stop RMSE: {study.best_value:.3f}")
    with open(CACHE + ".tuned_params.json", "w") as f:
        import json
        json.dump({"tuned": tuned, "study_best": best,
                   "stop_rmse": study.best_value}, f, indent=2)
    return tuned


def refit_compare(dep, feat, tuned_params):
    train, stop, hold = split_sets(dep)
    label_tr = train["TAXITIME_SEC_mvt"].values
    label_st = stop["TAXITIME_SEC_mvt"].values
    label_ho = hold["TAXITIME_SEC_mvt"].values
    dt = lgb.Dataset(train[feat], label=label_tr,
                     categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(stop[feat], label=label_st,
                     categorical_feature=CAT_COLS, reference=dt, free_raw_data=True)
    clean = (label_ho >= 30) & (label_ho <= 7200)
    results = {}
    for name, params in [("BEST_PARAMS", BEST_PARAMS), ("tuned", tuned_params)]:
        p = {**params,
             "objective": "regression", "metric": "rmse", "bagging_freq": 5,
             "verbosity": -1, "num_threads": -1,
             "seed": 42, "bagging_seed": 42}
        b = lgb.train(p, dt, num_boost_round=5000, valid_sets=[dv],
                      valid_names=["stop"],
                      callbacks=[lgb.early_stopping(100)])
        pred = np.clip(b.predict(hold[feat], num_iteration=b.best_iteration), 0, None)
        results[name] = {
            "best_iter": b.best_iteration,
            "full": rmse(label_ho, pred),
            "clean": rmse(label_ho[clean], pred[clean]),
        }
        print(f"{name}: best_iter {results[name]['best_iter']}  "
              f"hold FULL {results[name]['full']:.3f}  "
              f"CLEAN {results[name]['clean']:.3f}")
    delta = results["BEST_PARAMS"]["clean"] - results["tuned"]["clean"]
    print(f"CLEAN gain of tuned vs BEST_PARAMS: {delta:+.3f} s "
          f"(2-s evidence bar applies, one-shot comparison)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "tune", "compare"])
    args = ap.parse_args()
    if args.stage == "build":
        build_and_cache()
        return
    dep, feat = load_cached()
    if args.stage == "tune":
        train, stop, _ = split_sets(dep)
        run_tune(train, stop, feat)
        return
    import json
    with open(CACHE + ".tuned_params.json") as f:
        tuned = json.load(f)["tuned"]
    refit_compare(dep, feat, tuned)


if __name__ == "__main__":
    main()
