"""v26 = v24 recipe WITHOUT ec_* features (Step 5 correct).

Trains 3 seeds with random 12% honest stop. Drops the 38 ec_* columns and the
18 opdi_* columns per doc Section 7 (dead weight + coverage collapse).

`--plan` adds the ARVT_1 planned-time features and writes `lgbm_r_all_v38_*` files plus
a paired hold-out report against the shipped v26 members (docs/MODEL_ANALYSIS.md section 4).
"""
import argparse, glob, json, os, pickle, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion_v2 import add_congestion_v2, CONG_V2_NUM_COLS
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS
from features_advanced import add_advanced, ADV_NUM_COLS
from features_osm_path import add_osm_path, OSM_PATH_NUM_COLS
from features_turnaround import add_turnaround, TURN_NUM_COLS
from features_disruption import add_disruption, DISR_NUM_COLS
from features_plan import add_plan_times, fit_route_medians, add_plan_residual, PLAN_NUM_COLS
from train_lgbm_v21 import add_obt_features, CAT_COLS, BEST_PARAMS
from train_lgbm_v23 import BASE_NUM_COLS
from train_r_all_v21 import SIGNED_LOG_BASE, SIGNED_LOG_COLS, TEMPERATURE_COLS, signed_log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}
STOP_FRAC = 0.12
DEFAULT_SEEDS = [42, 43, 44]
SHIPPED_TAG = "v26"
PLAN_TAG = "v38"


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def compare_to_shipped(test, mean, tag):
    """Paired hold-out report of the new mean against the shipped v26 3-member mean."""
    with open(os.path.join(MODELS, f"lgbm_r_all_{SHIPPED_TAG}.features.txt")) as f:
        feat_old = f.read().splitlines()
    old = np.mean([np.clip(lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_all_{SHIPPED_TAG}_s{s}.txt"))
                           .predict(test[feat_old]), 0, None) for s in DEFAULT_SEEDS], axis=0)
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    apt = test["ADEP_mvt"].astype(str).values
    clean = (y >= 30) & (y <= 7200)
    report = {"clean_new": rmse(y[clean], mean[clean]), "clean_old": rmse(y[clean], old[clean]),
              "full_new": rmse(y, mean), "full_old": rmse(y, old),
              "clean_delta_by_airport": {a: rmse(y[clean & (apt == a)], mean[clean & (apt == a)]) -
                                            rmse(y[clean & (apt == a)], old[clean & (apt == a)])
                                         for a in sorted(set(apt))}}
    report["clean_delta"] = report["clean_new"] - report["clean_old"]
    print(json.dumps(report, indent=1))
    with open(os.path.join(MODELS, f"lgbm_r_all_{tag}.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)


def main(seeds=None, plan=False):
    seeds = seeds or DEFAULT_SEEDS
    tag = PLAN_TAG if plan else SHIPPED_TAG
    write_shared = set(seeds) == set(DEFAULT_SEEDS)
    t_all = time.time()
    print("Loading...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
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
    if plan:
        dep = add_plan_times(dep)
        route_medians = fit_route_medians(dep[~dep["month"].isin(HOLDOUT_MONTHS)])
        dep = add_plan_residual(dep, route_medians)
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]
    for base, col in zip(SIGNED_LOG_BASE, SIGNED_LOG_COLS):
        dep[col] = signed_log(dep[base])

    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","sched_ts",
             "RUNWAY_mvt","STAND_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt",
             "BLOCK_TIME_UTC_mvt"]].copy()
    ctx_adv = m[m["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("Feature build (no ec_*, no opdi_*)...")
    t = time.time()
    dep = add_weather(dep)
    dep = add_congestion_v2(dep, ctx[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts",
                                       "RUNWAY_mvt","TAXITIME_SEC_mvt"]])
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, ctx_adv, daily_ec=ec_daily)  # keeps ades_arr_atfm_delay_today
    del ctx_adv, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_turnaround(dep, ctx)
    dep = add_disruption(dep, ctx)
    del ctx; gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  Done in {time.time()-t:.1f}s")

    non_hold = dep.loc[~hold].copy()
    rng = np.random.default_rng(1234)
    stop_mask_local = rng.random(len(non_hold)) < STOP_FRAC
    train = non_hold[~stop_mask_local].copy()
    stop = non_hold[stop_mask_local].copy()
    test = dep.loc[hold].copy()
    del non_hold, dep; gc.collect()
    print(f"Train {len(train):,}   Stop (random 12%) {len(stop):,}   Test {len(test):,}")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        stop[c] = pd.Categorical(stop[c], categories=train[c].cat.categories)
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    op_cols = operator_numeric_cols()
    feat = (CAT_COLS + BASE_NUM_COLS + SIGNED_LOG_COLS + TEMPERATURE_COLS +
            CONG_V2_NUM_COLS + op_cols +
            TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS +
            TURN_NUM_COLS + DISR_NUM_COLS + (PLAN_NUM_COLS if plan else []))
    feat = [c for c in feat if c in train.columns]
    print(f"Feature count: {len(feat)}  (v24 had 143; dropped ec_* + opdi_*)")

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    y_st = stop["TAXITIME_SEC_mvt"].values.astype(float)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)

    dt = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(stop[feat], label=y_st, categorical_feature=CAT_COLS,
                     reference=dt, free_raw_data=True)

    preds = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        params = {**BEST_PARAMS,
                  "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0,
                  "num_leaves": 220, "seed": seed, "bagging_seed": seed,
                  "bagging_freq": 5, "verbosity": -1, "num_threads": -1}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["stop"],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(400)])
        print(f"  best iter {b.best_iteration}  time {time.time()-t0:.1f}s")
        b.save_model(os.path.join(MODELS, f"lgbm_r_all_{tag}_s{seed}.txt"))
        p = np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None)
        preds.append(p)
        del b; gc.collect()

    if write_shared:
        with open(os.path.join(MODELS, f"lgbm_r_all_{tag}.features.txt"), "w") as f:
            f.write("\n".join(feat))
        with open(os.path.join(MODELS, f"lgbm_r_all_{tag}.encoders.pkl"), "wb") as f:
            pickle.dump(encoders, f)
        if plan:
            route_medians.rename("plan_block_median").to_frame().to_parquet(
                os.path.join(MODELS, f"lgbm_r_all_{tag}.route_medians.parquet"))

    print("\n=== Per-seed and mean ===")
    clean = (y_te >= 30) & (y_te <= 7200)
    for s, p in zip(seeds, preds):
        print(f"  seed {s}:  FULL {rmse(y_te, p):.2f}   CLEAN {rmse(y_te[clean], p[clean]):.2f}")
    mean = np.mean(preds, axis=0)
    print(f"  MEAN   :  FULL {rmse(y_te, mean):.2f}   CLEAN {rmse(y_te[clean], mean[clean]):.2f}")
    print(f"  v24 baseline: FULL 396.75   CLEAN 266.23")
    if plan and write_shared:
        compare_to_shipped(test, mean, tag)
    print(f"\nTotal wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the base regressor members.")
    ap.add_argument("seeds", nargs="*", type=int, help="member seeds; default 42 43 44")
    ap.add_argument("--plan", action="store_true", help="add the ARVT_1 planned-time features (v38)")
    args = ap.parse_args()
    main(seeds=args.seeds or None, plan=args.plan)
