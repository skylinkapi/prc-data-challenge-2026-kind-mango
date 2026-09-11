"""v24 Step 4: 3-seed R_all with honest stop set.

Retrain R_all_v21 recipe 3 times with different seeds. Use a random 12 % of
non-hold-out months as the early-stop set (per doc Section 6: Nov+Dec is the
mix least like Jan+Jul, so the deployed model is under-trained).
"""
import glob, os, pickle, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion_v2 import add_congestion_v2, CONG_V2_NUM_COLS
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS
from features_advanced import add_advanced, ADV_NUM_COLS
from features_osm_path import add_osm_path, OSM_PATH_NUM_COLS
from features_opdi import add_opdi, OPDI_NUM_COLS
from features_opdi_live import add_opdi_live, OPDI_LIVE_NUM_COLS
from features_turnaround import add_turnaround, TURN_NUM_COLS
from features_disruption import add_disruption, DISR_NUM_COLS
from train_lgbm_v21 import add_obt_features, CAT_COLS, BEST_PARAMS
from train_lgbm_v23 import BASE_NUM_COLS
from train_r_all_v21 import SIGNED_LOG_BASE, SIGNED_LOG_COLS, TEMPERATURE_COLS, signed_log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}
STOP_FRAC = 0.12
SEEDS = [42, 43, 44]


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
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
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]
    for base, col in zip(SIGNED_LOG_BASE, SIGNED_LOG_COLS):
        dep[col] = signed_log(dep[base])

    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","sched_ts",
             "RUNWAY_mvt","STAND_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt",
             "BLOCK_TIME_UTC_mvt"]].copy()
    ctx_adv = m[m["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("Feature build (once for all seeds)...")
    t = time.time()
    dep = add_weather(dep)
    dep = add_congestion_v2(dep, ctx[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts",
                                       "RUNWAY_mvt","TAXITIME_SEC_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, ctx_adv, daily_ec=ec_daily)
    del ctx_adv, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    dep = add_turnaround(dep, ctx)
    dep = add_disruption(dep, ctx)
    del ctx; gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  Done in {time.time()-t:.1f}s")

    # honest split: non-holdout rows -> 88% train, 12% stop (random)
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    non_hold = dep.loc[~hold].copy()
    rng = np.random.default_rng(1234)  # fixed for all seeds
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

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = (CAT_COLS + BASE_NUM_COLS + SIGNED_LOG_COLS + TEMPERATURE_COLS +
            CONG_V2_NUM_COLS + ec_cols + op_cols +
            TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS +
            OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS + TURN_NUM_COLS + DISR_NUM_COLS)
    feat = [c for c in feat if c in train.columns]
    print(f"Feature count: {len(feat)}")

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    y_st = stop["TAXITIME_SEC_mvt"].values.astype(float)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)

    dt = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(stop[feat], label=y_st, categorical_feature=CAT_COLS,
                     reference=dt, free_raw_data=True)

    preds = []
    for seed in SEEDS:
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
        b.save_model(os.path.join(ROOT, "models", f"lgbm_r_all_v24_s{seed}.txt"))
        p = np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None)
        preds.append(p)
        del b; gc.collect()

    with open(os.path.join(ROOT, "models", "lgbm_r_all_v24.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(ROOT, "models", "lgbm_r_all_v24.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    # Report
    print("\n=== Per-seed and mean ===")
    clean = (y_te >= 30) & (y_te <= 7200)
    for s, p in zip(SEEDS, preds):
        print(f"  seed {s}:  FULL {rmse(y_te, p):.2f}   CLEAN {rmse(y_te[clean], p[clean]):.2f}")
    mean = np.mean(preds, axis=0)
    print(f"  MEAN   :  FULL {rmse(y_te, mean):.2f}   CLEAN {rmse(y_te[clean], mean[clean]):.2f}")
    print(f"\nTotal wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
