"""v27 Step 3: 7-seed R_all with feature_fraction_seed varied.

Doc measured -1.99 s CLEAN going 3 -> 7 members with proper seed variation.
Rejects any member whose best_iter < half the median of members (Section 7.3).
Also drops 3 dead features per Section 4.5:
  - ice_accretion_1hr (0% coverage both years)
  - ades_arr_atfm_delay_today (last ec-file input, ends the outage risk)
"""
import glob, os, pickle, time, gc
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
from train_lgbm_v21 import add_obt_features, CAT_COLS, BEST_PARAMS
from train_lgbm_v23 import BASE_NUM_COLS
from train_r_all_v21 import SIGNED_LOG_BASE, SIGNED_LOG_COLS, TEMPERATURE_COLS, signed_log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}
STOP_FRAC = 0.12
SEEDS_V27 = [42, 43, 44, 45, 46, 47, 48]  # 7 members per doc 7.3

# Doc Section 4.5: drop dead features
DROP_FEATURES = {"ice_accretion_1hr", "ades_arr_atfm_delay_today", "taxi_dist_known"}


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

    print("Feature build (drop ec_*/opdi_*, drop 3 dead features)...")
    t = time.time()
    dep = add_weather(dep)
    dep = add_congestion_v2(dep, ctx[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts",
                                       "RUNWAY_mvt","TAXITIME_SEC_mvt"]])
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, ctx_adv, daily_ec=ec_daily)
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
            TURN_NUM_COLS + DISR_NUM_COLS)
    feat = [c for c in feat if c in train.columns and c not in DROP_FEATURES]
    print(f"Feature count: {len(feat)}  (v26 had 105; dropped {len(DROP_FEATURES & set(train.columns))} dead)")

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    y_st = stop["TAXITIME_SEC_mvt"].values.astype(float)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)

    dt = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(stop[feat], label=y_st, categorical_feature=CAT_COLS,
                     reference=dt, free_raw_data=True)

    preds = {}
    best_iters = {}
    for seed in SEEDS_V27:
        print(f"\n=== Seed {seed} ===")
        params = {**BEST_PARAMS,
                  "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0,
                  "num_leaves": 220,
                  "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
                  "bagging_freq": 5, "verbosity": -1, "num_threads": -1}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["stop"],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
        print(f"  best iter {b.best_iteration}  time {time.time()-t0:.1f}s")
        b.save_model(os.path.join(ROOT, "models", f"lgbm_r_all_v27_s{seed}.txt"))
        p = np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None)
        preds[seed] = p
        best_iters[seed] = b.best_iteration
        del b; gc.collect()

    # Drop under-trained members (best_iter < half median)
    med_iter = int(np.median(list(best_iters.values())))
    threshold = med_iter // 2
    kept = [s for s in SEEDS_V27 if best_iters[s] >= threshold]
    dropped = [s for s in SEEDS_V27 if best_iters[s] < threshold]
    print(f"\nMedian best_iter {med_iter}, threshold {threshold}")
    print(f"Kept seeds: {kept}   Dropped (under-trained): {dropped}")

    with open(os.path.join(ROOT, "models", "lgbm_r_all_v27.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(ROOT, "models", "lgbm_r_all_v27.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(ROOT, "models", "lgbm_r_all_v27.kept_seeds.txt"), "w") as f:
        f.write(",".join(str(s) for s in kept))

    print("\n=== Per-seed and mean ===")
    clean = (y_te >= 30) & (y_te <= 7200)
    for s in SEEDS_V27:
        p = preds[s]
        marker = "*" if s in kept else "!"
        print(f"  seed {s} {marker}: FULL {rmse(y_te, p):.2f}   CLEAN {rmse(y_te[clean], p[clean]):.2f}   iter {best_iters[s]}")
    mean_kept = np.mean([preds[s] for s in kept], axis=0)
    print(f"  MEAN({len(kept)} kept) : FULL {rmse(y_te, mean_kept):.2f}   CLEAN {rmse(y_te[clean], mean_kept[clean]):.2f}")
    print(f"  v24 3-seed baseline: FULL 396.75   CLEAN 266.23")
    print(f"  v26 3-seed baseline: FULL 392.33   CLEAN 266.46")
    print(f"\nTotal wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
