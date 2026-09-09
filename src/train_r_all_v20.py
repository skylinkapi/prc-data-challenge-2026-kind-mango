"""Train R_all for v20: v21 stack + turnaround + disruption meters.

Per doc Section 5.3:
  - R_all: all rows (no fallback filter)
  - Optuna re-run optional; skip for now, use v21's Optuna hyperparams
  - Early stop on Nov+Dec (honest split; the current pipeline uses Jan+Jul).
    Keep Jan+Jul as final test.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}   # test only
EARLYSTOP_MONTHS = {11, 12}
TRAIN_MONTHS = {2, 3, 4, 5, 6, 8, 9, 10}


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

    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","sched_ts",
             "RUNWAY_mvt","STAND_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt"]].copy()
    ctx_adv = m[m["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("Features (v21 stack + turnaround + disruption)...")
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
    print(f"  Feature build done in {time.time()-t:.1f}s")

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    stop = dep["month"].isin(EARLYSTOP_MONTHS)
    train_only = dep["month"].isin(TRAIN_MONTHS)

    train = dep.loc[train_only].copy()
    valid = dep.loc[stop].copy()
    test = dep.loc[hold].copy()
    print(f"Train {len(train):,}   EarlyStop {len(valid):,}   Test {len(test):,}")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        valid[c] = pd.Categorical(valid[c], categories=train[c].cat.categories)
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = (CAT_COLS + BASE_NUM_COLS + CONG_V2_NUM_COLS + ec_cols + op_cols +
            TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS +
            OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS + TURN_NUM_COLS + DISR_NUM_COLS)
    print(f"Feature count: {len(feat)}  (v24 had 127; +{len(TURN_NUM_COLS)+len(DISR_NUM_COLS)} new)")

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    y_va = valid["TAXITIME_SEC_mvt"].values.astype(float)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)
    dt = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(valid[feat], label=y_va, categorical_feature=CAT_COLS,
                     reference=dt, free_raw_data=True)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining R_all (honest early stop on Nov+Dec)...")
    t0 = time.time()
    b = lgb.train(params, dt, num_boost_round=5000,
                  valid_sets=[dv], valid_names=["stop"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {b.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None)

    print(f"\n=== R_all Jan+Jul evaluation (test set, one shot) ===")
    print(f"FULL   RMSE: {rmse(y_te, p_te):.2f}")
    clean = (y_te >= 30) & (y_te <= 7200)
    print(f"CLEAN  RMSE: {rmse(y_te[clean], p_te[clean]):.2f}   n={clean.sum():,}   (target 270)")

    print("\nPer-airport clean RMSE (v21 baseline in parens):")
    v21 = {"LIRF":565.10,"EGLL":303.20,"LFPG":289.60,"LTFM":271.80,"LEBL":230.50,
           "EDDF":225.90,"LSZH":221.60,"EHAM":215.30,"EDDM":211.70,"LEMD":191.60}
    apt_te = test["ADEP_mvt"].astype(str).values
    for apt in sorted(v21, key=lambda k: -v21[k]):
        mask = (apt_te == apt) & clean
        if mask.sum() > 0:
            r = rmse(y_te[mask], p_te[mask])
            print(f"  {apt}: R_all {r:>6.2f}   v21 {v21[apt]:>6.2f}   delta {r-v21[apt]:+.2f}")

    imp = pd.Series(b.feature_importance(importance_type="gain"),
                    index=feat).sort_values(ascending=False)
    print("\nNew feature ranks:")
    for c in TURN_NUM_COLS + DISR_NUM_COLS:
        r = list(imp.index).index(c) + 1
        print(f"  {c:22s} rank {r:>3d}/{len(feat)}  gain {imp[c]:.2e}")

    out_dir = os.path.join(ROOT, "models")
    b.save_model(os.path.join(out_dir, "lgbm_r_all_v20.txt"))
    with open(os.path.join(out_dir, "lgbm_r_all_v20.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_r_all_v20.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {out_dir}/lgbm_r_all_v20.*   total wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
