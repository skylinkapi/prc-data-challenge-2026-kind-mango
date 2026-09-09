"""Section 6.2 step 1: R_norm regressor.

v21 recipe trained on non-fallback, non-24h rows only:
  |y - sd| >= 60  AND  y < 80,000

Report clean RMSE per airport. LIRF must fall well below 565 s.
"""
import glob, os, pickle, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS
from features_advanced import add_advanced, ADV_NUM_COLS
from features_osm_path import add_osm_path, OSM_PATH_NUM_COLS
from features_opdi import add_opdi, OPDI_NUM_COLS
from features_opdi_live import add_opdi_live, OPDI_LIVE_NUM_COLS
from train_lgbm_v21 import add_obt_features, CAT_COLS, NUM_COLS_V21, BEST_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    t_all = time.time()
    print("Loading...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    sd = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds()
    dep["sched_delay"] = sd
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep = add_obt_features(dep)
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    dep = dep[y > 0]
    # R_norm filter: exclude fallback rows and 24h rows
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    sd_v = dep["sched_delay"].astype(float)
    is_fb = (y - sd_v).abs() < 60
    is_24h = y > 80000
    keep_norm = ~is_fb & ~is_24h
    n_before = len(dep)
    dep = dep[keep_norm]
    print(f"R_norm filter: kept {len(dep):,} / {n_before:,}  (removed {(~keep_norm).sum():,}: fb {is_fb.sum():,}, 24h {is_24h.sum():,})")

    m_ctx = m[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("Adding features...")
    t = time.time()
    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m_ctx, daily_ec=ec_daily)
    del m_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  Done in {time.time()-t:.1f}s")

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train = dep.loc[~hold].copy()
    test = dep.loc[hold].copy()
    print(f"Train {len(train):,}   Hold-out {len(test):,}")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V21 + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS + OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dvalid = lgb.Dataset(test[feat], label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=True)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining R_norm...")
    t0 = time.time()
    b = lgb.train(params, dtrain, num_boost_round=5000,
                  valid_sets=[dvalid], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {b.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = b.predict(test[feat], num_iteration=b.best_iteration)
    p_pos = np.clip(p_te, 0, None)

    print(f"\n=== R_norm hold-out (all non-fallback rows) ===")
    print(f"FULL  RMSE: {rmse(y_te, p_pos):>7.2f}   n={len(y_te):,}")
    clean = (y_te >= 30) & (y_te <= 7200)
    print(f"CLEAN RMSE: {rmse(y_te[clean], p_pos[clean]):>7.2f}   n={clean.sum():,}")

    print("\nPer-airport clean RMSE (v21 in parens):")
    v21_rmse = {"LIRF": 565.1, "EGLL": 303.2, "LFPG": 289.6, "LTFM": 271.8,
                "LEBL": 230.5, "EDDF": 225.9, "LSZH": 221.6, "EHAM": 215.3,
                "EDDM": 211.7, "LEMD": 191.6}
    apt_te = test["ADEP_mvt"].astype(str).values
    for apt in sorted(v21_rmse, key=lambda k: -v21_rmse[k]):
        mask = (apt_te == apt) & clean
        if mask.sum() > 0:
            r = rmse(y_te[mask], p_pos[mask])
            v21 = v21_rmse[apt]
            print(f"  {apt}: R_norm {r:>6.2f}   v21 {v21:>6.2f}   delta {r-v21:+.2f}")

    out_dir = os.path.join(ROOT, "models")
    b.save_model(os.path.join(out_dir, "lgbm_r_norm.txt"))
    with open(os.path.join(out_dir, "lgbm_r_norm.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_r_norm.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {out_dir}/lgbm_r_norm.*   total wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
