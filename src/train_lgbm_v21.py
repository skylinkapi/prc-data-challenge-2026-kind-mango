"""lgbm_v21 = v15 with the label/feature fixes from MODEL_ANALYSIS.md step 2.

Changes vs v15:
  - Keep all rows with y > 0 (no `between(30, 7200)` filter)
  - Keep BLOCK==SCHED rows (no `_qual_bad` filter)
  - Unclipped `sched_delay`
  - New features: `sd_mod_86400`, `mvt_eobt1`, `mvt_iobt`, `eobt1_sched`,
                   `eobt1_iobt`, `flt_id_null`
  - Report clean AND full hold-out RMSE
"""
import glob
import os
import pickle
import time
import gc
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]

# Note: sched_delay is UNCLIPPED here; sd_mod_86400 exposes 24h-offset rows.
# New OBT-delta features have very high correlation with y (see section 3.2).
NUM_COLS_V21 = ["hour", "dow", "month", "sched_delay", "sd_mod_86400",
                "mvt_eobt1", "mvt_iobt", "eobt1_sched", "eobt1_iobt",
                "sknt", "gust_kt", "wind_cross_kt", "wind_head_kt",
                "vis_km", "low_vis", "very_low_vis",
                "ceiling_ft", "low_ceiling",
                "wx_precip", "wx_snow", "wx_thunder", "wx_freezing",
                "dep_load_prev_15m", "dep_load_prev_30m", "dep_load_prev_60m",
                "arr_load_prev_15m", "arr_load_prev_30m", "arr_load_prev_60m",
                "dep_same_rwy_prev_15m", "dep_same_rwy_prev_30m",
                "dep_same_rwy_prev_60m", "dep_queue_next_10m",
                "flt_null", "flt_id_null"]

BEST_PARAMS = {
    "learning_rate": 0.022810159868114487,
    "num_leaves": 440,
    "min_data_in_leaf": 76,
    "feature_fraction": 0.6737480156722359,
    "bagging_fraction": 0.9366340145580665,
    "lambda_l1": 0.0887022084743886,
    "lambda_l2": 0.3423614383968606,
    "min_gain_to_split": 1.8309877581737415,
}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def add_obt_features(dep: pd.DataFrame) -> pd.DataFrame:
    """The five OBT-delta features from MODEL_ANALYSIS.md section 3.2."""
    dep["eobt1_ts"] = pd.to_datetime(dep["EOBT_1_flt"], errors="coerce")
    dep["iobt_ts"]  = pd.to_datetime(dep["IOBT_flt"],  errors="coerce")
    to_s = lambda a, b: (a - b).dt.total_seconds()
    dep["mvt_eobt1"]   = to_s(dep["mvt_ts"], dep["eobt1_ts"])
    dep["mvt_iobt"]    = to_s(dep["mvt_ts"], dep["iobt_ts"])
    dep["eobt1_sched"] = to_s(dep["eobt1_ts"], dep["sched_ts"])
    dep["eobt1_iobt"]  = to_s(dep["eobt1_ts"], dep["iobt_ts"])
    dep["flt_id_null"] = dep["FLIGHT_ID_mvt"].isna().astype(np.int8)
    return dep


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
    # UNCLIPPED sched_delay (was clip(-1800, 3600) in v15).
    sd = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds()
    dep["sched_delay"] = sd
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep = add_obt_features(dep)
    # Keep only rows with a valid positive target.
    y_raw = dep["TAXITIME_SEC_mvt"].astype(float)
    dep = dep[y_raw > 0]
    print(f"DEP rows kept: {len(dep):,}   with y>7200: {(dep['TAXITIME_SEC_mvt']>7200).sum():,}   with y>80000: {(dep['TAXITIME_SEC_mvt']>80000).sum():,}")

    m_ctx = m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("Adding features...")
    t = time.time()
    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
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
    print(f"Train {len(train):,}   Hold-out (full) {len(test):,}")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V21 + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS + OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS
    print(f"Feature count: {len(feat)}  (v15 had 124, +6 = 130)")

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=True)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining v21...")
    t0 = time.time()
    b = lgb.train(params, dtrain, num_boost_round=5000,
                  valid_sets=[dvalid], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {b.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = b.predict(test[feat], num_iteration=b.best_iteration)
    p_pos = np.clip(p_te, 0, None)  # step 2: no upper clip

    print(f"\n=== v21 hold-out RMSE ===")
    print(f"FULL   (all rows):        {rmse(y_te, p_pos):>7.2f} s   n={len(y_te):,}")
    clean = (y_te >= 30) & (y_te <= 7200)
    print(f"CLEAN  (30 <= y <= 7200): {rmse(y_te[clean], p_pos[clean]):>7.2f} s   n={clean.sum():,}")
    tail = y_te > 7200
    print(f"TAIL   (y > 7200):        {rmse(y_te[tail], p_pos[tail]):>7.2f} s   n={tail.sum():,}   mean_y {y_te[tail].mean():.0f}   mean_p {p_pos[tail].mean():.0f}")

    print("\nPer-airport (full):")
    tab = pd.DataFrame({"apt": test["ADEP_mvt"].astype(str).values, "y": y_te, "p": p_pos}) \
        .groupby("apt").apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(2)
    print(tab.sort_values("rmse").to_string())

    # Feature importance for the new columns
    imp = pd.Series(b.feature_importance(importance_type="gain"),
                    index=feat).sort_values(ascending=False)
    print("\nNew-feature ranks:")
    for c in ["sched_delay", "sd_mod_86400", "mvt_eobt1", "mvt_iobt",
              "eobt1_sched", "eobt1_iobt", "flt_id_null"]:
        r = list(imp.index).index(c) + 1
        print(f"  {c:15s} rank {r:>3d} / {len(feat)}   gain {imp[c]:.2e}")

    out_dir = os.path.join(ROOT, "models")
    b.save_model(os.path.join(out_dir, "lgbm_v21.txt"))
    with open(os.path.join(out_dir, "lgbm_v21.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_v21.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {out_dir}/lgbm_v21.*   total wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
