"""lgbm_v20 = v15 recipe + aggressive data cleanup.

Cleanup applied (in order):
  1. Fill missing AIRCRAFT_TYPE_mvt from AIRCRAFT_TYPE_flt where available
  2. Drop rows with suspicious runway strings ('NA', '-', '?', 'N/A', '')
  3. Drop rows with suspicious aircraft types ('ZZZZ', 'GLID', 'UNKN', '', '-')
  4. Bucket stands with < 10 flights per airport as 'OTHER'
  5. Bucket aircraft types with < 5 flights as 'OTHER'
  6. Drop rows where BLOCK > MVT (impossible ordering)
  7. Drop rows with |sched_delay| > 21600 (>6h drift)
  8. Existing BLOCK==SCHED filter (v6+)
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

SUSPICIOUS_RUNWAYS = {"NA", "-", "?", "N/A", "", "NAN"}
SUSPICIOUS_TYPES = {"ZZZZ", "GLID", "UNKN", "", "-", "NAN"}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def clean(dep: pd.DataFrame) -> pd.DataFrame:
    """Apply the eight cleanup steps. Returns cleaned copy + report."""
    n_start = len(dep)

    # 1. Fill AIRCRAFT_TYPE_mvt from _flt where possible
    mask_fill = dep["AIRCRAFT_TYPE_mvt"].isna() & dep["AIRCRAFT_TYPE_flt"].notna()
    dep.loc[mask_fill, "AIRCRAFT_TYPE_mvt"] = dep.loc[mask_fill, "AIRCRAFT_TYPE_flt"]
    print(f"  1. Filled AIRCRAFT_TYPE_mvt from _flt: {mask_fill.sum():,}")

    # 2. Drop suspicious runway strings
    rwy_upper = dep["RUNWAY_mvt"].astype(str).str.strip().str.upper()
    bad_rwy = rwy_upper.isin(SUSPICIOUS_RUNWAYS)
    n_bad = bad_rwy.sum()
    dep = dep[~bad_rwy]
    print(f"  2. Dropped suspicious runways: {n_bad:,}")

    # 3. Drop suspicious aircraft types
    typ_upper = dep["AIRCRAFT_TYPE_mvt"].astype(str).str.strip().str.upper()
    bad_typ = typ_upper.isin(SUSPICIOUS_TYPES)
    n_bad = bad_typ.sum()
    dep = dep[~bad_typ]
    print(f"  3. Dropped suspicious types: {n_bad:,}")

    # 4. Bucket rare stands (< 10 flights per airport) as OTHER
    stand_counts = dep.groupby(["ADEP_mvt", "STAND_mvt"]).size()
    rare_pairs = set(stand_counts[stand_counts < 10].index)
    is_rare_stand = dep.apply(lambda r: (r["ADEP_mvt"], r["STAND_mvt"]) in rare_pairs, axis=1)
    n_rare = is_rare_stand.sum()
    dep.loc[is_rare_stand, "STAND_mvt"] = "OTHER"
    print(f"  4. Bucketed rare stands as OTHER: {n_rare:,}")

    # 5. Bucket rare aircraft types (< 5 flights globally) as OTHER
    type_counts = dep["AIRCRAFT_TYPE_mvt"].value_counts()
    rare_types = set(type_counts[type_counts < 5].index)
    is_rare_type = dep["AIRCRAFT_TYPE_mvt"].isin(rare_types)
    n_rare_t = is_rare_type.sum()
    dep.loc[is_rare_type, "AIRCRAFT_TYPE_mvt"] = "OTHER_TYPE"
    print(f"  5. Bucketed rare aircraft types as OTHER_TYPE: {n_rare_t:,}")

    # 6. Drop BLOCK > MVT ordering violations
    bad_order = (dep["mvt_ts"] < dep["blk_ts"]) & dep["mvt_ts"].notna() & dep["blk_ts"].notna()
    n_bad = bad_order.sum()
    dep = dep[~bad_order]
    print(f"  6. Dropped BLOCK > MVT: {n_bad:,}")

    # 7. Drop extreme sched_delay
    bad_sd = ~dep["sched_delay"].between(-1800, 21600)
    n_bad = bad_sd.sum()
    dep = dep[~bad_sd]
    print(f"  7. Dropped |sd|>range: {n_bad:,}")

    n_end = len(dep)
    print(f"\n  Total: {n_start:,} -> {n_end:,}  (removed {n_start-n_end:,}, {(n_start-n_end)/n_start*100:.2f}%)")
    return dep


def main():
    t0 = time.time()
    print("Loading...")
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

    print(f"\nApplying cleanup:")
    dep = clean(dep)

    m_ctx = m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("\nAdding features...")
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
    train_all = dep.loc[~hold]
    n_bad = train_all["_qual_bad"].sum()
    train = train_all[~train_all["_qual_bad"]].copy()
    test = dep.loc[hold].copy()
    print(f"Filtered {n_bad:,} BLOCK==SCHED. Train {len(train):,}   Hold-out {len(test):,}")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS + OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS
    print(f"Feature count: {len(feat)}  (same as v15)")

    y_tr, y_te = train["TAXITIME_SEC_mvt"].values, test["TAXITIME_SEC_mvt"].values
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=True)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining v20...")
    t0 = time.time()
    b = lgb.train(params, dtrain, num_boost_round=5000,
                  valid_sets=[dvalid], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {b.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = b.predict(test[feat], num_iteration=b.best_iteration)
    print(f"\nLGBM v20 hold-out RMSE: {rmse(y_te, p_te):.2f}s   (v15 = 293.09)")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].astype(str).values, "y": y_te, "p": p_te}) \
        .groupby("apt").apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(2)
    v15 = pd.Series({"LEMD":204.61,"LSZH":213.31,"LEBL":237.37,"EDDF":245.31,"EDDM":241.38,
                     "EHAM":248.83,"LTFM":292.85,"LFPG":329.60,"EGLL":359.21,"LIRF":461.63})
    tab["v15"] = v15
    tab["delta"] = (tab["rmse"] - tab["v15"]).round(2)
    print("\nPer-airport RMSE (v20 vs v15):")
    print(tab.sort_values("rmse").to_string())

    out_dir = os.path.join(ROOT, "models")
    b.save_model(os.path.join(out_dir, "lgbm_v20.txt"))
    with open(os.path.join(out_dir, "lgbm_v20.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_v20.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {out_dir}/lgbm_v20.*")


if __name__ == "__main__":
    main()
