"""lgbm_v10 = v9 recipe + 3 OSM real-path features:
  taxi_path_length_m, n_turns, path_vs_haversine
"""
import glob
import os
import pickle
import time
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


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


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

    print("Adding features (weather + congestion + EC + operator + distance + advanced + OSM path)...")
    t = time.time()
    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]],
                       daily_ec=ec_daily)
    dep = add_osm_path(dep)

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
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS
    print(f"Feature count: {len(feat)}  (v9 had 103)")

    print("OSM path feature coverage:")
    for c in OSM_PATH_NUM_COLS:
        print(f"  {c:30s} train {train[c].notna().mean()*100:5.1f}%   test {test[c].notna().mean()*100:5.1f}%")

    y_tr, y_te = train["TAXITIME_SEC_mvt"].values, test["TAXITIME_SEC_mvt"].values
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining v10...")
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=5000,
                        valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {booster.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = booster.predict(test[feat], num_iteration=booster.best_iteration)
    print(f"\nLGBM v10 hold-out RMSE: {rmse(y_te, p_te):.2f}s   (v9 = 294.58)")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].astype(str).values, "y": y_te, "p": p_te}) \
        .groupby("apt").apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(2)
    v9 = pd.Series({"LEMD":205.22,"LSZH":227.27,"LEBL":239.85,"EDDF":244.44,"EDDM":245.37,
                    "EHAM":250.66,"LTFM":293.40,"LFPG":329.88,"EGLL":360.04,"LIRF":461.70})
    tab["v9"] = v9
    tab["delta"] = (tab["rmse"] - tab["v9"]).round(2)
    print("\nPer-airport RMSE (v10 vs v9):")
    print(tab.sort_values("rmse").to_string())

    print("\nOSM-path feature ranks (of 106 features):")
    imp = pd.Series(booster.feature_importance(importance_type="gain"),
                    index=feat).sort_values(ascending=False)
    for c in OSM_PATH_NUM_COLS:
        r = list(imp.index).index(c) + 1
        print(f"  {c:30s} rank {r:3d}  gain {imp[c]:.2e}")

    out_dir = os.path.join(ROOT, "models")
    booster.save_model(os.path.join(out_dir, "lgbm_v10.txt"))
    with open(os.path.join(out_dir, "lgbm_v10.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_v10.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {out_dir}/lgbm_v10.*")


if __name__ == "__main__":
    main()
