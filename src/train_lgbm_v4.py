"""LightGBM v4: v3 features + physical stand-to-runway distance from OSM."""
import glob
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS

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
               "dep_same_rwy_prev_60m", "dep_queue_next_10m"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    t0 = time.time()
    print("Loading...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    print(f"  {len(m):,} in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Building features...")
    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep = dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)]
    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  {len(dep):,} DEP rows in {time.time()-t0:.1f}s")

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train, test = dep.loc[~hold].copy(), dep.loc[hold].copy()
    print(f"Train {len(train):,}   Hold-out {len(test):,}")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS
    print(f"Feature count: {len(feat)}  (v3 had 91)")

    y_tr, y_te = train["TAXITIME_SEC_mvt"].values, test["TAXITIME_SEC_mvt"].values
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.05, "num_leaves": 255,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
        "lambda_l2": 1.0, "verbosity": -1, "num_threads": -1,
    }

    print("\nTraining LightGBM v4...")
    t0 = time.time()
    booster = lgb.train(
        params, dtrain, num_boost_round=3000,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    print(f"  best iter: {booster.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = booster.predict(test[feat], num_iteration=booster.best_iteration)
    print(f"\nLGBM v4 overall RMSE: {rmse(y_te, p_te):.1f}s   (v3 = 298.6)")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].values, "y": y_te, "p": p_te}) \
        .groupby("apt", observed=True).apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(1)
    v3 = pd.Series({"LEMD":210.2,"LSZH":231.7,"LEBL":243.3,"EDDM":247.4,"EDDF":248.4,
                    "EHAM":252.9,"LTFM":296.5,"LFPG":335.8,"EGLL":365.1,"LIRF":466.5})
    tab["v3"] = v3
    tab["delta"] = (tab["rmse"] - tab["v3"]).round(1)
    print("\nPer-airport RMSE (v4 vs v3):")
    print(tab.sort_values("rmse").to_string())

    print("\nTop-25 features by gain:")
    imp = pd.Series(booster.feature_importance(importance_type="gain"),
                    index=feat).sort_values(ascending=False)
    print(imp.head(25).round(0).to_string())

    print(f"\nRank of taxi_dist_m: {list(imp.index).index('taxi_dist_m')+1} of {len(feat)}")

    out_dir = os.path.join(ROOT, "models")
    booster.save_model(os.path.join(out_dir, "lgbm_v4.txt"))
    with open(os.path.join(out_dir, "lgbm_v4.features.txt"), "w") as f:
        f.write("\n".join(feat))
    import pickle
    with open(os.path.join(out_dir, "lgbm_v4.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {out_dir}/lgbm_v4.*")


if __name__ == "__main__":
    main()
