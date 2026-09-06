"""lgbm_v13 = v11 features + P_tail as a regressor input.

The tail classifier's calibrated probability is added as ONE feature.
LightGBM decides how to use it (blended with airport/operator/etc.).
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS_DIR = os.path.join(ROOT, "models")
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
               "flt_null", "p_tail"]

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

    m_ctx = m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("Adding features...")
    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m_ctx, daily_ec=ec_daily)
    del m_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    gc.collect()

    # Predict P_tail with the classifier
    print("Predicting P_tail with classifier...")
    tc = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lgbm_tail_classifier.txt"))
    with open(os.path.join(MODELS_DIR, "lgbm_tail_classifier.features.txt")) as f:
        tc_feat = f.read().splitlines()
    # Cats need same categories as classifier saw
    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")
    dep["p_tail"] = tc.predict(dep[tc_feat])

    print(f"P_tail dist on all data: mean {dep['p_tail'].mean():.5f} p95 {dep['p_tail'].quantile(0.95):.5f} p99 {dep['p_tail'].quantile(0.99):.5f}")

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
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS + OPDI_NUM_COLS
    print(f"Feature count: {len(feat)}  (v11 had 115)")

    y_tr, y_te = train["TAXITIME_SEC_mvt"].values, test["TAXITIME_SEC_mvt"].values
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining v13...")
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=5000,
                        valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {booster.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = booster.predict(test[feat], num_iteration=booster.best_iteration)
    print(f"\nLGBM v13 hold-out RMSE: {rmse(y_te, p_te):.2f}s   (v11 = 294.18)")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].astype(str).values, "y": y_te, "p": p_te}) \
        .groupby("apt").apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(2)
    v11 = pd.Series({"LEMD":205.05,"LSZH":225.50,"LEBL":239.14,"EDDF":244.53,"EDDM":246.46,
                     "EHAM":248.59,"LTFM":292.64,"LFPG":329.80,"EGLL":360.19,"LIRF":461.44})
    tab["v11"] = v11
    tab["delta"] = (tab["rmse"] - tab["v11"]).round(2)
    print("\nPer-airport RMSE (v13 vs v11):")
    print(tab.sort_values("rmse").to_string())

    imp = pd.Series(booster.feature_importance(importance_type="gain"),
                    index=feat).sort_values(ascending=False)
    r = list(imp.index).index("p_tail") + 1
    print(f"\np_tail rank: {r} of {len(feat)}  gain {imp['p_tail']:.2e}")

    booster.save_model(os.path.join(MODELS_DIR, "lgbm_v13.txt"))
    with open(os.path.join(MODELS_DIR, "lgbm_v13.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(MODELS_DIR, "lgbm_v13.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {MODELS_DIR}/lgbm_v13.*")


if __name__ == "__main__":
    main()
