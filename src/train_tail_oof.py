"""Compute leak-free P_tail via 5-fold OOF on training data.

For each fold: train classifier on other 4 folds, predict on this fold.
For hold-out (Jan+Jul 2025): train classifier on ALL training data (any leakage
here would only be about hold-out itself, and there's no target-based split).

Saves: models/p_tail_oof.parquet  — DataFrame with MVT_ID_mvt, p_tail
"""
import glob
import os
import time
import gc
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

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
TAIL_THRESHOLD = 3600.0
N_FOLDS = 5

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


def main():
    t0 = time.time()
    print("Loading + features...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep = dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)]

    m_ctx = m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]].copy()
    del m; gc.collect()

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

    dep["is_tail"] = (dep["TAXITIME_SEC_mvt"] > TAIL_THRESHOLD).astype(np.int8)
    print(f"  Features + encoders in {time.time()-t0:.1f}s")

    train_mask = ~dep["month"].isin(HOLDOUT_MONTHS)
    train_df = dep[train_mask].copy()
    hold_df = dep[~train_mask].copy()

    for c in CAT_COLS:
        train_df[c] = train_df[c].astype("category")
        hold_df[c] = pd.Categorical(hold_df[c], categories=train_df[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS + OPDI_NUM_COLS

    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "learning_rate": 0.03,
        "num_leaves": 255,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": -1,
    }

    print(f"\nRunning {N_FOLDS}-fold OOF on {len(train_df):,} training rows...")
    train_idx_arr = np.arange(len(train_df))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(train_df), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_idx_arr)):
        t0 = time.time()
        sub_tr = train_df.iloc[tr_idx]
        sub_va = train_df.iloc[va_idx]
        dtr = lgb.Dataset(sub_tr[feat], label=sub_tr["is_tail"].values,
                          categorical_feature=CAT_COLS, free_raw_data=False)
        dva = lgb.Dataset(sub_va[feat], label=sub_va["is_tail"].values,
                          categorical_feature=CAT_COLS, reference=dtr, free_raw_data=False)
        b = lgb.train(params, dtr, num_boost_round=2000,
                      valid_sets=[dva], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(60, verbose=False, first_metric_only=True),
                                 lgb.log_evaluation(0)])
        oof[va_idx] = b.predict(sub_va[feat], num_iteration=b.best_iteration)
        print(f"  fold {fold+1}/{N_FOLDS}  iter {b.best_iteration}  time {time.time()-t0:.1f}s")

    print(f"\nOOF P_tail: mean {oof.mean():.5f}  p99 {np.percentile(oof, 99):.5f}  max {oof.max():.5f}")
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc_oof = roc_auc_score(train_df["is_tail"].values, oof)
    ap_oof  = average_precision_score(train_df["is_tail"].values, oof)
    print(f"OOF AUC: {auc_oof:.4f}   OOF AP: {ap_oof:.4f}")

    # Train final classifier on ALL training and predict hold-out
    print("\nFinal classifier on all training rows -> predict hold-out...")
    dfin = lgb.Dataset(train_df[feat], label=train_df["is_tail"].values,
                       categorical_feature=CAT_COLS, free_raw_data=False)
    dh   = lgb.Dataset(hold_df[feat], label=hold_df["is_tail"].values,
                       categorical_feature=CAT_COLS, reference=dfin, free_raw_data=False)
    bfinal = lgb.train(params, dfin, num_boost_round=2000,
                       valid_sets=[dh], valid_names=["hold"],
                       callbacks=[lgb.early_stopping(60, verbose=False, first_metric_only=True),
                                  lgb.log_evaluation(200)])
    hold_p = bfinal.predict(hold_df[feat], num_iteration=bfinal.best_iteration)

    from sklearn.metrics import roc_auc_score as ras
    print(f"Hold-out AUC: {ras(hold_df['is_tail'].values, hold_p):.4f}")

    # Save the P_tail values per MVT_ID_mvt for the regressor
    out = pd.concat([
        pd.DataFrame({"MVT_ID_mvt": train_df["MVT_ID_mvt"].values, "p_tail": oof}),
        pd.DataFrame({"MVT_ID_mvt": hold_df["MVT_ID_mvt"].values,  "p_tail": hold_p}),
    ])
    out_path = os.path.join(MODELS_DIR, "p_tail_oof.parquet")
    out.to_parquet(out_path)
    print(f"\nSaved -> {out_path}  ({len(out):,} rows)")

    # Also save final classifier for ranking-file inference
    bfinal.save_model(os.path.join(MODELS_DIR, "lgbm_tail_classifier_final.txt"))
    with open(os.path.join(MODELS_DIR, "lgbm_tail_classifier_final.features.txt"), "w") as f:
        f.write("\n".join(feat))


if __name__ == "__main__":
    main()
