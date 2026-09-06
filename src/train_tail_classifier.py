"""Binary tail classifier for the mixture model.

Predicts P(TAXITIME_SEC_mvt > 3600) using the same feature set as v11.
Reports AUC and per-airport precision-recall so we can judge whether the tail
is actually predictable (if AUC ~ 0.5 the mixture idea is dead).

Uses LightGBM binary objective with class_weight to handle imbalance
(~0.35 % positive rate globally).
"""
import glob
import os
import pickle
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

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
HOLDOUT_MONTHS = {1, 7}
TAIL_THRESHOLD = 3600.0

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
    print("Loading + building features...")
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
    gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  Features in {time.time()-t0:.1f}s")

    dep["is_tail"] = (dep["TAXITIME_SEC_mvt"] > TAIL_THRESHOLD).astype(np.int8)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train = dep.loc[~hold].copy()
    test = dep.loc[hold].copy()

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS + OPDI_NUM_COLS
    print(f"Feature count: {len(feat)}")

    y_tr = train["is_tail"].values
    y_te = test["is_tail"].values
    pos_rate = y_tr.mean()
    print(f"\nTrain tail rate: {pos_rate*100:.3f}%  ({y_tr.sum():,} / {len(y_tr):,})")
    print(f"Hold-out tail rate: {y_te.mean()*100:.3f}%  ({y_te.sum():,} / {len(y_te):,})")

    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision", "binary_logloss"],
        "learning_rate": 0.02,
        "num_leaves": 255,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        # NO scale_pos_weight — keep calibrated probabilities
        "verbosity": -1,
        "num_threads": -1,
    }

    print("\nTraining tail classifier (calibrated, no pos-weight)...")
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=3000,
                        valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(80, first_metric_only=True),
                                   lgb.log_evaluation(200)])
    print(f"  best iter: {booster.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = booster.predict(test[feat], num_iteration=booster.best_iteration)
    print(f"\nP_tail dist: mean {p_te.mean():.5f} med {np.median(p_te):.5f} "
          f"p95 {np.percentile(p_te, 95):.5f} p99 {np.percentile(p_te, 99):.5f} "
          f"max {p_te.max():.5f}")
    print(f"Rows P>0.5: {(p_te>0.5).sum():,}   P>0.1: {(p_te>0.1).sum():,}   P>0.05: {(p_te>0.05).sum():,}")

    # Global metrics
    auc = roc_auc_score(y_te, p_te)
    ap = average_precision_score(y_te, p_te)
    print(f"\nGlobal hold-out AUC: {auc:.4f}   average precision: {ap:.4f}")

    # Per-airport metrics
    print("\nPer-airport tail metrics on hold-out:")
    apt = test["ADEP_mvt"].astype(str).values
    print(f"{'apt':<6}{'n_test':<10}{'n_tail':<8}{'AUC':<8}{'AP':<8}")
    for a in sorted(set(apt)):
        mask = apt == a
        n_tail = y_te[mask].sum()
        if n_tail < 5 or n_tail == mask.sum():
            print(f"{a:<6}{mask.sum():<10}{n_tail:<8}insufficient tail rows")
            continue
        try:
            auc_a = roc_auc_score(y_te[mask], p_te[mask])
            ap_a = average_precision_score(y_te[mask], p_te[mask])
            print(f"{a:<6}{mask.sum():<10}{n_tail:<8}{auc_a:<8.3f}{ap_a:<8.3f}")
        except: pass

    # Save
    out_dir = os.path.join(ROOT, "models")
    booster.save_model(os.path.join(out_dir, "lgbm_tail_classifier.txt"))
    with open(os.path.join(out_dir, "lgbm_tail_classifier.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_tail_classifier.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    # Also save per-airport tail medians for the mixture step
    tail_train = train[train["is_tail"] == 1]
    tail_medians = tail_train.groupby("ADEP_mvt", observed=True)["TAXITIME_SEC_mvt"].median().to_dict()
    tail_medians = {str(k): float(v) for k, v in tail_medians.items()}
    import json
    with open(os.path.join(out_dir, "tail_medians.json"), "w") as f:
        json.dump(tail_medians, f, indent=2)
    print(f"\nPer-airport tail medians:")
    for a, v in sorted(tail_medians.items()):
        print(f"  {a}: {v:.0f}s")

    print(f"\nSaved -> {out_dir}/lgbm_tail_classifier.* + tail_medians.json")


if __name__ == "__main__":
    main()
