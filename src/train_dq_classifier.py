"""Data-quality classifier: predict P(row has BLOCK==SCHED fallback).

For those rows the ranking truth is (MVT_TIME - SCHED_TIME) = sched_delay,
NOT the real taxi time. If we detect them, we can output that formula directly.

5-fold OOF on training data + one-shot classifier on all training for
predicting on hold-out + ranking rows.

Uses the memory-lite features cache (v5-era, 97 cols) to avoid OOM issues.
"""
import os
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "models", "features_cache")
MODELS_DIR = os.path.join(ROOT, "models")
N_FOLDS = 5

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]


def main():
    t0 = time.time()
    print("Loading cached features + reading BLOCK/SCHED from training parquets...")
    train_feat = pd.read_parquet(CACHE + ".train.parquet")
    test_feat = pd.read_parquet(CACHE + ".test.parquet")
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().splitlines()

    # Recover is_dq labels for training and hold-out from the raw parquets
    import glob
    from features_weather import TARGET_ICAOS
    rows_needed_train = len(train_feat)
    rows_needed_test = len(test_feat)
    print(f"  train {rows_needed_train:,}   test {rows_needed_test:,}")

    frames = [pd.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                          "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                          "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
              for f in sorted(glob.glob(os.path.join(ROOT, "training", "*.parquet")))]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw["PHASE_mvt"] == "DEP") & raw["ADEP_mvt"].isin(TARGET_ICAOS)]
    raw["mvt_ts"] = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], errors="coerce")
    raw["sched_ts"] = pd.to_datetime(raw["SCHED_TIME_UTC_mvt"], errors="coerce")
    raw["blk_ts"] = pd.to_datetime(raw["BLOCK_TIME_UTC_mvt"], errors="coerce")
    raw["is_dq"] = ((raw["blk_ts"] - raw["sched_ts"]).dt.total_seconds().abs() < 1).astype(np.int8)
    raw = raw[raw["TAXITIME_SEC_mvt"].between(30, 7200)]
    raw["month"] = raw["mvt_ts"].dt.month
    print(f"  raw filtered rows: {len(raw):,}, DQ rate: {raw['is_dq'].mean()*100:.2f}%")

    # Match train/test to raw via row order (cached parquets built from same filter chain)
    tr_raw = raw[~raw["month"].isin({1, 7})].reset_index(drop=True)
    te_raw = raw[raw["month"].isin({1, 7})].reset_index(drop=True)
    assert len(tr_raw) == len(train_feat), f"mismatch: raw {len(tr_raw)} vs cache {len(train_feat)}"
    assert len(te_raw) == len(test_feat), f"mismatch: raw {len(te_raw)} vs cache {len(test_feat)}"

    y_tr = tr_raw["is_dq"].values
    y_te = te_raw["is_dq"].values
    print(f"  train DQ rate: {y_tr.mean()*100:.2f}% ({y_tr.sum():,})")
    print(f"  test  DQ rate: {y_te.mean()*100:.2f}% ({y_te.sum():,})")

    # Downcast + categoricals
    for c in feat:
        if c in CAT_COLS: continue
        if c in train_feat.columns and train_feat[c].dtype == np.float64:
            train_feat[c] = train_feat[c].astype(np.float32)
        if c in test_feat.columns and test_feat[c].dtype == np.float64:
            test_feat[c] = test_feat[c].astype(np.float32)
    for c in CAT_COLS:
        train_feat[c] = train_feat[c].astype("category")
        test_feat[c] = pd.Categorical(test_feat[c], categories=train_feat[c].cat.categories)
    gc.collect()

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

    print(f"\n{N_FOLDS}-fold OOF for DQ classifier...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(train_feat), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(len(train_feat)))):
        t = time.time()
        dtr = lgb.Dataset(train_feat[feat].iloc[tr_idx], label=y_tr[tr_idx],
                          categorical_feature=CAT_COLS, free_raw_data=True)
        dva = lgb.Dataset(train_feat[feat].iloc[va_idx], label=y_tr[va_idx],
                          categorical_feature=CAT_COLS, reference=dtr, free_raw_data=True)
        b = lgb.train(params, dtr, num_boost_round=1500,
                      valid_sets=[dva], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(50, verbose=False, first_metric_only=True),
                                 lgb.log_evaluation(0)])
        oof[va_idx] = b.predict(train_feat[feat].iloc[va_idx], num_iteration=b.best_iteration)
        print(f"  fold {fold+1}: iter {b.best_iteration}  time {time.time()-t:.1f}s")
        del dtr, dva, b; gc.collect()

    auc = roc_auc_score(y_tr, oof)
    ap = average_precision_score(y_tr, oof)
    print(f"\nOOF AUC: {auc:.4f}   AP: {ap:.4f}")
    print(f"OOF P_dq: mean {oof.mean():.5f} p95 {np.percentile(oof, 95):.5f} p99 {np.percentile(oof, 99):.5f} max {oof.max():.5f}")

    # Final classifier on all training, predict hold-out
    print("\nFinal on all-train -> hold-out...")
    dfin = lgb.Dataset(train_feat[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dh   = lgb.Dataset(test_feat[feat], label=y_te,
                       categorical_feature=CAT_COLS, reference=dfin, free_raw_data=True)
    bf = lgb.train(params, dfin, num_boost_round=1500,
                   valid_sets=[dh], valid_names=["hold"],
                   callbacks=[lgb.early_stopping(50, verbose=False, first_metric_only=True),
                              lgb.log_evaluation(200)])
    hold_p = bf.predict(test_feat[feat], num_iteration=bf.best_iteration)
    print(f"Hold-out AUC: {roc_auc_score(y_te, hold_p):.4f}   AP: {average_precision_score(y_te, hold_p):.4f}")

    # Save
    pd.DataFrame({"row": np.arange(len(train_feat)), "p_dq": oof}).to_parquet(
        os.path.join(MODELS_DIR, "p_dq_train_oof.parquet"))
    pd.DataFrame({"row": np.arange(len(test_feat)), "p_dq": hold_p}).to_parquet(
        os.path.join(MODELS_DIR, "p_dq_test.parquet"))
    bf.save_model(os.path.join(MODELS_DIR, "lgbm_dq_classifier.txt"))
    with open(os.path.join(MODELS_DIR, "lgbm_dq_classifier.features.txt"), "w") as f:
        f.write("\n".join(feat))
    print(f"\nSaved. Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
