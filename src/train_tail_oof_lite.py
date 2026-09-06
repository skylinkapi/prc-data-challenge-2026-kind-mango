"""Memory-optimized 5-fold OOF P_tail using cached features.

Uses models/features_cache.{train,test}.parquet (v5-era features, 97 cols).
Missing OSM path / OPDI / VRS but keeps ~90% of tail signal from basics.
Trades slight AUC loss for reliable memory footprint.
"""
import os
import time
import gc
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "models", "features_cache")
MODELS_DIR = os.path.join(ROOT, "models")
TAIL_THRESHOLD = 3600.0
N_FOLDS = 5

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]


def main():
    t0 = time.time()
    print("Loading cached features...")
    train = pd.read_parquet(CACHE + ".train.parquet")
    test = pd.read_parquet(CACHE + ".test.parquet")
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().splitlines()
    print(f"  train {len(train):,}   test {len(test):,}   feats {len(feat)}")

    # Downcast numerics to float32 for memory
    for c in feat:
        if c in CAT_COLS: continue
        if c in train.columns and train[c].dtype == np.float64:
            train[c] = train[c].astype(np.float32)
        if c in test.columns and test[c].dtype == np.float64:
            test[c] = test[c].astype(np.float32)
    gc.collect()

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    train["is_tail"] = (train["TAXITIME_SEC_mvt"] > TAIL_THRESHOLD).astype(np.int8)
    test["is_tail"] = (test["TAXITIME_SEC_mvt"] > TAIL_THRESHOLD).astype(np.int8)
    print(f"Train tail rate: {train['is_tail'].mean()*100:.3f}%   hold-out: {test['is_tail'].mean()*100:.3f}%")

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

    print(f"\n{N_FOLDS}-fold OOF ...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(train), dtype=np.float32)
    y = train["is_tail"].values

    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(len(train)))):
        t = time.time()
        dtr = lgb.Dataset(train[feat].iloc[tr_idx],
                          label=y[tr_idx],
                          categorical_feature=CAT_COLS,
                          free_raw_data=True)
        dva = lgb.Dataset(train[feat].iloc[va_idx],
                          label=y[va_idx],
                          categorical_feature=CAT_COLS,
                          reference=dtr,
                          free_raw_data=True)
        b = lgb.train(params, dtr, num_boost_round=1500,
                      valid_sets=[dva], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(50, verbose=False, first_metric_only=True),
                                 lgb.log_evaluation(0)])
        oof[va_idx] = b.predict(train[feat].iloc[va_idx], num_iteration=b.best_iteration)
        print(f"  fold {fold+1}: iter {b.best_iteration} time {time.time()-t:.1f}s")
        del dtr, dva, b; gc.collect()

    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    print(f"\nOOF AUC: {auc:.4f}   AP: {ap:.4f}")
    print(f"OOF P_tail: mean {oof.mean():.5f}  p99 {np.percentile(oof, 99):.5f}  max {oof.max():.5f}")

    # Final classifier on all training -> predict hold-out
    print("\nFinal classifier on all training -> hold-out...")
    dfin = lgb.Dataset(train[feat], label=y, categorical_feature=CAT_COLS, free_raw_data=True)
    dh   = lgb.Dataset(test[feat], label=test["is_tail"].values,
                       categorical_feature=CAT_COLS, reference=dfin, free_raw_data=True)
    b = lgb.train(params, dfin, num_boost_round=1500,
                  valid_sets=[dh], valid_names=["hold"],
                  callbacks=[lgb.early_stopping(50, verbose=False, first_metric_only=True),
                             lgb.log_evaluation(200)])
    hold_p = b.predict(test[feat], num_iteration=b.best_iteration)
    print(f"Hold-out AUC: {roc_auc_score(test['is_tail'].values, hold_p):.4f}")
    print(f"Hold-out AP:  {average_precision_score(test['is_tail'].values, hold_p):.4f}")

    # Save OOF + hold-out P_tail as row-index-aligned parquet arrays
    pd.DataFrame({"row": np.arange(len(train)), "p_tail": oof}).to_parquet(
        os.path.join(MODELS_DIR, "p_tail_train_oof.parquet"))
    pd.DataFrame({"row": np.arange(len(test)),  "p_tail": hold_p}).to_parquet(
        os.path.join(MODELS_DIR, "p_tail_test.parquet"))
    print(f"Saved p_tail_train_oof.parquet + p_tail_test.parquet")
    b.save_model(os.path.join(MODELS_DIR, "lgbm_tail_classifier_final.txt"))
    with open(os.path.join(MODELS_DIR, "lgbm_tail_classifier_final.features.txt"), "w") as f:
        f.write("\n".join(feat))
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
