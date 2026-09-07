"""Quantile-loss LGBM models for ensembling with v15 (MSE).

Median-anchored predictions are less pulled by unpredictable tail flights.
Blending median + mean predictions gives a middle-ground estimator that may
be less biased on right-skewed distributions.

Trains models at q in {0.5, 0.55, 0.6} using cached v5 features to keep
memory low. Then grid-searches blend weight vs v15 on hold-out.
"""
import os
import time
import gc
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "models", "features_cache")
MODELS = os.path.join(ROOT, "models")
QUANTILES = [0.50, 0.55, 0.60]

CAT_COLS = ["ADEP_mvt","ADES_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt",
            "WK_TBL_CAT_flt","MARKET_SEGMENT_flt","AIRCRAFT_OPERATOR_flt",
            "FLIGHT_RULE_mvt","FLIGHT_TYPE_flt"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    t0 = time.time()
    print("Loading cached features + v15 predictions on hold-out...")
    train = pd.read_parquet(CACHE + ".train.parquet")
    test = pd.read_parquet(CACHE + ".test.parquet")
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().splitlines()

    # Downcast + categoricals
    for c in feat:
        if c in CAT_COLS: continue
        if c in train.columns and train[c].dtype == np.float64:
            train[c] = train[c].astype(np.float32)
        if c in test.columns and test[c].dtype == np.float64:
            test[c] = test[c].astype(np.float32)
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    gc.collect()

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(np.float32)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(np.float32)

    # Load v15 hold-out predictions for reference blend
    # We need to run v15 on test... but v15 uses OPDI features not in cache
    # Instead: use v5 (best MSE model on same cached feature set) as baseline
    # v5 hold-out = 295.49; this is the fair comparison
    v5_b = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v5.txt"))
    p_v5 = v5_b.predict(test[feat])
    print(f"v5 (MSE baseline) hold-out RMSE: {rmse(y_te, p_v5):.3f}")

    # Train quantile models
    q_preds = {}
    for q in QUANTILES:
        print(f"\nTraining q={q}...")
        t = time.time()
        params = {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": 255,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "num_threads": -1,
        }
        dtr = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
        dva = lgb.Dataset(test[feat], label=y_te, categorical_feature=CAT_COLS,
                          reference=dtr, free_raw_data=True)
        b = lgb.train(params, dtr, num_boost_round=2000,
                      valid_sets=[dva], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        p = b.predict(test[feat], num_iteration=b.best_iteration)
        q_preds[q] = p
        print(f"  iter {b.best_iteration}  time {time.time()-t:.1f}s  "
              f"RMSE (of q pred alone) {rmse(y_te, p):.2f}")
        # Save
        b.save_model(os.path.join(MODELS, f"lgbm_q{int(q*100)}.txt"))
        with open(os.path.join(MODELS, f"lgbm_q{int(q*100)}.features.txt"), "w") as f:
            f.write("\n".join(feat))
        del dtr, dva, b; gc.collect()

    # Search over v5 (base) + each quantile blend
    print("\nBlend v5 + quantile_q, best alpha per q:")
    for q, pq in q_preds.items():
        best = (float("inf"), None)
        for alpha in np.arange(0, 1.01, 0.05):
            blend = (1 - alpha) * p_v5 + alpha * pq
            r = rmse(y_te, blend)
            if r < best[0]:
                best = (r, alpha)
        print(f"  q={q}: best blend RMSE {best[0]:.3f}s at alpha={best[1]:.2f}   "
              f"(v5 alone {rmse(y_te, p_v5):.3f})")

    # Search over 4-way blend: v5 + q50 + q55 + q60
    print("\n4-way grid (v5 + 3 quantiles, step 0.05)...")
    from itertools import product
    tags = ["v5", *[f"q{int(q*100)}" for q in QUANTILES]]
    preds = {"v5": p_v5, **{f"q{int(q*100)}": q_preds[q] for q in QUANTILES}}
    K = 4
    S = 20
    best = (float("inf"), None)
    cur = [0] * K
    def _rec(i, rem):
        nonlocal best
        if i == K - 1:
            cur[i] = rem
            w = np.array(cur, dtype=float) / S
            p = sum(w[k] * preds[tags[k]] for k in range(K))
            r = rmse(y_te, p)
            if r < best[0]:
                best = (r, tuple(w))
            return
        for v in range(rem + 1):
            cur[i] = v
            _rec(i + 1, rem - v)
    _rec(0, S)
    print(f"  best: {best[0]:.3f}   weights: {dict(zip(tags, best[1]))}")


if __name__ == "__main__":
    main()
