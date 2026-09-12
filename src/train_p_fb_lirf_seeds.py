"""Train five p_fb_LIRF members with the deployed v23 recipe; only the seed varies.

Priced in section 4 of docs/MODEL_ANALYSIS.md. Reuses the frame and the fallback-rate
encodings of train_lirf_regime_v23.py so the recipe stays in one place.
"""
import json, os, pickle, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from train_lgbm_v21 import CAT_COLS
from train_lirf_regime_v23 import (build_features_lirf, add_fallback_rate_features, MODELS,
                                   TRAIN_MONTHS, EARLYSTOP_MONTHS)

SEEDS = [42, 43, 44, 45, 46]
PARAMS_CLS = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 5,
              "verbosity": -1, "num_threads": -1}


def main():
    t_all = time.time()
    dep, _ = build_features_lirf()
    add_fallback_rate_features(dep, float(dep["is_fb"].mean()))
    with open(os.path.join(MODELS, "lirf_regime_v23.features.txt")) as f:
        feat = f.read().splitlines()
    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")
    train = dep[dep["month"].isin(TRAIN_MONTHS)].copy()
    stop = dep[dep["month"].isin(EARLYSTOP_MONTHS)].copy()
    for c in CAT_COLS:
        stop[c] = pd.Categorical(stop[c], categories=train[c].cat.categories)

    dt = lgb.Dataset(train[feat], label=train["is_fb"].values,
                     categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(stop[feat], label=stop["is_fb"].values,
                     categorical_feature=CAT_COLS, reference=dt, free_raw_data=True)
    log = {}
    for s in SEEDS:
        params = {**PARAMS_CLS, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}
        b = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv], valid_names=["stop"],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        p_st = b.predict(stop[feat], num_iteration=b.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_st, stop["is_fb"].values)
        auc = roc_auc_score(stop["is_fb"], p_st)
        b.save_model(os.path.join(MODELS, f"lgbm_p_fb_lirf_s{s}.txt"), num_iteration=b.best_iteration)
        with open(os.path.join(MODELS, f"lirf_p_fb_s{s}.isotonic.pkl"), "wb") as f:
            pickle.dump(iso, f)
        log[s] = {"best_iter": b.best_iteration, "stop_auc": auc}
        print(f"  seed {s}: best_iter {b.best_iteration}  stop AUC {auc:.4f}")

    iters = [v["best_iter"] for v in log.values()]
    with open(os.path.join(MODELS, "lirf_p_fb_seeds.log.json"), "w") as f:
        json.dump(log, f, indent=1)
    if min(iters) < np.median(iters) / 2:
        raise RuntimeError(f"A best iteration is below half the median of {iters}. "
                           "The recipe drifted. Inspect the members before pricing.")
    print(f"Wall {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
