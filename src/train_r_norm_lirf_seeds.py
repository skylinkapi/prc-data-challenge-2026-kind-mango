"""Train five R_norm_LIRF members with the deployed recipe.

Priced in section 5.2 of docs/MODEL_ANALYSIS.md. Reuses build_features from
train_lirf_regime.py so the frame stays in one place (section 6 debt 2).
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

from train_lirf_regime import (build_features, MODELS, TRAIN_MONTHS,
                               EARLYSTOP_MONTHS, HOLDOUT_MONTHS, FB_TOL, rmse)
from train_lgbm_v21 import CAT_COLS, BEST_PARAMS

SEEDS = [42, 43, 44, 45, 46]


def main():
    t_all = time.time()
    dep, _ = build_features()
    with open(os.path.join(MODELS, "lirf_regime.features.txt")) as f:
        feat = f.read().splitlines()

    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")

    train = dep[dep["month"].isin(TRAIN_MONTHS)].copy()
    stop = dep[dep["month"].isin(EARLYSTOP_MONTHS)].copy()
    test = dep[dep["month"].isin(HOLDOUT_MONTHS)].copy()
    for c in CAT_COLS:
        stop[c] = pd.Categorical(stop[c], categories=train[c].cat.categories)
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    y_tr = train["TAXITIME_SEC_mvt"].astype(float).values
    sd_tr = train["sched_delay"].astype(float).values
    y_st = stop["TAXITIME_SEC_mvt"].astype(float).values
    sd_st = stop["sched_delay"].astype(float).values
    y_te = test["TAXITIME_SEC_mvt"].astype(float).values
    sd_te = test["sched_delay"].astype(float).values

    g_tr = (np.abs(y_tr - sd_tr) >= FB_TOL) & (y_tr < 80000)
    g_st = (np.abs(y_st - sd_st) >= FB_TOL) & (y_st < 80000)

    tr_g = train[g_tr]
    st_g = stop[g_st]
    print(f"Genuine train: {len(tr_g):,}  stop: {len(st_g):,}")

    dt = lgb.Dataset(tr_g[feat], label=tr_g["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(st_g[feat], label=st_g["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS, reference=dt, free_raw_data=True)

    best_iters, hold_rmse = [], []
    for s in SEEDS:
        params = {**BEST_PARAMS,
                  "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0,
                  "num_leaves": 127,
                  "bagging_freq": 5, "verbosity": -1, "num_threads": -1,
                  "seed": s, "bagging_seed": s, "feature_fraction_seed": s}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=3000,
                      valid_sets=[dv], valid_names=["stop"],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        p_te = np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None)
        r = rmse(y_te, p_te)
        print(f"  seed {s}: best_iter {b.best_iteration}  hold FULL RMSE {r:.2f}  ({time.time()-t0:.1f}s)")
        b.save_model(os.path.join(MODELS, f"lgbm_r_norm_lirf_s{s}.txt"))
        best_iters.append(b.best_iteration)
        hold_rmse.append(r)

    print(f"\nBest iterations: {best_iters}")
    print(f"Hold FULL RMSE:  {[f'{r:.2f}' for r in hold_rmse]}")
    med = float(np.median(best_iters))
    if min(best_iters) < med / 2:
        raise RuntimeError(f"A best iteration is below half the median ({med}). "
                           f"Recipe drift — inspect before shipping.")
    print(f"\nWall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
