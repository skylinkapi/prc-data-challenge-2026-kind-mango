"""Paired test of the tempo columns on the LIRF regressor R_norm_LIRF.

Builds the LIRF frame with train_lirf_regime.build_features, joins the 13
tempo, order, stand-gap and queue columns by movement id, and trains the five
deployed members again with the same recipe plus those columns. Scores the
whole LIRF head (mixture with the v23 gate, Step A, ITY340) on the hold-out
months for the deployed members and for the new ones. Writes the new members
as lgbm_r_norm_lirf_v41_s{seed} and a per-class report.
"""
import json
import logging
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from eval_v33_holdout import R_NORM_FILES, lirf_head, read_list
from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
from train_lirf_regime import EARLYSTOP_MONTHS, FB_TOL, HOLDOUT_MONTHS, MODELS, TRAIN_MONTHS, build_features
from train_r_all_v40 import TEMPO_COLS, V2_FRAME, class_table, classify
from train_r_norm_lirf_seeds import SEEDS

N_HOLDOUT = 344336
log = logging.getLogger(__name__)


def genuine(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["TAXITIME_SEC_mvt"].astype(float).values
    sd = frame["sched_delay"].astype(float).values
    return frame[(np.abs(y - sd) >= FB_TOL) & (y < 80000)]


def train_members(dep: pd.DataFrame, feat: list[str]) -> list[int]:
    """Five members on the genuine LIRF rows; returns the best iterations."""
    d = dep.copy()
    for c in CAT_COLS:
        d[c] = d[c].astype("category")
    tr = genuine(d[d["month"].isin(TRAIN_MONTHS)])
    st = genuine(d[d["month"].isin(EARLYSTOP_MONTHS)])
    dt = lgb.Dataset(tr[feat], label=tr["TAXITIME_SEC_mvt"].astype(float).values, categorical_feature=CAT_COLS)
    dv = lgb.Dataset(st[feat], label=st["TAXITIME_SEC_mvt"].astype(float).values, categorical_feature=CAT_COLS,
                     reference=dt)
    iters = []
    for s in SEEDS:
        params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse", "linear_tree": True,
                  "linear_lambda": 1.0, "num_leaves": 127, "bagging_freq": 5, "verbosity": -1,
                  "num_threads": -1, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=3000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        b.save_model(os.path.join(MODELS, f"lgbm_r_norm_lirf_v41_s{s}.txt"))
        iters.append(b.best_iteration)
        log.info("v41 seed %d best iter %d in %.0fs", s, b.best_iteration, time.time() - t0)
    return iters


def score_head(hold: pd.DataFrame, members: list[str], feat: list[str]) -> dict:
    pred = lirf_head(hold.copy(), members, feat)
    y = hold["TAXITIME_SEC_mvt"].astype(float).values
    sd = hold["sched_delay"].astype(float).values
    rep = class_table(pred, y, classify(y, sd), np.full(len(y), "LIRF"))
    rep["class_mse"] = {k: v * len(y) / N_HOLDOUT for k, v in rep["class_mse"].items()}
    return rep


def main() -> None:
    dep, _ = build_features()
    tempo = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    feat_old = read_list("lirf_regime.features.txt")
    feat_new = feat_old + TEMPO_COLS
    iters = train_members(dep, feat_new)
    with open(os.path.join(MODELS, "lirf_regime_v41.features.txt"), "w") as f:
        f.write("\n".join(feat_new))

    hold = dep[dep["month"].isin(HOLDOUT_MONTHS)].copy()
    new_files = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in SEEDS]
    report = {"control": score_head(hold, R_NORM_FILES, feat_old),
              "v41": {"best_iters": iters, **score_head(hold, new_files, feat_new)}}
    for k in ("control", "v41"):
        log.info("%s LIRF head: FULL %.2f CLEAN %.2f class %s", k, report[k]["full_rmse"],
                 report[k]["clean_rmse"], {c: round(v) for c, v in report[k]["class_mse"].items()})
    with open(os.path.join(MODELS, "lirf_regime_v41.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
