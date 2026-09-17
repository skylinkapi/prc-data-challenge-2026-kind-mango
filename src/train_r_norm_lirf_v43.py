"""L5: R_norm_LIRF trained on the 5-second genuine mask.

Fourteenth pass, MODEL_ANALYSIS section 4.1 L5: the deployed R_norm_LIRF fits
rows with abs(y - sd) >= 60, which drops the 3.8 points of punctual genuine
taxis and keeps a biased sample. This script rebuilds the five v41 members
with FB_TOL_L5 = 5 on the fit mask only. The gate keeps its 60-second target;
the mixture stays consistent because the gate serves probabilities, not rows.
Writes lgbm_r_norm_lirf_v43_s{seed}.txt and lirf_regime_v43.holdout.json
(paired against v41 with r_norm_clip=None, matching the shipped stack).
"""
import json
import logging
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from eval_v33_holdout import lirf_head, read_list
from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
from train_lirf_regime import EARLYSTOP_MONTHS, HOLDOUT_MONTHS, MODELS, TRAIN_MONTHS, build_features
from train_r_all_v40 import TEMPO_COLS, V2_FRAME, class_table, classify
from train_r_norm_lirf_seeds import SEEDS

FB_TOL_L5 = 5
N_HOLDOUT = 344336
LIRF_CACHE = os.path.join(MODELS, "lirf_frame_cache.parquet")
log = logging.getLogger(__name__)


def genuine(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["TAXITIME_SEC_mvt"].astype(float).values
    sd = frame["sched_delay"].astype(float).values
    return frame[(np.abs(y - sd) >= FB_TOL_L5) & (y < 80000)]


def train_members(dep: pd.DataFrame, feat: list[str]) -> list[int]:
    d = dep.copy()
    for c in CAT_COLS:
        d[c] = d[c].astype("category")
    tr = genuine(d[d["month"].isin(TRAIN_MONTHS)])
    st = genuine(d[d["month"].isin(EARLYSTOP_MONTHS)])
    log.info("train %d stop %d (FB_TOL=%d)", len(tr), len(st), FB_TOL_L5)
    dt = lgb.Dataset(tr[feat], label=tr["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS)
    dv = lgb.Dataset(st[feat], label=st["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS, reference=dt)
    iters: list[int] = []
    for s in SEEDS:
        params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0, "num_leaves": 127,
                  "bagging_freq": 5, "verbosity": -1, "num_threads": -1,
                  "seed": s, "bagging_seed": s, "feature_fraction_seed": s}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=3000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        b.save_model(os.path.join(MODELS, f"lgbm_r_norm_lirf_v43_s{s}.txt"))
        iters.append(b.best_iteration)
        log.info("v43 seed %d best iter %d in %.0fs", s, b.best_iteration, time.time() - t0)
    return iters


def score_head(hold: pd.DataFrame, members: list[str], feat: list[str]) -> dict:
    pred = lirf_head(hold.copy(), members, feat, r_norm_clip=None)
    y = hold["TAXITIME_SEC_mvt"].astype(float).values
    sd = hold["sched_delay"].astype(float).values
    rep = class_table(pred, y, classify(y, sd), np.full(len(y), "LIRF"))
    rep["class_mse"] = {k: v * len(y) / N_HOLDOUT for k, v in rep["class_mse"].items()}
    return rep


def load_dep(rebuild: bool) -> pd.DataFrame:
    if not rebuild and os.path.exists(LIRF_CACHE):
        return pd.read_parquet(LIRF_CACHE)
    dep, _ = build_features()
    dep.to_parquet(LIRF_CACHE)
    log.info("rebuilt the LIRF frame cache: %d rows", len(dep))
    return dep


def main(rebuild: bool = False) -> None:
    dep = load_dep(rebuild)
    tempo = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    feat = read_list("lirf_regime_v41.features.txt")
    iters = train_members(dep, feat)

    hold = dep[dep["month"].isin(HOLDOUT_MONTHS)].copy()
    v41_files = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in SEEDS]
    new_files = [f"lgbm_r_norm_lirf_v43_s{s}.txt" for s in SEEDS]
    report = {"fb_tol": FB_TOL_L5,
              "control_v41": score_head(hold, v41_files, feat),
              "v43_l5": {"best_iters": iters, **score_head(hold, new_files, feat)}}
    for k in ("control_v41", "v43_l5"):
        r = report[k]
        log.info("%s LIRF head: FULL %.2f CLEAN %.2f class %s", k,
                 r["full_rmse"], r["clean_rmse"],
                 {c: round(v) for c, v in r["class_mse"].items()})
    delta = sum(report["control_v41"]["class_mse"].values()) - sum(report["v43_l5"]["class_mse"].values())
    log.info("paired price (control_v41 - v43_l5): %+.0f MSE (gate >= 200)", delta)
    report["paired_price_mse"] = delta
    with open(os.path.join(MODELS, "lirf_regime_v43.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-lirf", action="store_true")
    args = ap.parse_args()
    main(rebuild=args.rebuild_lirf)
