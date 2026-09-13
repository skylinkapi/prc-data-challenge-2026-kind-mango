"""Paired test of the neighbour-tempo columns on the shipped base recipe.

Reads the 97-column frame cached by tune_lgbm_v36.py, joins the 13 tempo,
order, stand-gap and queue columns from the src_v2 frame by movement id, and
trains the R_all_v26 recipe twice on the same split: control (97 columns) and
tempo (110 columns), 3 seeds each. Writes the tempo members as lgbm_r_all_v40
and a per-class hold-out report against the control.
"""
import json
import logging
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS, HOLDOUT_MONTHS, MODELS, STOP_FRAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(MODELS, "v36_tune_cache.parquet")
V2_FRAME = os.path.join(MODELS, "v2", "cache", "frame_train.parquet")
ID_MAP = os.path.join(MODELS, "v36_tune_cache.ids.parquet")
TEMPO_COLS = ["nb_eobt_med_apt30", "nb_eobt_mean_apt30", "nb_eobt_cnt_apt30",
              "nb_eobt_med_apt60", "nb_eobt_mean_apt60", "nb_eobt_cnt_apt60",
              "nb_eobt_med_rwy30", "nb_eobt_mean_rwy30", "nb_eobt_cnt_rwy30",
              "order_later_eobt_30", "order_total_30", "stand_gap", "queue_eobt_mvt"]
FB_TOL = 5
log = logging.getLogger(__name__)


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    """Cached 97-column frame plus the tempo columns joined by movement id."""
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().split()
    dep = pd.read_parquet(CACHE)
    dep["MVT_ID_mvt"] = pd.read_parquet(ID_MAP)["MVT_ID_mvt"].values
    tempo = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    log.info("frame %d rows, tempo coverage %.3f", len(dep), dep["nb_eobt_med_apt30"].notna().mean())
    return dep, feat


def classify(y: np.ndarray, sd: np.ndarray) -> np.ndarray:
    fb = np.abs(y - sd) <= FB_TOL
    out = np.full(len(y), "invalid", dtype=object)
    out[(y >= 30) & (y <= 7200) & ~fb] = "clean"
    out[fb & (y > 0)] = "fallback"
    out[(y > 7200) & (y <= 80000) & ~fb] = "tail"
    out[(y > 80000) & ~fb] = "24h"
    return out


def class_table(pred: np.ndarray, y: np.ndarray, cls: np.ndarray, apt: np.ndarray) -> dict:
    """FULL, CLEAN and per-airport class MSE on the hold-out scale."""
    e2 = (pred - y) ** 2
    n = len(y)
    per_apt = {a: {c: float(e2[(apt == a) & (cls == c)].sum() / n)
                   for c in ("clean", "fallback", "tail", "24h")} for a in sorted(set(apt))}
    return {"full_rmse": float(np.sqrt(e2.mean())),
            "clean_rmse": float(np.sqrt(e2[cls == "clean"].mean())),
            "class_mse": {c: float(e2[cls == c].sum() / n) for c in ("clean", "fallback", "tail", "24h")},
            "per_airport": per_apt}


def train_members(train: pd.DataFrame, stop: pd.DataFrame, test: pd.DataFrame,
                  feat: list[str], tag: str) -> tuple[np.ndarray, list[int]]:
    """Three members of the v26 recipe; returns the test mean and best iterations."""
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values, categorical_feature=CAT_COLS)
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values, categorical_feature=CAT_COLS, reference=dt)
    preds, iters = [], []
    for seed in DEFAULT_SEEDS:
        params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0, "num_leaves": 220,
                  "seed": seed, "bagging_seed": seed, "bagging_freq": 5,
                  "verbosity": -1, "num_threads": -1}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        log.info("%s seed %d best iter %d in %.0fs", tag, seed, b.best_iteration, time.time() - t0)
        if tag == "v40":
            b.save_model(os.path.join(MODELS, f"lgbm_r_all_v40_s{seed}.txt"))
        preds.append(np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None))
        iters.append(b.best_iteration)
    return np.mean(preds, axis=0), iters


def main() -> None:
    dep, feat = load_frame()
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    non_hold = dep.loc[~hold]
    stop_mask = np.random.default_rng(1234).random(len(non_hold)) < STOP_FRAC
    train, stop, test = non_hold[~stop_mask], non_hold[stop_mask], dep.loc[hold]
    y, sd, apt = (test["TAXITIME_SEC_mvt"].values.astype(float), test["sched_delay"].values.astype(float),
                  test["ADEP_mvt"].astype(str).values)
    cls = classify(y, sd)

    report = {"tempo_cols": TEMPO_COLS}
    for tag, cols in (("control", feat), ("v40", feat + TEMPO_COLS)):
        pred, iters = train_members(train, stop, test, cols, tag)
        report[tag] = {"best_iters": iters, **class_table(pred, y, cls, apt)}
        log.info("%s FULL %.2f CLEAN %.2f iters %s", tag, report[tag]["full_rmse"],
                 report[tag]["clean_rmse"], iters)
    report["delta_v40_minus_control"] = {
        k: report["v40"][k] - report["control"][k] for k in ("full_rmse", "clean_rmse")}
    with open(os.path.join(MODELS, "lgbm_r_all_v40.features.txt"), "w") as f:
        f.write("\n".join(feat + TEMPO_COLS))
    shutil.copy(os.path.join(MODELS, "lgbm_r_all_v26.encoders.pkl"),
                os.path.join(MODELS, "lgbm_r_all_v40.encoders.pkl"))
    with open(os.path.join(MODELS, "lgbm_r_all_v40.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)
    log.info("delta FULL %+.2f CLEAN %+.2f", *report["delta_v40_minus_control"].values())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
