"""L10 (WINNING_PLAN R4): out-of-fold isotonic map for the v56 LIRF gate.

The v56 map is fit on the in-sample scores of the full-year booster, so it
maps in-sample scores to probabilities. 2026 rows are out of sample. This
run trains one gate booster per month-pair fold with the v56 recipe and
rounds, scores each hold-out pair, and fits the isotonic map on the
concatenated out-of-fold scores. The v56 booster stays.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS

from src_v3 import config as C
from src_v3.train_p_fb_v56 import GATE_PARAMS, load_gate_frame

V56_BOOSTER = C.ROOT / "models" / "lgbm_p_fb_lirf_v56.txt"
V56_ISO = C.ROOT / "models" / "lirf_regime_v56.isotonic.pkl"
SD_BINS = [-np.inf, 3_600, 14_400, np.inf]
EPS = 1e-6
log = logging.getLogger(__name__)


def score_oof(dep: pd.DataFrame, feat: list[str], n_iter: int) -> np.ndarray:
    """Raw gate score of every row from the fold booster that did not see its month."""
    oof = np.full(len(dep), np.nan)
    for hold in C.FOLDS:
        is_hold = dep["month"].isin(hold).values
        dt = lgb.Dataset(dep.loc[~is_hold, feat], label=dep["is_fb"].values[~is_hold],
                         categorical_feature=CAT_COLS, params={"feature_pre_filter": False})
        b = lgb.train(GATE_PARAMS, dt, num_boost_round=n_iter)
        oof[is_hold] = b.predict(dep.loc[is_hold, feat])
        log.info("fold %s scored %d rows", hold, is_hold.sum())
    return oof


def compare_maps(oof: np.ndarray, y: np.ndarray, month: pd.Series,
                 sd: np.ndarray, v56_iso: IsotonicRegression) -> dict:
    """Log loss and calibration gap per `sd` bin: v56 map against a nested out-of-fold map."""
    p_new = np.empty_like(oof)
    for hold in C.FOLDS:
        h = month.isin(hold).values
        p_new[h] = IsotonicRegression(out_of_bounds="clip").fit(oof[~h], y[~h]).transform(oof[h])
    p_old = v56_iso.transform(oof)
    report = {}
    for label, m in zip(["sd<=3600", "3600<sd<=14400", "sd>14400"],
                        [(sd > lo) & (sd <= hi) for lo, hi in zip(SD_BINS, SD_BINS[1:])]):
        report[label] = {"n": int(m.sum()), "rate": float(y[m].mean()),
                         **{f"{k}_log_loss": float(log_loss(y[m], np.clip(p[m], EPS, 1 - EPS), labels=[0, 1]))
                            for k, p in (("v56", p_old), ("v64", p_new))},
                         **{f"{k}_gap": float(p[m].mean() - y[m].mean())
                            for k, p in (("v56", p_old), ("v64", p_new))}}
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    n_iter = lgb.Booster(model_file=str(V56_BOOSTER)).num_trees()
    dep, feat = load_gate_frame()
    y = dep["is_fb"].values
    oof = score_oof(dep, feat, n_iter)

    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    with open(C.ROOT / "models" / "lirf_regime_v64.isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(V56_ISO, "rb") as f:
        report = compare_maps(oof, y, dep["month"], dep["sched_delay"].astype(float).values,
                              pickle.load(f))
    log.info("calibration by sd bin: %s", json.dumps(report, indent=1))
    (C.ROOT / "models" / "lirf_regime_v64.meta.json").write_text(json.dumps({
        "rounds_per_fold": n_iter, "folds": C.FOLDS, "n_rows": int(len(dep)),
        "calibration": report}, indent=1))


if __name__ == "__main__":
    main()
