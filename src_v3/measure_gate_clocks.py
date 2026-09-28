"""Paired price of the NM clock columns in the LIRF gate on the 2025 hold-out.

Both arms follow the v64 protocol: one gate booster per month-pair fold
with the v56 recipe and rounds, out-of-fold scores, and a nested isotonic
map per fold. The control reads the 167 v23 columns; the treatment adds
the five clock columns. The report gives log loss per `sd` bin over all
folds, and the MSE of the served mixture `p sd + (1 - p) R_norm` on the
LIRF rows of months 1 and 7 outside Step A, where the five v41 `R_norm`
members are out of sample. The scale is 344,336 rows. `--extra` prices the
columns of other parquets instead of the clocks (v83: the arrival link).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_r_all_v40 import TEMPO_COLS, V2_FRAME

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import OUT_TRAIN as CLOCKS_TRAIN
from src_v3.measure_catboost_r_norm import V41_SEEDS
from src_v3.measure_tail_rules import HOLD, mse_delta
from src_v3.train_p_fb_v56 import load_gate_frame
from src_v3.train_p_fb_v64 import EPS, SD_BINS, V56_BOOSTER, score_oof

log = logging.getLogger(__name__)


def nested_map(oof: np.ndarray, y: np.ndarray, month: pd.Series) -> np.ndarray:
    """Calibrated probability of each row from an isotonic map fit on the other folds."""
    p = np.empty_like(oof)
    for hold in C.FOLDS:
        h = month.isin(hold).values
        p[h] = IsotonicRegression(out_of_bounds="clip").fit(oof[~h], y[~h]).transform(oof[h])
    return p


def r_norm_v41(frame: pd.DataFrame) -> np.ndarray:
    """Mean of the five v41 `R_norm` members, clipped at 0."""
    feat = [x for x in (C.ROOT / "models" / "lirf_regime_v41.features.txt").read_text().splitlines() if x]
    return np.mean([np.clip(lgb.Booster(model_file=str(C.ROOT / "models" / f"lgbm_r_norm_lirf_v41_s{s}.txt"))
                            .predict(frame[feat]), 0, None) for s in V41_SEEDS], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="clocks")
    ap.add_argument("--extra", nargs="+", default=[str(CLOCKS_TRAIN)],
                    help="Parquets of treatment columns keyed on MVT_ID_mvt.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    n_iter = lgb.Booster(model_file=str(V56_BOOSTER)).num_trees()
    dep, feat = load_gate_frame()
    extra = []
    for path in args.extra:
        cols = pd.read_parquet(C.ROOT / path)
        dep = dep.merge(cols, on="MVT_ID_mvt", how="left", validate="m:1")
        extra += [c for c in cols.columns if c != "MVT_ID_mvt"]
    dep = dep.merge(pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS]), on="MVT_ID_mvt",
                    how="left", validate="m:1")
    y, month = dep["is_fb"].values, dep["month"]
    sd = dep["sched_delay"].astype(float).values
    p = {arm: nested_map(score_oof(dep, cols, n_iter), y, month)
         for arm, cols in (("control", feat), ("treatment", feat + extra))}
    report = {"log_loss": {}}
    for label, lo, hi in zip(["sd<=3600", "3600<sd<=14400", "sd>14400"], SD_BINS, SD_BINS[1:]):
        m = (sd > lo) & (sd <= hi)
        report["log_loss"][label] = {arm: float(log_loss(y[m], np.clip(q[m], EPS, 1 - EPS), labels=[0, 1]))
                                     for arm, q in p.items()}
    served = (dep["month"].isin(HOLD) & (dep["TAXITIME_SEC_mvt"] > 0)
              & ~((dep["flt_null"] == 1) & (dep["sched_delay"] > C.STEP_A_SD_THRESHOLD))).values
    h = dep[served]
    r = r_norm_v41(h)
    t = h["TAXITIME_SEC_mvt"].values.astype(float)
    mix = {arm: q[served] * sd[served] + (1 - q[served]) * r for arm, q in p.items()}
    report["columns"] = extra
    report["n_mixture_rows"] = int(served.sum())
    report["mixture_treatment_minus_control"] = mse_delta(mix["treatment"], mix["control"], t)
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / f"gate_{args.tag}.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
