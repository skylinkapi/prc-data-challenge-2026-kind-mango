"""Paired price of the NM clock columns (WINNING_PLAN L1, L2) on the 2025 hold-out months 1 and 7.

Three arms, one LightGBM member each with the v43 parameters, on the MB3
rows of months 2 to 6 and 8 to 10, at a fixed 2,000 rounds (as in
`measure_mf2.py`): the v65 columns (control), plus the own clocks (L1),
plus the own and the neighbour clocks (L1 + L2). The report gives each
treatment minus the control at four round counts, overall and per airport,
on the hold-out rows outside LIRF. The arm predictions at 2,000 rounds go
to a parquet for the blend with CatBoost.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import FEATURES as CLOCKS, OUT_TRAIN as CLOCKS_TRAIN
from src_v3.build_nm_neighbours_v73 import FEATURES as NB, OUT_TRAIN as NB_TRAIN
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, holdout_rows
from src_v3.measure_lgbm_control import fit_early_stopped
from src_v3.measure_mf2 import CHECKPOINTS, ROUNDS
from src_v3.measure_tail_rules import mse_delta

PRED_OUT = C.MODELS / "nm_clocks_holdout.parquet"
log = logging.getLogger(__name__)


def add_clocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Frame with the own and the neighbour clock columns merged on `MVT_ID_mvt`."""
    for path in (CLOCKS_TRAIN, NB_TRAIN):
        frame = frame.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame()
    train = add_clocks(dep[dep["month"].isin(FIT_MONTHS)])
    del dep
    h, v46_pred = holdout_rows()
    h = add_clocks(h)
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    apt = h["ADEP_mvt"].astype(str).values
    cols = {"control": feat, "clocks": feat + CLOCKS, "clocks_nb": feat + CLOCKS + NB}
    report = {"n_holdout": len(h), "minus_control": {}}
    preds = {}
    for arm, c in cols.items():
        b = fit_early_stopped(train, None, c, rounds=ROUNDS)
        preds[arm] = {n: np.clip(b.predict(h[c], num_iteration=n), 0, None) for n in CHECKPOINTS}
        log.info("%s fitted", arm)
    for arm in ("clocks", "clocks_nb"):
        report["minus_control"][arm] = {str(n): {
            "all": mse_delta(preds[arm][n], preds["control"][n], y),
            **{a: mse_delta(preds[arm][n][apt == a], preds["control"][n][apt == a], y[apt == a])
               for a in sorted(set(apt))}} for n in CHECKPOINTS}
    report["rmse_at_2000"] = {arm: float(np.sqrt(((p[ROUNDS] - y) ** 2).mean())) for arm, p in preds.items()}
    report["rmse_v46"] = float(np.sqrt(((v46_pred - y) ** 2).mean()))
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "nm_clocks.holdout.json").write_text(json.dumps(report, indent=1))
    pd.DataFrame({"MVT_ID_mvt": h["MVT_ID_mvt"].values, "y": y, "v46": v46_pred,
                  **{arm: p[ROUNDS] for arm, p in preds.items()}}).to_parquet(PRED_OUT)


if __name__ == "__main__":
    main()
