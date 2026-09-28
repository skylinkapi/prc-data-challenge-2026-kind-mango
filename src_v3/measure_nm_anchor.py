"""Paired price of the residual target on the `AOBT_3` anchor (WINNING_PLAN L5) on the 2025 hold-out.

The anchor is `clip(MVT - AOBT_3, 0, 7,200)`. One LightGBM member with the
v43 parameters and the v73 columns learns `y - anchor` on the MB3 rows of
months 2 to 6 and 8 to 10 that have an anchor, at a fixed 2,000 rounds.
It serves `anchor + offset`, and the clock member (`measure_nm_clocks.py`)
where the anchor is null. The report gives the anchored member and its
blends with the clock member, minus the clock member, on the hold-out rows
outside LIRF.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import FEATURES as CLOCKS
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, holdout_rows
from src_v3.measure_lgbm_control import fit_early_stopped
from src_v3.measure_mf2 import ROUNDS
from src_v3.measure_nm_clocks import PRED_OUT, add_clocks
from src_v3.measure_tail_rules import mse_delta

BLEND_WEIGHTS = (0.3, 0.5, 1.0)
log = logging.getLogger(__name__)


def anchor(frame: pd.DataFrame) -> np.ndarray:
    return frame["mvt_aobt3"].clip(0, C.NM_CLOCK_CLIP_S).values


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame()
    train = add_clocks(dep[dep["month"].isin(FIT_MONTHS)])
    del dep
    train = train[train["mvt_aobt3"].notna()].copy()
    train["TAXITIME_SEC_mvt"] = train["TAXITIME_SEC_mvt"] - anchor(train)
    cols = feat + CLOCKS
    b = fit_early_stopped(train, None, cols, rounds=ROUNDS)
    h, _ = holdout_rows()
    h = add_clocks(h)
    ref = pd.read_parquet(PRED_OUT)
    if not (ref["y"].values == h["TAXITIME_SEC_mvt"].values).all():
        raise AssertionError("The saved clock predictions are not in hold-out row order. "
                             "Rerun measure_nm_clocks, then rerun.")
    y, clocks = ref["y"].values, ref["clocks"].values
    anchored = np.where(h["mvt_aobt3"].notna(), np.clip(anchor(h) + b.predict(h[cols]), 0, None), clocks)
    apt = h["ADEP_mvt"].astype(str).values
    report = {"n_anchor_train": len(train), "minus_clocks": {}}
    for w in BLEND_WEIGHTS:
        p = (1 - w) * clocks + w * anchored
        report["minus_clocks"][str(w)] = {"all": mse_delta(p, clocks, y), **{
            a: mse_delta(p[apt == a], clocks[apt == a], y[apt == a]) for a in sorted(set(apt))}}
    report["rows_over_7200"] = {"clocks": int((clocks > 7200).sum()), "anchored": int((anchored > 7200).sum()),
                                "labels": int((y > 7200).sum())}
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "nm_anchor.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
