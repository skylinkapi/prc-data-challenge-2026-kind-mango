"""Price stack variants for v96 against the v95 control on the 62 January and July 2025 ADS-B days.

Arms change one thing each: the A4 neighbour columns, 63 leaves, a residual
clip of 7,200 s, the A5 stand bound raised by an offset (fixed, or the
per-airport quantile of the label excess over the bound, fitted on the
other month), or the ADS-B specialist column (`adsb_specialist.py`). Every number is the MSE change against the control on the
hold-out scale; the bar for a change is `AUTO_BAR_MSE`.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3 import stack_pushback_v90 as sp
from src_v3.measure_tail_rules import mse_delta
from src_v3.tune_stack_v93 import frame

REPORT = C.MODELS / "tune_stack_v96.holdout.json"
log = logging.getLogger(__name__)


def served(s: pd.DataFrame, offset: np.ndarray | float = 0.0) -> np.ndarray:
    return np.fmax(s["stack"].values, sp.stand_bound(s) + offset)


def airport_offset(s: pd.DataFrame, q: float) -> np.ndarray:
    """Per-airport quantile `q` of `y - bound`, fitted on the other month, floored at 0."""
    excess = s["y"] - sp.stand_bound(s)
    out = np.zeros(len(s))
    for g in (True, False):
        fit = excess[s["fold_jan"] != g].groupby(s["ADEP_mvt"][s["fold_jan"] != g]).quantile(q).clip(lower=0)
        m = (s["fold_jan"] == g).values
        out[m] = s.loc[m, "ADEP_mvt"].map(fit).fillna(0).values
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    f = frame(C.AUTO_MLP_WEIGHT, False)
    ctrl, fold = sp.crossfit(f)
    y, ref = ctrl["y"].values, served(ctrl)
    arms = {"control_vs_base": mse_delta(ref, ctrl["base"].values, y)}
    for k in C.STACK_V96_BOUND_OFFSETS_S:
        arms[f"bound_offset_{k}"] = mse_delta(served(ctrl, k), ref, y)
    for q in C.STACK_V96_BOUND_QUANTILES:
        arms[f"bound_airport_q{q}"] = mse_delta(served(ctrl, airport_offset(ctrl, q)), ref, y)
    variants = {"leaves_63": dict(params={**C.STACK_V90_PARAMS, "num_leaves": 63}),
                "res_clip_7200": dict(res_clip=7200)}
    for name, kw in variants.items():
        s, _ = sp.crossfit(f, **kw)
        arms[name] = mse_delta(served(s), ref, y)
    s, _ = sp.crossfit(frame(C.AUTO_MLP_WEIGHT, True))
    arms["a4_neighbours"] = mse_delta(served(s), ref, y)
    spec = sp.spec_paths(C.SPEC_TAG)["holdout"]
    if spec.exists():
        s, _ = sp.crossfit(frame(C.AUTO_MLP_WEIGHT, False, (spec,)))
        arms["adsb_specialist"] = mse_delta(served(s), ref, y)
        arms["adsb_specialist_per_airport"] = {
            a: mse_delta(served(s)[m], ref[m], y[m]) for a in sorted(set(ctrl["ADEP_mvt"].astype(str)))
            for m in [(ctrl["ADEP_mvt"].astype(str) == a).values]}
    report = {"fold": fold, "rows": int(len(ctrl)), "arms": arms}
    log.info("%s", json.dumps(report, indent=1))
    REPORT.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
