"""Fit the three served blend weights of the clock-era base on the 2025 hold-out (v77).

The served base outside LIRF is
`(1 - w) [(1 - a) LGB + a LGB_anchor] + w [(1 - b) Cat + b Cat_anchor]`,
with `(w, a, b)` = (0.5, 0.5, 0.5) in v75. A grid search picks the weights
on January and scores them on July, and the reverse, so each month is
priced out of sample. The report gives the cross-fitted MSE delta
against the v75 weights on the 344,336-row scale, and the weights fitted
on both months for serving.
"""
from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.measure_member_v77 import out_path
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta

log = logging.getLogger(__name__)


def blend(p: dict, w: float, a: float, b: float) -> np.ndarray:
    return (1 - w) * ((1 - a) * p["lgb"] + a * p["lgb_anchor"]) + w * ((1 - b) * p["cat"] + b * p["cat_anchor"])


def best_weights(p: dict, y: np.ndarray, m: np.ndarray) -> tuple:
    grid = itertools.product(C.BLEND_V77_GRID, repeat=3)
    return min(grid, key=lambda g: float(np.mean((blend(p, *g)[m] - y[m]) ** 2)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ref = pd.read_parquet(PRED_OUT)
    members = {k: pd.read_parquet(out_path(k)) for k in ("lgb_anchor", "cat_plain", "cat_anchor")}
    p = {"lgb": ref["clocks"].values, "lgb_anchor": members["lgb_anchor"]["pred"].values,
         "cat": members["cat_plain"]["pred"].values, "cat_anchor": members["cat_anchor"]["pred"].values}
    y, month = ref["y"].values, members["cat_plain"]["month"].values
    served = blend(p, *C.BLEND_V77_SERVED)
    cross = served.copy()
    fitted = {}
    for fit_m, test_m in ((1, 7), (7, 1)):
        fitted[str(fit_m)] = best_weights(p, y, month == fit_m)
        cross[month == test_m] = blend(p, *fitted[str(fit_m)])[month == test_m]
    final = best_weights(p, y, np.ones(len(y), bool))
    report = {"weights_fitted_on": fitted, "weights_all": final,
              "member_rmse": {k: float(np.sqrt(np.mean((v - y) ** 2))) for k, v in p.items()},
              "served_rmse": float(np.sqrt(np.mean((served - y) ** 2))),
              "cross_minus_served": mse_delta(cross, served, y),
              "cross_minus_served_by_month": {str(m): mse_delta(cross[month == m], served[month == m], y[month == m])
                                              for m in (1, 7)},
              "all_minus_served_in_sample": mse_delta(blend(p, *final), served, y)}
    report["ship"] = (report["cross_minus_served"] <= -C.TUNE_V76_MIN_GAIN_MSE
                      and all(v < 0 for v in report["cross_minus_served_by_month"].values()))
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "blend_v77.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
