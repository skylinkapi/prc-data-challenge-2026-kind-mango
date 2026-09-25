"""Price a second base CatBoost (variant B, option B) on the 2025 hold-out months 1 and 7.

Member A repeats the v67 hold-out fit (config parameters, 5,000-round cap).
Member B uses `CATBOOST_PARAMS_B` with a 15,000-round cap. Both train on
months 2 to 6 and 8 to 10 and early-stop on months 11 and 12. The report
prices, against the v46 members:

- `a`: 0.5 v46 + 0.5 A, the v67 structure (reference);
- `ab`: 0.5 v46 + 0.25 A + 0.25 B, the v70 candidate;
- `b`: 0.5 v46 + 0.5 B.

v70 is built only if `ab` beats `a` by at least `MIN_GAIN_MSE`.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src_v3 import config as C
from src_v3.catboost_base import fit, load_frame, paths, predict
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS, holdout_rows
from src_v3.measure_tail_rules import mse_delta

B_ROUND_CAP = 15_000
MIN_GAIN_MSE = 150
PREDICTIONS = C.MODELS / "catboost_pair_holdout.parquet"
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame()
    train, stop = dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)]
    h, v46 = holdout_rows()
    del dep
    model_a = fit(train, feat, stop=stop)
    pred_a = predict(model_a, h, feat)
    del model_a
    model_b = fit(train, feat, stop=stop, max_rounds=B_ROUND_CAP, params=C.CATBOOST_PARAMS_B)
    pred_b = predict(model_b, h, feat)
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    pd.DataFrame({"MVT_ID_mvt": h["MVT_ID_mvt"].values, "y": y, "v46": v46,
                  "cat_a": pred_a, "cat_b": pred_b}).to_parquet(PREDICTIONS)
    blends = {"a": 0.5 * v46 + 0.5 * pred_a,
              "ab": 0.5 * v46 + 0.25 * pred_a + 0.25 * pred_b,
              "b": 0.5 * v46 + 0.5 * pred_b}
    report = {"best_iteration": int(model_b.get_best_iteration()) + 1, "n_fit": int(len(train)),
              "n_holdout": int(len(h)), "min_gain_mse": MIN_GAIN_MSE,
              "blend_vs_v46": {k: mse_delta(p, v46, y) for k, p in blends.items()}}
    report["ab_minus_a"] = report["blend_vs_v46"]["ab"] - report["blend_vs_v46"]["a"]
    report["build_v70"] = report["ab_minus_a"] <= -MIN_GAIN_MSE
    log.info("%s", json.dumps(report, indent=1))
    paths("v70")[1].write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
