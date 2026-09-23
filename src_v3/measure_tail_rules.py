"""Price two tail rules on the 2025 hold-out months 1 and 7 (Discord suggestion, 2026-09-23).

The v46 base members train on months 2 to 6 and 8 to 10, so they score
January and July out of sample. Both rules fit on the other ten months.

1. Blend: on rows with a flight record and `mvt_eobt1` > 3,600 s, mix the
   base with an isotonic curve of y on `mvt_eobt1` ("outliers like the
   training outliers").
2. Cap: on rows with a flight record and `mvt_eobt1` < 1,800 s, cap the
   base at the airport's 99.99 % label quantile.
"""
from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src_v3 import config as C
from src_v3.frames import load_v47_frame

HOLD = (1, 7)
N_HOLDOUT = 344_336
BLEND_WEIGHTS = (0.75, 0.5, 0.25, 0.0)
log = logging.getLogger(__name__)


def score_holdout(dep: pd.DataFrame) -> np.ndarray:
    """Mean of the three v46 base members, clipped at 0 as served."""
    feat = (C.ROOT / "models" / "lgbm_r_all_v46.features.txt").read_text().split()
    return np.mean([np.clip(lgb.Booster(model_file=str(C.ROOT / "models" / f"lgbm_r_all_v46_s{s}.txt"))
                            .predict(dep[feat]), 0, None) for s in (42, 43, 44)], axis=0)


def mse_delta(new: np.ndarray, old: np.ndarray, y: np.ndarray) -> float:
    return float((((new - y) ** 2).sum() - ((old - y) ** 2).sum()) / N_HOLDOUT)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep = load_v47_frame()
    dep = dep[(dep["ADEP_mvt"].astype(str) != "LIRF") & (dep["TAXITIME_SEC_mvt"] > 0)]
    is_hold = dep["month"].isin(HOLD)
    fit, h = dep[~is_hold & (dep["TAXITIME_SEC_mvt"] <= 80_000)], dep[is_hold]
    y, base = h["TAXITIME_SEC_mvt"].values.astype(float), score_holdout(h)
    has_rec = (h["flt_null"] == 0).values

    held = has_rec & (h["mvt_eobt1"] > 3_600).values
    f = fit[(fit["flt_null"] == 0) & (fit["mvt_eobt1"] > 3_000)]
    curve = IsotonicRegression(out_of_bounds="clip").fit(f["mvt_eobt1"], f["TAXITIME_SEC_mvt"])
    g = curve.predict(h["mvt_eobt1"].values[held])
    blend = {}
    for w in BLEND_WEIGHTS:
        p = base.copy()
        p[held] = w * base[held] + (1 - w) * g
        blend[str(w)] = mse_delta(p, base, y)

    normal = has_rec & (h["mvt_eobt1"] < 1_800).values
    fn = fit[(fit["flt_null"] == 0) & (fit["mvt_eobt1"] < 1_800)]
    cap = h["ADEP_mvt"].astype(str).map(fn.groupby("ADEP_mvt", observed=True)["TAXITIME_SEC_mvt"].quantile(0.9999)).values
    capped = np.where(normal, np.minimum(base, cap), base)
    report = {"blend_rows": int(held.sum()), "blend_mse_delta_by_base_weight": blend,
              "cap_rows_moved": int((capped != base).sum()), "cap_mse_delta": mse_delta(capped, base, y)}
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "tail_rules.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
