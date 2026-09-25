"""Price a CatBoost blend on the LIRF `R_norm` term on the 2025 hold-out months 1 and 7.

CatBoost trains on the LIRF genuine rows of months 2 to 6 and 8 to 10 and
early-stops on months 11 and 12, as the v41 `R_norm` members did. The v41
members score the same hold-out rows. The report gives the MSE delta of
`(1 - w) * v41 + w * CatBoost` on the genuine rows, on the 344,336-row
scale, for each pre-registered weight.
"""
from __future__ import annotations

import json
import logging

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS

from src_v3 import config as C
from src_v3.catboost_base import fit, predict
from src_v3.catboost_r_norm import HOLDOUT_REPORT, load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS
from src_v3.measure_tail_rules import HOLD, mse_delta

V41_SEEDS = (42, 43, 44, 45, 46)
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame()
    train, stop = dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)]
    h = dep[dep["month"].isin(HOLD)].reset_index(drop=True)
    x = h[feat].copy()
    for c in CAT_COLS:
        x[c] = x[c].astype("category")
    v41 = np.mean([np.clip(lgb.Booster(model_file=str(C.ROOT / "models" / f"lgbm_r_norm_lirf_v41_s{s}.txt"))
                           .predict(x), 0, None) for s in V41_SEEDS], axis=0)
    model = fit(train, feat, stop=stop)
    cat = predict(model, h, feat)
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    report = {"best_iteration": int(model.get_best_iteration()) + 1, "n_fit": int(len(train)),
              "n_holdout": int(len(h)), "v41_rmse": float(np.sqrt(np.mean((v41 - y) ** 2))),
              "catboost_rmse": float(np.sqrt(np.mean((cat - y) ** 2))),
              "blend": {str(w): mse_delta((1 - w) * v41 + w * cat, v41, y)
                        for w in (*C.CATBOOST_BLEND_WEIGHTS, 1.0)}}
    log.info("%s", json.dumps(report, indent=1))
    HOLDOUT_REPORT.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
