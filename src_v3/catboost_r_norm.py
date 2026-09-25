"""MB7 on the LIRF head: CatBoost as a second class for `R_norm`.

Same rows (LIRF genuine: `abs(y - sd) >= 60`, `y < 80,000`) and 166 columns
as the v55 `R_norm` members. The full-year fit takes its round count from
the hold-out fit (`measure_catboost_r_norm.py`), scaled by the row ratio.
`blend_r_norm` is the serving hook that `predict_v30.main` calls.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lirf_regime import FB_TOL
from train_r_all_v40 import TEMPO_COLS, V2_FRAME

from src_v3 import config as C
from src_v3.catboost_base import fit, predict

LIRF_CACHE = C.ROOT / "models" / "lirf_frame_cache.parquet"
FEATURES = C.ROOT / "models" / "lirf_regime_v41.features.txt"
MODEL = C.ROOT / "models" / "catboost_r_norm_lirf_v68.cbm"
HOLDOUT_REPORT = C.MODELS / "catboost_r_norm.holdout.json"
log = logging.getLogger(__name__)


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    """LIRF genuine rows with the tempo columns, and the 166 `R_norm` feature names."""
    dep = pd.read_parquet(LIRF_CACHE).merge(
        pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS]), on="MVT_ID_mvt", how="left")
    y, sd = dep["TAXITIME_SEC_mvt"].astype(float), dep["sched_delay"].astype(float)
    dep = dep[((y - sd).abs() >= FB_TOL) & (y < 80_000)].reset_index(drop=True)
    return dep, [x for x in FEATURES.read_text().splitlines() if x]


def blend_r_norm(weight: float):
    """Return the hook `(frame, r_norm) -> (1 - w) * r_norm + w * CatBoost` for `predict_v30.main`."""
    model = CatBoostRegressor()
    model.load_model(str(MODEL))
    feat = [x for x in FEATURES.read_text().splitlines() if x]

    def hook(frame: pd.DataFrame, r_norm: np.ndarray) -> np.ndarray:
        return (1 - weight) * r_norm + weight * predict(model, frame, feat)
    return hook


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = json.loads(HOLDOUT_REPORT.read_text())
    dep, feat = load_frame()
    n_iter = int(math.ceil(report["best_iteration"] * len(dep) / report["n_fit"]))
    log.info("full-year LIRF CatBoost: %d rows, %d rounds", len(dep), n_iter)
    fit(dep, feat, iterations=n_iter).save_model(str(MODEL))
    (C.ROOT / "models" / "catboost_r_norm_lirf_v68.meta.json").write_text(json.dumps({
        "params": C.CATBOOST_PARAMS, "iterations": n_iter, "n_rows": len(dep)}, indent=1))


if __name__ == "__main__":
    main()
