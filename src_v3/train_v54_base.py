"""MB1: base retrained on the anchored offset target.

Target: y - anchor, where anchor = mvt_eobt1 if that column is not-null,
else sched_delay. The offset has a tight distribution (5th to 95th
percentile ~-1500 to +250 s) once LIRF and y > 80,000 rows leave via MB3.
A tree on the offset does not need a linear leaf to reproduce the y = sd
slope on fallback rows; it just needs to predict 0.

Recipe: MB3 filter (v51), retuned Optuna params, 3 seeds, all 12 months,
deterministic. Writes lgbm_r_all_v54_s{42,43,44} and a
`v54_offset_bounds.json` with per-airport 0.1/99.9 % offset quantiles
that predict time uses to clip the served prediction.

Feature list matches v47 (117 columns). Anchor and offset live outside
the feature set - the anchor is a per-row scalar the predict script
recovers from mvt_eobt1 and sched_delay.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS

from src_v3 import config as C
from src_v3.frames import load_v47_frame, v47_feature_names

TUNED = C.ROOT / "models" / "tune_lgbm_v43.tuned_params.json"
V46_REPORT = C.ROOT / "models" / "lgbm_r_all_v46.holdout.json"
log = logging.getLogger(__name__)


def _compute_anchor(dep: pd.DataFrame) -> np.ndarray:
    sd = dep["sched_delay"].astype(float).values
    mvt_eobt1 = dep["mvt_eobt1"].astype(float).values
    return np.where(np.isnan(mvt_eobt1), sd, mvt_eobt1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(TUNED) as f:
        tuned = json.load(f)["tuned"]
    with open(V46_REPORT) as f:
        v46_iters = json.load(f)["v46"]["best_iters"]

    log.info("loading v47 feature frame")
    dep = load_v47_frame()
    feat = v47_feature_names()

    adep = dep["ADEP_mvt"].astype(str)
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    keep = (adep != "LIRF") & (y > 0) & (y <= 80_000)
    dep = dep[keep].reset_index(drop=True)
    log.info("MB3 filter: %d rows", len(dep))

    anchor = _compute_anchor(dep)
    offset = dep["TAXITIME_SEC_mvt"].astype(float).values - anchor
    log.info("offset percentiles: q0.001=%.0f q0.5=%.0f q0.999=%.0f",
             np.quantile(offset, 0.001), np.quantile(offset, 0.5),
             np.quantile(offset, 0.999))

    # Per-airport [q0.001, q0.999] offset bounds for predict-time clipping
    bounds: dict[str, dict[str, float]] = {}
    apt = dep["ADEP_mvt"].astype(str).values
    for a in np.unique(apt):
        m = (apt == a)
        bounds[a] = {"q_low": float(np.quantile(offset[m], 0.001)),
                     "q_high": float(np.quantile(offset[m], 0.999)),
                     "n": int(m.sum())}
    (C.ROOT / "models" / "v54_offset_bounds.json").write_text(json.dumps(bounds, indent=1))

    # Scale iterations like v51 (served-row ratio vs v46's training footprint)
    n_v46_train = int(0.88 * 10 / 12 * 2_084_659)
    v54_scale = len(dep) / n_v46_train
    scaled = [int(math.ceil(i * v54_scale)) for i in v46_iters]
    log.info("v46 iters %s -> v54 %s (x %.3f)", v46_iters, scaled, v54_scale)

    dt = lgb.Dataset(dep[feat], label=offset,
                     categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})

    for seed, n_iter in zip(DEFAULT_SEEDS, scaled):
        p = {**tuned, "seed": seed, "bagging_seed": seed,
             "feature_fraction_seed": seed, "linear_tree": True,
             "bagging_freq": 5, "verbosity": -1,
             "num_threads": C.NUM_THREADS,
             "deterministic": C.DETERMINISTIC}
        t0 = time.time()
        b = lgb.train(p, dt, num_boost_round=n_iter)
        b.save_model(str(C.ROOT / "models" / f"lgbm_r_all_v54_s{seed}.txt"))
        log.info("v54 seed %d %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    with open(C.ROOT / "models" / "lgbm_r_all_v54.features.txt", "w") as f:
        f.write("\n".join(feat))
    shutil.copy(str(C.ROOT / "models" / "lgbm_r_all_v26.encoders.pkl"),
                str(C.ROOT / "models" / "lgbm_r_all_v54.encoders.pkl"))
    (C.ROOT / "models" / "lgbm_r_all_v54.meta.json").write_text(json.dumps({
        "tuned": tuned, "v46_iters": v46_iters, "scaled_iters": scaled,
        "scale_factor": v54_scale, "n_served": len(dep),
        "target": "y - anchor, anchor = mvt_eobt1 if not-null else sched_delay",
        "filter": "adep != 'LIRF' and 0 < y <= 80_000",
        "per_airport_offset_bounds": bounds,
    }, indent=1))
    log.info("wrote v54 members and metadata")


if __name__ == "__main__":
    main()
