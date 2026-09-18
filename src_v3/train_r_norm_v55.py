"""MB8 applied to R_norm_LIRF: retrain on all 12 months, no early-stop.

Mirrors the L2 12-month refit that gave the base -2.57 s live. Trains 5
seeds on every LIRF genuine row (|y - sd| >= 60 and y < 80,000) across
all 12 months, using the fixed round counts scaled from the shipped v41
iters (234, 371, 430, 286, 235) by 1/0.88 = 1.136. Same recipe as v41
otherwise: linear_tree, 127 leaves, linear_lambda 1.0, deterministic.

Writes lgbm_r_norm_lirf_v55_s{42..46}.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
from train_lirf_regime import FB_TOL, MODELS as MODELS_OLD
from train_r_all_v40 import TEMPO_COLS, V2_FRAME
from train_r_norm_lirf_seeds import SEEDS

from src_v3 import config as C

LIRF_CACHE = C.ROOT / "models" / "lirf_frame_cache.parquet"
V41_HOLDOUT = C.ROOT / "models" / "lirf_regime_v41.holdout.json"
SCALE = 1.0 / (1.0 - 0.12)   # ~ 1.136
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(V41_HOLDOUT) as f:
        v41_iters = json.load(f)["v41"]["best_iters"]
    scaled = [int(math.ceil(i * SCALE)) for i in v41_iters]
    log.info("v41 iters %s -> v55 %s (x %.3f)", v41_iters, scaled, SCALE)

    log.info("loading LIRF frame cache + tempo columns")
    dep = pd.read_parquet(LIRF_CACHE)
    tempo = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    log.info("LIRF frame %d rows", len(dep))

    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")

    # Genuine LIRF rows only, ALL 12 months (no hold-out or stop split)
    y = dep["TAXITIME_SEC_mvt"].astype(float).values
    sd = dep["sched_delay"].astype(float).values
    genuine = (np.abs(y - sd) >= FB_TOL) & (y < 80000)
    dep = dep[genuine].reset_index(drop=True)
    log.info("genuine rows: %d (all 12 months)", len(dep))

    with open(C.ROOT / "models" / "lirf_regime_v41.features.txt") as f:
        feat = [x for x in f.read().splitlines() if x]
    log.info("R_norm features: %d", len(feat))

    dt = lgb.Dataset(dep[feat], label=dep["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})

    for seed, n_iter in zip(SEEDS, scaled):
        params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0, "num_leaves": 127,
                  "bagging_freq": 5, "verbosity": -1,
                  "num_threads": C.NUM_THREADS,
                  "deterministic": C.DETERMINISTIC,
                  "seed": seed, "bagging_seed": seed,
                  "feature_fraction_seed": seed}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=n_iter)
        b.save_model(str(C.ROOT / "models" / f"lgbm_r_norm_lirf_v55_s{seed}.txt"))
        log.info("v55 seed %d %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    (C.ROOT / "models" / "lgbm_r_norm_lirf_v55.meta.json").write_text(json.dumps({
        "v41_iters": v41_iters, "scaled_iters": scaled, "scale_factor": SCALE,
        "n_rows": len(dep), "seeds": list(SEEDS),
        "reuses_v41_features": True,
    }, indent=1))
    log.info("wrote v55 R_norm members")


if __name__ == "__main__":
    main()
