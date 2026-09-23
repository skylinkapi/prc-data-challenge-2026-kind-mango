"""L4 (WINNING_PLAN R1): v57 base recipe with the v45 parameters, not the v43 retune.

v45 to v46 changed only the base parameters and landed +3.01 s live. This
run keeps the v57 frame, the MB3 filter and 12 months, and swaps the
parameters back. Rounds are the v45plan best iterations scaled by the
same served-row ratio as v57.
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS

from src_v3 import config as C
from src_v3.frames import v47_feature_names
from src_v3.train_v57_base import _load_frame_v57

V45_REPORT = C.ROOT / "models" / "lgbm_r_all_v45plan.holdout.json"
N_V46_TRAIN = int(0.88 * 10 / 12 * 2_084_659)
V45_PARAMS = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "linear_lambda": 1.0, "num_leaves": 220}
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    v45_iters = json.loads(V45_REPORT.read_text())["v45plan"]["best_iters"]

    dep = _load_frame_v57()
    feat = v47_feature_names()
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    dep = dep[(dep["ADEP_mvt"].astype(str) != "LIRF") & (y > 0) & (y <= 80_000)]
    log.info("MB3 frame %d rows, %d features", len(dep), len(feat))

    scale = len(dep) / N_V46_TRAIN
    scaled = [int(math.ceil(i * scale)) for i in v45_iters]
    log.info("v45 iters %s -> v61 %s (x %.4f)", v45_iters, scaled, scale)

    dt = lgb.Dataset(dep[feat], label=dep["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})
    for seed, n_iter in zip(DEFAULT_SEEDS, scaled):
        p = {**V45_PARAMS, "seed": seed, "bagging_seed": seed,
             "feature_fraction_seed": seed, "linear_tree": True,
             "bagging_freq": 5, "verbosity": -1,
             "num_threads": C.NUM_THREADS, "deterministic": C.DETERMINISTIC}
        t0 = time.time()
        b = lgb.train(p, dt, num_boost_round=n_iter)
        b.save_model(str(C.ROOT / "models" / f"lgbm_r_all_v61_s{seed}.txt"))
        log.info("v61 seed %d %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    (C.ROOT / "models" / "lgbm_r_all_v61.features.txt").write_text("\n".join(feat))
    shutil.copy(str(C.ROOT / "models" / "lgbm_r_all_v26.encoders.pkl"),
                str(C.ROOT / "models" / "lgbm_r_all_v61.encoders.pkl"))
    (C.ROOT / "models" / "lgbm_r_all_v61.meta.json").write_text(json.dumps({
        "params": V45_PARAMS, "v45_iters": v45_iters, "scaled_iters": scaled,
        "scale_factor": scale, "n_served": len(dep),
        "changed": "v45 parameters on the v57 recipe (WINNING_PLAN L4)",
    }, indent=1))


if __name__ == "__main__":
    main()
