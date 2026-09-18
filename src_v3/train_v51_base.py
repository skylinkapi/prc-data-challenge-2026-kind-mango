"""MB3: base retrained on rows it actually serves.

The v47 base trains on every `y > 0` row across all 10 airports and all 12
months, but the served path overwrites every LIRF row with the LIRF head
and applies Step A on top (predict_v30.py:197-215). 16.6 % of LIRF rows
are fallback (T8) and 12 of the 14 24-h rows sit at LIRF (T6); the base
fits linear leaves on those rows and never predicts them. MB3 drops
LIRF and y > 80,000 rows from training and keeps the rest.

Recipe: same as v47 (retuned Optuna params from
tune_lgbm_v43.tuned_params.json, linear_tree=True, deterministic). Round
counts scale from the v46 paired report by the served-row growth ratio.
Saves lgbm_r_all_v51_s{42,43,44} plus features/encoders/meta.
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
from train_r_all_v26 import DEFAULT_SEEDS, MODELS as MODELS_OLD, STOP_FRAC

from src_v3 import config as C
from src_v3.frames import load_v47_frame, v47_feature_names

TUNED = C.ROOT / "models" / "tune_lgbm_v43.tuned_params.json"
V46_REPORT = C.ROOT / "models" / "lgbm_r_all_v46.holdout.json"
SCALE = 1.0 / (1.0 - STOP_FRAC)  # v47 scale, kept for meta
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(TUNED) as f:
        tuned = json.load(f)["tuned"]
    with open(V46_REPORT) as f:
        v46_iters = json.load(f)["v46"]["best_iters"]

    log.info("loading v47 feature frame")
    dep = load_v47_frame()
    feat = v47_feature_names()
    n_all = len(dep)
    adep = dep["ADEP_mvt"].astype(str)
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    keep = (adep != "LIRF") & (y > 0) & (y <= 80_000)
    dep = dep[keep].reset_index(drop=True)
    n_served = len(dep)
    log.info("filtered rows: %d -> %d (drop %d, %.2f %%)",
             n_all, n_served, n_all - n_served,
             100 * (n_all - n_served) / n_all)

    # Scale v46 iterations by served-row ratio. v46 was paired trained on
    # months {2..12} \ {1, 7} = 10 months with random 12 % stop, so its
    # training set was 0.88 * 10/12 of the 2,084,659 total = 1,528,483 rows.
    # v51 trains on all 12 months but only the served rows (~1,913,000).
    n_v46_train = int(0.88 * 10 / 12 * 2_084_659)
    v51_scale = n_served / n_v46_train
    scaled = [int(math.ceil(i * v51_scale)) for i in v46_iters]
    log.info("v46 iters %s -> v51 scaled %s (x %.3f)",
             v46_iters, scaled, v51_scale)

    dt = lgb.Dataset(dep[feat], label=dep["TAXITIME_SEC_mvt"].values,
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
        b.save_model(str(C.ROOT / "models" / f"lgbm_r_all_v51_s{seed}.txt"))
        log.info("v51 seed %d %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    with open(C.ROOT / "models" / "lgbm_r_all_v51.features.txt", "w") as f:
        f.write("\n".join(feat))
    shutil.copy(str(C.ROOT / "models" / "lgbm_r_all_v26.encoders.pkl"),
                str(C.ROOT / "models" / "lgbm_r_all_v51.encoders.pkl"))
    meta = {"tuned": tuned, "v46_iters": v46_iters, "scaled_iters": scaled,
            "scale_factor": v51_scale, "n_served": n_served,
            "filter": "adep != 'LIRF' and 0 < y <= 80_000",
            "features": feat}
    (C.ROOT / "models" / "lgbm_r_all_v51.meta.json").write_text(json.dumps(meta, indent=1))
    log.info("wrote v51 members and metadata")


if __name__ == "__main__":
    main()
