"""L2: 12-month refit of the v46 stack. MODEL_ANALYSIS 5.2 step 6.

Trains three seeds on ALL months with the retuned params (from
tune_lgbm_v43.tuned_params.json), no early-stop split. The iteration counts
come from the v46 paired run (models/lgbm_r_all_v46.holdout.json) scaled by
1/0.88 to compensate for the larger training set. This ship removes the
hold-out; the live delta on the leaderboard is the price.
"""
import json
import logging
import math
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_lgbm_v21 import CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS, MODELS, STOP_FRAC
from train_r_all_v40 import CACHE, ID_MAP, TEMPO_COLS, V2_FRAME
from train_r_all_v44 import EXTRA_COLS as EOBT_P2575
from train_r_all_v45plan import EXTRA_COLS as PLAN_COLS
from train_r_all_v45plan import PLAN

SCALE = 1.0 / (1.0 - STOP_FRAC)  # ~1.136
TUNED = os.path.join(MODELS, "tune_lgbm_v43.tuned_params.json")
V46_REPORT = os.path.join(MODELS, "lgbm_r_all_v46.holdout.json")
log = logging.getLogger(__name__)


def load_all_months() -> tuple[pd.DataFrame, list[str]]:
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().split()
    dep = pd.read_parquet(CACHE)
    dep["MVT_ID_mvt"] = pd.read_parquet(ID_MAP)["MVT_ID_mvt"].values
    tempo = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    dep = dep.merge(pd.read_parquet(os.path.join(MODELS, "tempo_p2575_train.parquet")),
                    on="MVT_ID_mvt", how="left")
    dep = dep.merge(pd.read_parquet(PLAN), on="MVT_ID_mvt", how="left")
    return dep, feat


def main() -> None:
    with open(TUNED) as f:
        tuned = json.load(f)["tuned"]
    with open(V46_REPORT) as f:
        v46 = json.load(f)
    v46_iters = v46["v46"]["best_iters"]
    scaled = [int(math.ceil(i * SCALE)) for i in v46_iters]
    log.info("v46 iters %s -> scaled %s (x %.3f)", v46_iters, scaled, SCALE)

    dep, feat = load_all_months()
    v47_cols = feat + TEMPO_COLS + EOBT_P2575 + PLAN_COLS
    log.info("training on all %d rows, %d features", len(dep), len(v47_cols))

    dt = lgb.Dataset(dep[v47_cols], label=dep["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})

    for seed, n_iter in zip(DEFAULT_SEEDS, scaled):
        p = {**tuned, "seed": seed, "bagging_seed": seed,
             "feature_fraction_seed": seed, "linear_tree": True,
             "bagging_freq": 5, "verbosity": -1, "num_threads": -1}
        t0 = time.time()
        b = lgb.train(p, dt, num_boost_round=n_iter)
        b.save_model(os.path.join(MODELS, f"lgbm_r_all_v47_s{seed}.txt"))
        log.info("v47 seed %d trained for %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    with open(os.path.join(MODELS, "lgbm_r_all_v47.features.txt"), "w") as f:
        f.write("\n".join(v47_cols))
    shutil.copy(os.path.join(MODELS, "lgbm_r_all_v26.encoders.pkl"),
                os.path.join(MODELS, "lgbm_r_all_v47.encoders.pkl"))
    meta = {"tuned": tuned, "v46_iters": v46_iters, "scaled_iters": scaled,
            "scale_factor": SCALE, "features": v47_cols}
    with open(os.path.join(MODELS, "lgbm_r_all_v47.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log.info("wrote v47 members (%d cols, %d rows trained)", len(v47_cols), len(dep))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
