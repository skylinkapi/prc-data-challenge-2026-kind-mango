"""MB8 for route medians: retrain v51 base with the v57 plan_taxi_res.

Same MB3 filter and same tuned params as v51 (lgbm_r_all_v51). Only the
`plan_taxi_res` column changes because its route medians now include
months 1 and 7. Everything else in the feature frame is identical.
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
from src_v3.frames import (H1, H1_FEAT, H1_IDS, V2_TRAIN, TEMPO_P2575,
                           V40_TEMPO, V44_QUARTILES, V45_PLAN, v47_feature_names)

TUNED = C.ROOT / "models" / "tune_lgbm_v43.tuned_params.json"
V46_REPORT = C.ROOT / "models" / "lgbm_r_all_v46.holdout.json"
PLAN_TRAIN_V57 = C.ROOT / "models" / "plan_taxi_res_train_v57.parquet"
log = logging.getLogger(__name__)


def _load_frame_v57() -> pd.DataFrame:
    dep = pd.read_parquet(H1)
    dep["MVT_ID_mvt"] = pd.read_parquet(H1_IDS)["MVT_ID_mvt"].values
    tempo = pd.read_parquet(V2_TRAIN, columns=["MVT_ID_mvt", *V40_TEMPO])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    quart = pd.read_parquet(TEMPO_P2575, columns=["MVT_ID_mvt", *V44_QUARTILES])
    dep = dep.merge(quart, on="MVT_ID_mvt", how="left")
    plan_v57 = pd.read_parquet(PLAN_TRAIN_V57, columns=["MVT_ID_mvt", *V45_PLAN])
    dep = dep.merge(plan_v57, on="MVT_ID_mvt", how="left")
    return dep


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(TUNED) as f:
        tuned = json.load(f)["tuned"]
    with open(V46_REPORT) as f:
        v46_iters = json.load(f)["v46"]["best_iters"]

    log.info("loading v57 feature frame (with all-months plan medians)")
    dep = _load_frame_v57()
    feat = v47_feature_names()
    log.info("frame %d rows, features %d", len(dep), len(feat))

    adep = dep["ADEP_mvt"].astype(str)
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    keep = (adep != "LIRF") & (y > 0) & (y <= 80_000)
    dep = dep[keep].reset_index(drop=True)
    log.info("MB3 filter: %d rows", len(dep))

    n_v46_train = int(0.88 * 10 / 12 * 2_084_659)
    v57_scale = len(dep) / n_v46_train
    scaled = [int(math.ceil(i * v57_scale)) for i in v46_iters]
    log.info("v46 iters %s -> v57 %s (x %.3f)", v46_iters, scaled, v57_scale)

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
        b.save_model(str(C.ROOT / "models" / f"lgbm_r_all_v57_s{seed}.txt"))
        log.info("v57 seed %d %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    with open(C.ROOT / "models" / "lgbm_r_all_v57.features.txt", "w") as f:
        f.write("\n".join(feat))
    shutil.copy(str(C.ROOT / "models" / "lgbm_r_all_v26.encoders.pkl"),
                str(C.ROOT / "models" / "lgbm_r_all_v57.encoders.pkl"))
    (C.ROOT / "models" / "lgbm_r_all_v57.meta.json").write_text(json.dumps({
        "tuned": tuned, "v46_iters": v46_iters, "scaled_iters": scaled,
        "scale_factor": v57_scale, "n_served": len(dep),
        "changed": "plan_taxi_res uses all-12-month route medians (MB8)",
    }, indent=1))
    log.info("wrote v57 base members")


if __name__ == "__main__":
    main()
