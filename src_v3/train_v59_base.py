"""MF4: base retrained with 4 arrival-drift features on top of v57.

Adds `arr_txi_1d_med`, `arr_txi_7d_med`, `arr_txi_1d_drift`, `arr_txi_7d_drift`
to the v57 feature set (117 -> 121 columns). Same v51/v57 recipe otherwise
(MB3 filter, retuned params, 3 seeds).
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
from src_v3.frames import (H1, H1_IDS, V2_TRAIN, TEMPO_P2575,
                           V40_TEMPO, V44_QUARTILES, V45_PLAN, v47_feature_names)

TUNED = C.ROOT / "models" / "tune_lgbm_v43.tuned_params.json"
V46_REPORT = C.ROOT / "models" / "lgbm_r_all_v46.holdout.json"
PLAN_TRAIN_V57 = C.ROOT / "models" / "plan_taxi_res_train_v57.parquet"
ARR_DRIFT_TRAIN = C.ROOT / "models" / "arr_drift_train.parquet"
DRIFT_COLS = ["arr_txi_1d_med", "arr_txi_7d_med",
              "arr_txi_1d_drift", "arr_txi_7d_drift"]
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(TUNED) as f:
        tuned = json.load(f)["tuned"]
    with open(V46_REPORT) as f:
        v46_iters = json.load(f)["v46"]["best_iters"]

    log.info("loading v59 feature frame (v57 base + arrival drift)")
    dep = pd.read_parquet(H1)
    dep["MVT_ID_mvt"] = pd.read_parquet(H1_IDS)["MVT_ID_mvt"].values
    tempo = pd.read_parquet(V2_TRAIN, columns=["MVT_ID_mvt", *V40_TEMPO])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    quart = pd.read_parquet(TEMPO_P2575, columns=["MVT_ID_mvt", *V44_QUARTILES])
    dep = dep.merge(quart, on="MVT_ID_mvt", how="left")
    plan = pd.read_parquet(PLAN_TRAIN_V57, columns=["MVT_ID_mvt", *V45_PLAN])
    dep = dep.merge(plan, on="MVT_ID_mvt", how="left")
    drift = pd.read_parquet(ARR_DRIFT_TRAIN, columns=["MVT_ID_mvt", *DRIFT_COLS])
    dep = dep.merge(drift, on="MVT_ID_mvt", how="left")
    log.info("frame %d rows", len(dep))

    adep = dep["ADEP_mvt"].astype(str)
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    keep = (adep != "LIRF") & (y > 0) & (y <= 80_000)
    dep = dep[keep].reset_index(drop=True)
    log.info("MB3 filter: %d rows", len(dep))

    feat = v47_feature_names() + DRIFT_COLS
    log.info("feature count: %d (v47 117 + %d drift)", len(feat), len(DRIFT_COLS))

    n_v46_train = int(0.88 * 10 / 12 * 2_084_659)
    v59_scale = len(dep) / n_v46_train
    scaled = [int(math.ceil(i * v59_scale)) for i in v46_iters]
    log.info("v46 iters %s -> v59 %s (x %.3f)", v46_iters, scaled, v59_scale)

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
        b.save_model(str(C.ROOT / "models" / f"lgbm_r_all_v59_s{seed}.txt"))
        log.info("v59 seed %d %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    with open(C.ROOT / "models" / "lgbm_r_all_v59.features.txt", "w") as f:
        f.write("\n".join(feat))
    shutil.copy(str(C.ROOT / "models" / "lgbm_r_all_v26.encoders.pkl"),
                str(C.ROOT / "models" / "lgbm_r_all_v59.encoders.pkl"))
    (C.ROOT / "models" / "lgbm_r_all_v59.meta.json").write_text(json.dumps({
        "tuned": tuned, "v46_iters": v46_iters, "scaled_iters": scaled,
        "scale_factor": v59_scale, "n_served": len(dep),
        "changed": "adds arr_txi 1d/7d drift features (MF4)",
        "features": feat,
    }, indent=1))
    log.info("wrote v59 base members")


if __name__ == "__main__":
    main()
