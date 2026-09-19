"""MB8 for operator target encoders + base retrain (v58).

Refits `features_operator.fit_encoders` on ALL 12 months of 2025 training
rows (was months 2..12 excluding 7 in the deployed v26 encoders). Applies
the new encoders to the training frame to recompute openc_* columns,
then retrains the v51-recipe base on the updated frame.

Saves:
  lgbm_r_all_v58.encoders.pkl
  lgbm_r_all_v58_s{42,43,44}.txt
  lgbm_r_all_v58.features.txt (same as v51/v57)
"""
from __future__ import annotations

import glob
import json
import logging
import math
import pickle
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from train_lgbm_v21 import CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS

from src_v3 import config as C
from src_v3.frames import (H1, H1_FEAT, H1_IDS, V2_TRAIN, TEMPO_P2575,
                           V40_TEMPO, V44_QUARTILES, V45_PLAN, v47_feature_names)

TUNED = C.ROOT / "models" / "tune_lgbm_v43.tuned_params.json"
V46_REPORT = C.ROOT / "models" / "lgbm_r_all_v46.holdout.json"
PLAN_TRAIN_V57 = C.ROOT / "models" / "plan_taxi_res_train_v57.parquet"
log = logging.getLogger(__name__)


def _load_raw_all_months() -> pd.DataFrame:
    """All 2025 DEP rows at target airports for encoder fitting."""
    parts = []
    for f in sorted(glob.glob(str(C.TRAIN_DIR / "*.parquet"))):
        cols = ["AIRCRAFT_OPERATOR_flt", "ADEP_mvt", "RUNWAY_mvt",
                "STAND_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt",
                "TAXITIME_SEC_mvt"]
        t = pd.read_parquet(f, columns=cols)
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(C.TARGETS)]
        t = t[t["TAXITIME_SEC_mvt"].astype(float) > 0]
        t["hour"] = pd.to_datetime(t["MVT_TIME_UTC_mvt"], utc=True,
                                    errors="coerce").dt.hour
        parts.append(t[["AIRCRAFT_OPERATOR_flt", "ADEP_mvt", "RUNWAY_mvt",
                        "STAND_mvt", "hour", "TAXITIME_SEC_mvt"]])
    return pd.concat(parts, ignore_index=True)


def _load_frame_v58(encoders: dict) -> pd.DataFrame:
    """v47 feature frame with openc_* columns recomputed from v58 encoders."""
    dep = pd.read_parquet(H1)
    dep["MVT_ID_mvt"] = pd.read_parquet(H1_IDS)["MVT_ID_mvt"].values
    # Drop the old openc_* columns before re-applying with new encoders
    for c in operator_numeric_cols():
        if c in dep.columns:
            dep = dep.drop(columns=[c])
    dep = apply_encoders(dep, encoders)
    # Re-attach tempo, quartile, plan v57 columns
    tempo = pd.read_parquet(V2_TRAIN, columns=["MVT_ID_mvt", *V40_TEMPO])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    quart = pd.read_parquet(TEMPO_P2575, columns=["MVT_ID_mvt", *V44_QUARTILES])
    dep = dep.merge(quart, on="MVT_ID_mvt", how="left")
    plan = pd.read_parquet(PLAN_TRAIN_V57, columns=["MVT_ID_mvt", *V45_PLAN])
    dep = dep.merge(plan, on="MVT_ID_mvt", how="left")
    return dep


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(TUNED) as f:
        tuned = json.load(f)["tuned"]
    with open(V46_REPORT) as f:
        v46_iters = json.load(f)["v46"]["best_iters"]

    log.info("=== step 1: refit encoders on all 12 months ===")
    t0 = time.time()
    raw = _load_raw_all_months()
    log.info("all-months DEP rows: %d in %.0fs", len(raw), time.time() - t0)
    encoders = fit_encoders(raw)
    with open(C.ROOT / "models" / "lgbm_r_all_v58.encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    log.info("wrote v58 encoders (%d keys + _global %.1f)",
             len([k for k in encoders if k != "_global"]),
             float(encoders["_global"]))

    log.info("=== step 2: build v58 frame with recomputed openc_* ===")
    t0 = time.time()
    dep = _load_frame_v58(encoders)
    log.info("v58 frame %d rows in %.0fs", len(dep), time.time() - t0)

    feat = v47_feature_names()
    adep = dep["ADEP_mvt"].astype(str)
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    keep = (adep != "LIRF") & (y > 0) & (y <= 80_000)
    dep = dep[keep].reset_index(drop=True)
    log.info("MB3 filter: %d rows", len(dep))

    # apply_encoders merges may drop categorical dtypes; restore.
    for c in CAT_COLS:
        if c in dep.columns and str(dep[c].dtype) == "object":
            dep[c] = dep[c].astype("category")

    n_v46_train = int(0.88 * 10 / 12 * 2_084_659)
    v58_scale = len(dep) / n_v46_train
    scaled = [int(math.ceil(i * v58_scale)) for i in v46_iters]
    log.info("v46 iters %s -> v58 %s (x %.3f)", v46_iters, scaled, v58_scale)

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
        b.save_model(str(C.ROOT / "models" / f"lgbm_r_all_v58_s{seed}.txt"))
        log.info("v58 seed %d %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    with open(C.ROOT / "models" / "lgbm_r_all_v58.features.txt", "w") as f:
        f.write("\n".join(feat))
    (C.ROOT / "models" / "lgbm_r_all_v58.meta.json").write_text(json.dumps({
        "tuned": tuned, "v46_iters": v46_iters, "scaled_iters": scaled,
        "scale_factor": v58_scale, "n_served": len(dep),
        "changed": "features_operator encoders refit on all 12 months (MB8)",
    }, indent=1))
    log.info("wrote v58 members and metadata")


if __name__ == "__main__":
    main()
