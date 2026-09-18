"""MB4 + MB3: base retrained on served rows with cyclic calendar features.

Extends the v51 MB3 filter (LIRF and y > 80,000 excluded) with the MB4
calendar features (`doy_sin`, `doy_cos`, `hour_local`, `is_public_hol`)
computed by `src_v3.calendar_extra`. Drops the numeric `month` column
from the feature list per §5.3 MB4 (P2 defect).

Same recipe as v51 otherwise: retuned Optuna params, linear_tree,
deterministic, 3 seeds, scaled iteration counts.
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
from src_v3.calendar_extra import add_calendar_extra
from src_v3.frames import load_v47_frame, v47_feature_names

TUNED = C.ROOT / "models" / "tune_lgbm_v43.tuned_params.json"
V46_REPORT = C.ROOT / "models" / "lgbm_r_all_v46.holdout.json"
V2_TRAIN = C.ROOT / "models" / "v2" / "cache" / "frame_train.parquet"
EXTRA_FEATURES = ["doy_sin", "doy_cos", "hour_local", "is_public_hol"]
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(TUNED) as f:
        tuned = json.load(f)["tuned"]
    with open(V46_REPORT) as f:
        v46_iters = json.load(f)["v46"]["best_iters"]

    log.info("loading v47 feature frame + mvt_ts")
    dep = load_v47_frame()
    ts = pd.read_parquet(V2_TRAIN, columns=["MVT_ID_mvt", "mvt_ts"])
    dep = dep.merge(ts, on="MVT_ID_mvt", how="left")
    log.info("frame %d rows, mvt_ts coverage %.4f",
             len(dep), dep["mvt_ts"].notna().mean())
    dep = add_calendar_extra(dep)
    for c in EXTRA_FEATURES:
        log.info("  %s: non-null %.4f, mean %s", c,
                 dep[c].notna().mean(),
                 f"{dep[c].astype(float).mean():.3f}")

    # MB3 filter
    adep = dep["ADEP_mvt"].astype(str)
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    keep = (adep != "LIRF") & (y > 0) & (y <= 80_000)
    dep = dep[keep].reset_index(drop=True)
    log.info("filtered to %d served rows", len(dep))

    # Feature list: v47 features minus 'month', plus MB4 extras
    v47_feat = v47_feature_names()
    feat = [c for c in v47_feat if c != "month"] + EXTRA_FEATURES
    log.info("feature count: v47 %d -> v52 %d (dropped month, added %s)",
             len(v47_feat), len(feat), EXTRA_FEATURES)

    # Iteration scaling like v51
    n_served = len(dep)
    n_v46_train = int(0.88 * 10 / 12 * 2_084_659)
    v52_scale = n_served / n_v46_train
    scaled = [int(math.ceil(i * v52_scale)) for i in v46_iters]
    log.info("v46 iters %s -> v52 %s (x %.3f)", v46_iters, scaled, v52_scale)

    cat_cols = [c for c in CAT_COLS if c in feat]
    dt = lgb.Dataset(dep[feat], label=dep["TAXITIME_SEC_mvt"].values,
                     categorical_feature=cat_cols,
                     params={"linear_tree": True, "feature_pre_filter": False})

    for seed, n_iter in zip(DEFAULT_SEEDS, scaled):
        p = {**tuned, "seed": seed, "bagging_seed": seed,
             "feature_fraction_seed": seed, "linear_tree": True,
             "bagging_freq": 5, "verbosity": -1,
             "num_threads": C.NUM_THREADS,
             "deterministic": C.DETERMINISTIC}
        t0 = time.time()
        b = lgb.train(p, dt, num_boost_round=n_iter)
        b.save_model(str(C.ROOT / "models" / f"lgbm_r_all_v52_s{seed}.txt"))
        log.info("v52 seed %d %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    with open(C.ROOT / "models" / "lgbm_r_all_v52.features.txt", "w") as f:
        f.write("\n".join(feat))
    shutil.copy(str(C.ROOT / "models" / "lgbm_r_all_v26.encoders.pkl"),
                str(C.ROOT / "models" / "lgbm_r_all_v52.encoders.pkl"))
    (C.ROOT / "models" / "lgbm_r_all_v52.meta.json").write_text(json.dumps({
        "tuned": tuned, "v46_iters": v46_iters, "scaled_iters": scaled,
        "scale_factor": v52_scale, "n_served": n_served,
        "features": feat, "extra_features": EXTRA_FEATURES,
        "filter": "adep != 'LIRF' and 0 < y <= 80_000, month dropped",
    }, indent=1))
    log.info("wrote v52 members and metadata")


if __name__ == "__main__":
    main()
