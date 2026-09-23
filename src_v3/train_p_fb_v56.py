"""MB8 applied to the p_fb LIRF gate: retrain on all 12 months.

The shipped v23 gate trained on months 2..6+8..10, early-stopped on
11-12 and calibrated on the same stop set. The isotonic map is fit on
those 2 stop months. MB8 refits the same recipe on ALL 12 months with a
fixed round count scaled from the v23 best iter.

Rate features (fbrate/fbcount) are recomputed on the full-year map via
`add_fallback_rate_features` (returns the scoring maps). The isotonic map
is fit on the in-sample scores of the full-year booster; v64
(`train_p_fb_v64.py`) replaces it with an out-of-fold map.

Writes lgbm_p_fb_lirf_v56.txt + lirf_regime_v56.isotonic.pkl and reuses
v23's gate feature list (v56 shares the same 167 features).
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS
from train_lirf_regime_v23 import FB_TOL, add_fallback_rate_features

from src_v3 import config as C

LIRF_CACHE = C.ROOT / "models" / "lirf_frame_cache.parquet"
V23_FEATURES = C.ROOT / "models" / "lirf_regime_v23.features.txt"
V23_BOOSTER = C.ROOT / "models" / "lgbm_p_fb_lirf_v23.txt"
SCALE = 1.0 / (1.0 - 0.12)   # 1.136
GATE_PARAMS = {"objective": "binary", "metric": "binary_logloss",
               "learning_rate": 0.05, "num_leaves": 63,
               "min_data_in_leaf": 50, "feature_fraction": 0.8,
               "bagging_fraction": 0.9, "bagging_freq": 5,
               "verbosity": -1,
               "num_threads": C.NUM_THREADS,
               "deterministic": C.DETERMINISTIC,
               "seed": C.GLOBAL_SEED,
               "bagging_seed": C.GLOBAL_SEED,
               "feature_fraction_seed": C.GLOBAL_SEED}
log = logging.getLogger(__name__)


def load_gate_frame() -> tuple[pd.DataFrame, list[str]]:
    """LIRF frame with the `is_fb` label and the rate columns, plus the gate features."""
    dep = pd.read_parquet(LIRF_CACHE)
    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")
    y = dep["TAXITIME_SEC_mvt"].astype(float).values
    sd = dep["sched_delay"].astype(float).values
    dep["is_fb"] = ((np.abs(y - sd) < FB_TOL) & (y > 0)).astype(np.int8)
    base_rate = float(dep["is_fb"].mean())
    log.info("LIRF fallback base rate: %.4f", base_rate)
    add_fallback_rate_features(dep, base_rate)  # in-place, OOF within the frame
    feat = [x for x in V23_FEATURES.read_text().splitlines() if x]
    log.info("gate features: %d", len(feat))
    return dep, feat


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    v23 = lgb.Booster(model_file=str(V23_BOOSTER))
    v23_iter = v23.num_trees()
    scaled = int(math.ceil(v23_iter * SCALE))
    log.info("v23 gate iters %d -> v56 scaled %d (x %.3f)", v23_iter, scaled, SCALE)

    log.info("loading LIRF frame cache")
    dep, feat = load_gate_frame()

    # Train on ALL 12 months, fixed rounds
    dt = lgb.Dataset(dep[feat], label=dep["is_fb"].values,
                     categorical_feature=CAT_COLS,
                     params={"feature_pre_filter": False})
    t0 = time.time()
    b = lgb.train(GATE_PARAMS, dt, num_boost_round=scaled)
    log.info("v56 gate trained %d rounds in %.0fs", scaled, time.time() - t0)
    b.save_model(str(C.ROOT / "models" / "lgbm_p_fb_lirf_v56.txt"))

    # Fit isotonic on the whole set with the model's own predictions.
    # The rate features used are the OOF ones already in `dep`, so the
    # predictions carry no leak on the target from those columns; the
    # isotonic map still sees the full year.
    p_raw = b.predict(dep[feat])
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_raw, dep["is_fb"].values)
    auc = float(roc_auc_score(dep["is_fb"], p_raw))
    ll = float(log_loss(dep["is_fb"], np.clip(p_raw, 1e-6, 1 - 1e-6)))
    log.info("full-year AUC %.4f  log-loss %.4f", auc, ll)
    with open(C.ROOT / "models" / "lirf_regime_v56.isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    (C.ROOT / "models" / "lirf_regime_v56.meta.json").write_text(json.dumps({
        "v23_iter": v23_iter, "scaled_iter": scaled, "scale_factor": SCALE,
        "n_rows": int(len(dep)), "auc": auc, "log_loss": ll,
        "shares_features_with_v23": True,
    }, indent=1))
    log.info("wrote v56 gate + isotonic")


if __name__ == "__main__":
    main()
