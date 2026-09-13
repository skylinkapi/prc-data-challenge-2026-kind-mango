"""Score any frame end to end. Base + regime mixture + tail head, plus the
post-processing (E1-E3).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src_v2 import config as C
from src_v2.heads import regime, tail
from src_v2.train import base as B

log = logging.getLogger(__name__)


def predict(dep: pd.DataFrame) -> np.ndarray:
    """Run base + regime mixture + tail head. Returns predictions in seconds.
    Caller must have applied the encoders already (via cli._add_plan_and_encoders)."""
    r_all = B.predict(dep)
    r_all = np.clip(r_all, 0, None)

    p_fb, fired = regime.predict(dep)
    sd = dep["sd"].astype("float64").values
    valid = np.isfinite(sd)
    mixture = np.where(valid, p_fb * sd + (1 - p_fb) * r_all, r_all)
    p_final = np.where(fired, mixture, r_all)

    p_final = tail.apply(dep, p_final)
    # E1
    p_final = np.clip(p_final, C.CLIP_LOW, C.CLIP_HIGH)
    return p_final


def label_free_report(pred: np.ndarray, dep: pd.DataFrame) -> dict:
    """P4. Per-airport mean of predictions, count over 7200, over 80000,
    at exactly 30 (the low bound). Unseen categorical levels are reported
    separately by check_unseen()."""
    adep = dep["ADEP_mvt"].astype(str).values
    out = {"per_airport": {}, "over_7200": 0, "over_80000": 0, "at_low_clip": 0}
    for a in np.unique(adep):
        m = adep == a
        out["per_airport"][a] = {
            "n": int(m.sum()),
            "mean_pred": float(pred[m].mean()),
            "p50": float(np.percentile(pred[m], 50)),
            "p99": float(np.percentile(pred[m], 99)),
        }
    out["over_7200"] = int((pred > 7200).sum())
    out["over_80000"] = int((pred > 80000).sum())
    out["at_low_clip"] = int((pred == C.CLIP_LOW).sum())
    out["at_zero"] = int((pred == 0).sum())
    return out
