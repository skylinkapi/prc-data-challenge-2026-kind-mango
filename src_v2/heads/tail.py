"""D1. Null-flight tail head with Dirichlet prior.

Rows with a null flight record and sd > 3600. Three classes: fallback,
24-h, normal. Fit a multinomial (three separate one-vs-rest booster-like
scoring is overkill for 100-ish rows). Use group-level smoothed rates
per (airport, sd-bin) with a Dirichlet(2, 2, 2) prior for stability, plus
a global fallback rate at the airport level.

Serves p_fb * sd + p_24 * (86400 + normal_median) + p_norm * normal, where
`normal` is the mean of positive predictions at that airport when we do
not have a null-flight model output for it.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src_v2 import config as C
from src_v2 import labels as L

log = logging.getLogger(__name__)

SD_HIGH = 14400.0                # Restrict to the Step A range (LIRF band table).
SD_BINS = np.array([14400, 30000, 50000, 60000, 70000, 80000, 90000, 180000, np.inf])
NORMAL_TAIL = 1150.0     # y - 86400 median from T6
TAIL_AIRPORTS = ("LIRF",)       # T6 shows 12/14 24h rows land at LIRF (D4 accepts the outliers).


def _bin_sd(sd: np.ndarray) -> np.ndarray:
    return np.digitize(sd, SD_BINS)


def train(dep: pd.DataFrame) -> dict:
    d = dep[(dep["flt_null"] == 1) & (dep["sd"] > SD_HIGH)
            & dep["ADEP_mvt"].isin(TAIL_AIRPORTS)
            & dep["month"].isin(C.FIT_MONTHS)].copy()
    if len(d) == 0:
        return {}
    y = d["TAXITIME_SEC_mvt"].astype("float64").values
    sd = d["sd"].astype("float64").values
    cls = L.classify(pd.Series(y), pd.Series(sd)).values

    d["bin"] = _bin_sd(sd)
    d["cls"] = cls

    tbl = {}
    for (a, b), g in d.groupby(["ADEP_mvt", "bin"]):
        counts = g["cls"].value_counts().to_dict()
        fb = counts.get("fallback", 0)
        h24 = counts.get("24h", 0)
        norm = counts.get("clean", 0) + counts.get("tail", 0)
        prior = C.TAIL_PRIOR
        tot = fb + h24 + norm + 3 * prior
        p_fb = (fb + prior) / tot
        p_24 = (h24 + prior) / tot
        p_no = (norm + prior) / tot
        tbl.setdefault(a, {})[int(b)] = {"p_fb": p_fb, "p_24": p_24, "p_norm": p_no,
                                          "rows": int(len(g))}
    # Global airport fallback (no bin match) using same prior + rate.
    globs = {}
    for a, g in d.groupby("ADEP_mvt"):
        counts = g["cls"].value_counts().to_dict()
        fb = counts.get("fallback", 0)
        h24 = counts.get("24h", 0)
        norm = counts.get("clean", 0) + counts.get("tail", 0)
        prior = C.TAIL_PRIOR
        tot = fb + h24 + norm + 3 * prior
        globs[a] = {"p_fb": (fb + prior)/tot,
                    "p_24": (h24 + prior)/tot,
                    "p_norm": (norm + prior)/tot}

    with open(C.MODELS / "tail_null.json", "w") as f:
        json.dump({"table": tbl, "global": globs,
                   "bins": SD_BINS.tolist(),
                   "normal_tail_extra": NORMAL_TAIL}, f)
    return {"airports": list(tbl.keys()), "rows_total": int(len(d))}


def apply(dep: pd.DataFrame, normal_pred: np.ndarray) -> np.ndarray:
    """Return the mixture prediction for null-flight rows with sd > 3600.
    Rows outside this set get `normal_pred` unchanged. D2 forbids a 24-h
    prediction on rows that have a flight record — this is enforced by
    only entering the mixture when flt_null == 1."""
    with open(C.MODELS / "tail_null.json") as f:
        obj = json.load(f)
    tbl = obj["table"]
    globs = obj["global"]
    bins = np.array(obj["bins"])
    extra = float(obj["normal_tail_extra"])

    out = normal_pred.copy().astype("float64")
    sd = dep["sd"].astype("float64").values
    adep = dep["ADEP_mvt"].astype(str).values
    fn = dep["flt_null"].fillna(0).astype("int8").values

    airport_ok = np.isin(adep, list(TAIL_AIRPORTS))
    idx = np.where((fn == 1) & (sd > SD_HIGH) & np.isfinite(sd) & airport_ok)[0]
    if len(idx) == 0:
        return out
    b = np.digitize(sd[idx], bins)
    for j, i in enumerate(idx):
        a = adep[i]
        p = tbl.get(a, {}).get(str(int(b[j])), None)
        if p is None:
            p = globs.get(a, {"p_fb": 0.5, "p_24": 0.0, "p_norm": 0.5})
        normal = normal_pred[i] if np.isfinite(normal_pred[i]) else extra
        out[i] = p["p_fb"] * sd[i] + p["p_24"] * (86400 + extra) + p["p_norm"] * normal
    return out
