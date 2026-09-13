"""P4 label-free 2026 checks and a subset of the T1..T14 diagnostics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src_v2 import config as C
from src_v2 import labels as L


def class_shares(dep: pd.DataFrame) -> pd.DataFrame:
    """T2/T11 style. Fallback uses the 5-second window (A1)."""
    cls = L.classify(dep["TAXITIME_SEC_mvt"], dep["sd"])
    return (pd.crosstab(dep["ADEP_mvt"], cls)
              .div(pd.crosstab(dep["ADEP_mvt"], cls).sum(axis=1), axis=0))


def coverage_monitor(train: pd.DataFrame, rank: pd.DataFrame,
                     jan_mask, jul_mask, cols) -> pd.DataFrame:
    """A3. Per-column non-null share for 2025 (all), 2026 Jan and 2026 Jul.
    Returns a frame with the deltas."""
    def share(df, mask):
        d = df if mask is None else df[mask]
        return d[cols].notna().mean()

    tab = pd.DataFrame({
        "cov_2025": share(train, None),
        "cov_2026_jan": share(rank, jan_mask),
        "cov_2026_jul": share(rank, jul_mask),
    })
    tab["drop_jan_pts"] = 100 * (tab["cov_2025"] - tab["cov_2026_jan"])
    tab["drop_jul_pts"] = 100 * (tab["cov_2025"] - tab["cov_2026_jul"])
    return tab


def label_free(pred: np.ndarray, dep: pd.DataFrame) -> dict:
    adep = dep["ADEP_mvt"].astype(str).values
    out = {"per_airport": {}, "over_7200": 0, "over_80000": 0}
    for a in np.unique(adep):
        m = adep == a
        out["per_airport"][a] = {
            "n": int(m.sum()),
            "mean_pred": float(pred[m].mean()),
            "p50": float(np.percentile(pred[m], 50)),
            "p99": float(np.percentile(pred[m], 99)),
            "max": float(pred[m].max()),
        }
    out["over_7200"] = int((pred > 7200).sum())
    out["over_80000"] = int((pred > 80000).sum())
    out["at_low_clip"] = int((pred == C.CLIP_LOW).sum())
    return out
