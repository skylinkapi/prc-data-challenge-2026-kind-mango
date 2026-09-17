"""Per-airport 2025 support tables for MS1 bounds and MS2 floor.

The tables are computed once from the training frame using the MP10 label
classes, and are frozen at fit time. The class definition matches MD1 when a
fallback tolerance is provided per airport; otherwise it uses FB_TOL_DEFAULT.

The tables the bounds rule reads:

- `class_max`: the maximum `y` in the given class at each airport
- `q_low`: the 0.1 % quantile of clean-class `y` per airport (floor for MS2)
- `n_over_7200`: count of rows with y > 7,200 per airport, informational
- `n_by_class`: count per class per airport, informational
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.labels import classify


def build_support(train_frame: pd.DataFrame,
                  fb_tol_by_airport: dict[str, float] | None = None) -> dict:
    """Compute the frozen per-airport support tables from 2025 training rows.

    `train_frame` needs `ADEP_mvt`, `sched_delay`, `TAXITIME_SEC_mvt`.
    """
    y = train_frame["TAXITIME_SEC_mvt"].astype(float).values
    sd = train_frame["sched_delay"].astype(float).values
    adep = train_frame["ADEP_mvt"].astype(str).values
    cls = classify(y, sd, adep, fb_tol_by_airport)
    tables: dict[str, dict] = {}
    for a in sorted(set(adep)):
        m = (adep == a)
        y_a = y[m]
        cls_a = cls[m]
        clean_y = y_a[cls_a == "clean"]
        tail_y = y_a[cls_a == "tail"]
        h24_y = y_a[cls_a == "24h"]
        tables[a] = {
            "n": int(m.sum()),
            "n_by_class": {c: int((cls_a == c).sum())
                           for c in ("low", "clean", "fallback", "tail", "24h")},
            "class_max": {
                "clean": float(clean_y.max()) if clean_y.size else 0.0,
                "tail": float(tail_y.max()) if tail_y.size else 0.0,
                "24h": float(h24_y.max()) if h24_y.size else 0.0,
            },
            "q_low_clean": float(np.quantile(clean_y, 0.001)) if clean_y.size else 30.0,
            "n_over_7200": int((y_a > 7200).sum()),
            "n_over_3600": int((y_a > 3600).sum()),
        }
    return tables


def write_support(path: Path, tables: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tables, indent=1))
