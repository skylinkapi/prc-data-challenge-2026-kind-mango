"""Per-fold scoring, per-class per-airport reports and MP2 interval decisions.

The pass metric is the full MSE over the rows a component serves. The class
table per airport is a diagnostic (MP2). Every report reconciles: the class
sums equal the FULL MSE (MP10, `assert_class_sum_equals_full`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.folds import paired_fold_delta
from src_v3.labels import CLASSES, assert_class_sum_equals_full, class_mse, classify


@dataclass
class FoldReport:
    fold: tuple[int, int]
    full_mse: float
    full_rmse: float
    class_mse: dict[str, float]
    per_airport: dict[str, dict[str, float]]
    n: int

    def to_dict(self) -> dict:
        return {"fold": list(self.fold), "n": self.n,
                "full_mse": self.full_mse, "full_rmse": self.full_rmse,
                "class_mse": self.class_mse, "per_airport": self.per_airport}


def score_fold(pred: np.ndarray, y: np.ndarray, sd: np.ndarray,
               adep: np.ndarray, fold: tuple[int, int],
               fb_tol_by_airport: dict[str, float] | None = None
               ) -> FoldReport:
    """FULL MSE, class table and per-airport class table on one fold."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    e2 = (pred - y) ** 2
    n = int(len(y))
    full_mse = float(e2.mean())
    full_rmse = float(np.sqrt(full_mse))
    cls = classify(y, sd, adep, fb_tol_by_airport)
    cmse = class_mse(pred, y, cls, n_scale=n)
    per_apt: dict[str, dict[str, float]] = {}
    for a in sorted(set(np.asarray(adep))):
        m = (adep == a)
        cm = class_mse(pred[m], y[m], cls[m], n_scale=n)
        per_apt[str(a)] = cm
    assert_class_sum_equals_full(full_mse, cmse, n)
    return FoldReport(fold=fold, full_mse=full_mse, full_rmse=full_rmse,
                      class_mse=cmse, per_airport=per_apt, n=n)


def paired_prices(control: list[FoldReport], lever: list[FoldReport],
                  served_class: str = "clean",
                  served_airports: tuple[str, ...] | None = None,
                  alpha: float = C.FOLD_INTERVAL_ALPHA) -> dict:
    """Per-fold delta on the rows the lever's component serves.

    `served_class` is one of CLASSES; `served_airports` restricts to a subset
    (base outside LIRF passes ("EDDF", "EDDM", ..., "LTFM") without "LIRF").
    The delta is (lever - control) MSE contribution of the served subset.
    """
    if len(control) != len(lever):
        raise ValueError("fold count mismatch")
    apts = set(served_airports) if served_airports else None
    per_fold_full_delta: list[float] = []
    per_fold_served_delta: list[float] = []
    for c, l in zip(control, lever):
        per_fold_full_delta.append(l.full_mse - c.full_mse)
        c_srv = sum(v[served_class] for a, v in c.per_airport.items()
                    if apts is None or a in apts)
        l_srv = sum(v[served_class] for a, v in l.per_airport.items()
                    if apts is None or a in apts)
        per_fold_served_delta.append(l_srv - c_srv)
    return {
        "served_class": served_class,
        "served_airports": sorted(apts) if apts else None,
        "full_delta": paired_fold_delta(np.array(per_fold_full_delta), alpha),
        "served_delta": paired_fold_delta(np.array(per_fold_served_delta), alpha),
        "per_fold_reports": {
            "control": [r.to_dict() for r in control],
            "lever": [r.to_dict() for r in lever],
        },
    }


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
