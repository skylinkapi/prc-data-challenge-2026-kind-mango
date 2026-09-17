"""MP10: label classes in one module.

Adds a "low" class for 0 < y < 30 (P13) so class sums equal FULL MSE. The
fallback class uses the per-airport definition from MD1 when a table is
provided; otherwise it uses FB_TOL_DEFAULT.
"""
from __future__ import annotations

import numpy as np

from src_v3 import config as C

CLASSES = ("low", "clean", "fallback", "tail", "24h", "invalid")


def classify(y: np.ndarray, sd: np.ndarray, adep: np.ndarray,
             fb_tol_by_airport: dict[str, float] | None = None) -> np.ndarray:
    """Assign each row to exactly one class.

    A row is fallback when abs(y - sd) <= fb_tol at its ADEP (MD1). Fallback
    takes precedence over the y band, matching the deployed reports. Rows
    with y <= 0 are invalid (D9); rows with 0 < y < 30 are the low class
    (P13); 30 <= y <= 7200 clean; 7200 < y <= 80000 tail; y > 80000 24-h.
    """
    y = np.asarray(y, dtype=float)
    sd = np.asarray(sd, dtype=float)
    adep = np.asarray(adep)
    out = np.full(len(y), "invalid", dtype=object)

    if fb_tol_by_airport is None:
        fb_tol = np.full(len(y), C.FB_TOL_DEFAULT, dtype=float)
    else:
        fb_tol = np.array([fb_tol_by_airport.get(str(a), C.FB_TOL_DEFAULT)
                           for a in adep], dtype=float)

    valid_sd = ~np.isnan(sd)
    fb = valid_sd & (np.abs(y - sd) <= fb_tol) & (y > 0)
    low = (y > 0) & (y < C.Y_LOW_MAX) & ~fb
    clean = (y >= C.Y_CLEAN_MIN) & (y <= C.Y_CLEAN_MAX) & ~fb
    tail = (y > C.Y_CLEAN_MAX) & (y <= C.Y_TAIL_MAX) & ~fb
    h24 = (y > C.Y_24H_MIN) & ~fb

    out[low] = "low"
    out[clean] = "clean"
    out[fb] = "fallback"
    out[tail] = "tail"
    out[h24] = "24h"
    return out


def assert_class_sum_equals_full(mse_full: float, mse_by_class: dict[str, float],
                                 n: int, tol: float = 1e-6) -> None:
    """MP10 test: the class MSE contributions sum to the FULL MSE.

    Contributions come from `class_mse(..., n_scale=n)` and are already
    normalised by n, so their sum equals `mse_full` directly.
    """
    s = sum(mse_by_class.values())
    if abs(s - mse_full) > tol * max(1.0, mse_full):
        raise AssertionError(
            f"class sum {s:.6g} does not match FULL {mse_full:.6g}"
        )


def class_mse(pred: np.ndarray, y: np.ndarray, cls: np.ndarray,
              n_scale: int | None = None) -> dict[str, float]:
    """Squared error contribution per class, on the given scale.

    n_scale defaults to len(y). Pass 344,336 or 344,841 to compare against
    reports on those scales.
    """
    e2 = (pred - y) ** 2
    n = n_scale or len(y)
    return {c: float(e2[cls == c].sum() / n) for c in CLASSES}
