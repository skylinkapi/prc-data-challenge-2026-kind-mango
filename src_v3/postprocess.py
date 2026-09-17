"""Post-processing measures MS1 to MS4.

MS1. Per-airport upper bounds after the ensemble mean.
    - EDDF, EDDM, LEBL, LEMD: no 2025 row above 7,200 s outside fallback,
      so cap at the 2025 airport max of the tail class plus a margin.
      When the tail class is empty at that airport, cap at
      `clean_max + margin`.
    - EGLL, EHAM, LFPG, LSZH, LTFM: values above 7,200 s allowed only when
      the row has a flight record (`flt_null == 0`) AND `mvt_eobt1 > 5400`.
      Otherwise cap at the 2025 clean maximum at that airport.
    - LIRF: no bound applied here; MH4 handles LIRF.
MS2. Clip the ensemble mean, floor at the 2025 clean 0.1 % quantile per
     airport (from `support.q_low_clean`). No per-member clip.
MS3. Where three members disagree by more than 3,600 s outside LIRF, serve
     the member median.
MS4. Assert the row count and id set match the template; fail on missing
     predictions rather than fill.

Each function returns both the transformed prediction vector and a
per-airport bookkeeping dict that names which rows moved and by how much.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# MS1 airport classes
NO_LONG_TAIL = ("EDDF", "EDDM", "LEBL", "LEMD")   # cap at max class + margin
HAS_LONG_TAIL = ("EGLL", "EHAM", "LFPG", "LSZH", "LTFM")

MS1_TAIL_MARGIN_S = 600         # margin over 2025 airport tail max
MS3_DISAGREE_S = 3600           # member spread threshold
MS1_MVT_EOBT1_MIN = 5400        # long-tail row must have mvt_eobt1 above this


def _airport_cap(a: str, support_a: dict) -> float:
    """MS1 upper bound at airport `a` for a NO_LONG_TAIL airport."""
    tail_max = support_a["class_max"].get("tail", 0.0)
    clean_max = support_a["class_max"].get("clean", 7200.0)
    ref = tail_max if tail_max > 0 else clean_max
    return float(ref + MS1_TAIL_MARGIN_S)


def _clean_cap(a: str, support_a: dict) -> float:
    return float(support_a["class_max"].get("clean", 7200.0)) + MS1_TAIL_MARGIN_S


def apply_ms1(pred: np.ndarray, adep: np.ndarray,
              flt_null: np.ndarray, mvt_eobt1: np.ndarray,
              support: dict) -> tuple[np.ndarray, dict]:
    """Bound predictions per airport after the ensemble.

    LIRF rows are untouched (MH4 handles them).
    """
    out = pred.astype(float).copy()
    moved: dict[str, dict] = {}
    for a in np.unique(adep):
        if a == "LIRF" or a not in support:
            continue
        m = (adep == a)
        s = support[a]
        n_before = int(((out[m] > 7200) | (out[m] < 0)).sum())
        if a in NO_LONG_TAIL:
            cap = _airport_cap(a, s)
            new = np.minimum(out[m], cap)
        elif a in HAS_LONG_TAIL:
            cap = _clean_cap(a, s)
            eligible = (flt_null[m] == 0) & (mvt_eobt1[m] > MS1_MVT_EOBT1_MIN)
            row_cap = np.where(eligible, np.inf, cap)
            new = np.minimum(out[m], row_cap)
        else:
            continue
        n_moved = int((new != out[m]).sum())
        max_move = float(np.max(np.abs(new - out[m]))) if n_moved else 0.0
        moved[a] = {"rows_moved": n_moved, "max_move_s": max_move,
                    "before_over_7200": n_before, "cap_used": float(cap)}
        out[m] = new
    return out, moved


def apply_ms2(pred: np.ndarray, adep: np.ndarray,
              support: dict) -> tuple[np.ndarray, dict]:
    """Floor the ensemble mean at the 2025 clean 0.1 % quantile per airport."""
    out = pred.astype(float).copy()
    moved: dict[str, dict] = {}
    for a in np.unique(adep):
        if a not in support:
            continue
        m = (adep == a)
        floor = float(support[a]["q_low_clean"])
        n = int((out[m] < floor).sum())
        out[m] = np.maximum(out[m], floor)
        moved[a] = {"rows_moved": n, "floor_s": floor}
    return out, moved


def apply_ms3(members: np.ndarray, adep: np.ndarray) -> tuple[np.ndarray, dict]:
    """When 3 members disagree by more than MS3_DISAGREE_S outside LIRF,
    serve the member median instead of the mean.

    `members` is shape (n_members, n_rows).
    """
    mean = members.mean(axis=0)
    med = np.median(members, axis=0)
    spread = members.max(axis=0) - members.min(axis=0)
    disagree = (spread > MS3_DISAGREE_S) & (adep != "LIRF")
    out = np.where(disagree, med, mean)
    moved: dict[str, dict] = {}
    for a in np.unique(adep):
        m = (adep == a) & disagree
        if m.any():
            moved[a] = {"rows_moved": int(m.sum()),
                        "max_spread_s": float(spread[m].max())}
    return out, moved


def assert_ms4(pred: np.ndarray, template_ids: np.ndarray,
               pred_ids: np.ndarray) -> None:
    """MS4: id set and row count equal the template. Fail if not."""
    if len(pred) != len(template_ids):
        raise AssertionError(f"row count {len(pred)} != template {len(template_ids)}")
    if set(pred_ids) != set(template_ids):
        raise AssertionError("id set differs from template")
    if np.isnan(pred).any():
        raise AssertionError(f"{int(np.isnan(pred).sum())} predictions are NaN")


def apply_pipeline(members: np.ndarray, adep: np.ndarray,
                   flt_null: np.ndarray, mvt_eobt1: np.ndarray,
                   support: dict) -> tuple[np.ndarray, dict]:
    """Serve MS2 floor + MS3 disagreement + MS1 cap, in that order.

    MS3 chooses mean vs median per row, then MS1 caps the served value from
    above, then MS2 floors it from below. LIRF rows go through untouched
    (the LIRF head will overwrite them anyway).
    """
    served, ms3_moved = apply_ms3(members, adep)
    served, ms1_moved = apply_ms1(served, adep, flt_null, mvt_eobt1, support)
    served, ms2_moved = apply_ms2(served, adep, support)
    return served, {"MS1": ms1_moved, "MS2": ms2_moved, "MS3": ms3_moved}
