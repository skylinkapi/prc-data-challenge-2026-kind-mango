"""MC5: id-keyed merge assertions.

Every merge that joins served columns or predictions to the template must
preserve row count and id uniqueness. Two pipelines that compute the same
quantity (for example `mvt_eobt1`) must agree on every row. This module
holds one helper per rule so the intent is legible at the call site.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assert_one_to_one(left: pd.DataFrame, right: pd.DataFrame, key: str) -> None:
    """Both sides have unique, aligned key sets."""
    ldup = left[key].duplicated().sum()
    rdup = right[key].duplicated().sum()
    if ldup or rdup:
        raise AssertionError(
            f"key {key!r} has duplicates: left {ldup}, right {rdup}"
        )
    only_l = set(left[key]) - set(right[key])
    only_r = set(right[key]) - set(left[key])
    if only_l or only_r:
        raise AssertionError(
            f"key {key!r} does not match: left-only {len(only_l)}, "
            f"right-only {len(only_r)}"
        )


def assert_row_count(df: pd.DataFrame, expected: int, label: str = "df") -> None:
    if len(df) != expected:
        raise AssertionError(f"{label}: {len(df)} rows, expected {expected}")


def assert_agree(a: pd.Series, b: pd.Series, name: str,
                 tol: float = 1e-6) -> None:
    """Two computations of the same quantity agree on every row.

    NaN is allowed on both sides for the same row. Any position where one
    side is NaN and the other is not is a disagreement.
    """
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if len(a) != len(b):
        raise AssertionError(f"{name}: length {len(a)} vs {len(b)}")
    an = a.isna().values
    bn = b.isna().values
    both_nan = an & bn
    diff_nan = an ^ bn
    if diff_nan.any():
        raise AssertionError(
            f"{name}: NaN pattern differs on {int(diff_nan.sum())} rows"
        )
    va = a.values.astype(float)[~both_nan]
    vb = b.values.astype(float)[~both_nan]
    d = np.abs(va - vb)
    if d.size and d.max() > tol * max(1.0, float(np.max(np.abs(va)))):
        idx = np.argmax(d)
        raise AssertionError(
            f"{name}: {int((d > tol).sum())} disagreements; max |diff| = {d[idx]:.6g}"
        )
