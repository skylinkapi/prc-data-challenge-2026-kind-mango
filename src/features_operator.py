"""Operator-level historic taxi-time encodings.

For each key combination computes median (robust to tail) and std from the
TRAINING SLICE ONLY (non-hold-out months), then applies the lookup to any
frame. Using the training slice as the encoding source removes hold-out
leakage — the model at inference just sees historic-typical taxi times per
(operator, airport, ...) bucket.

On the ranking file: fitted encoders from full 2025 apply directly.
"""
import numpy as np
import pandas as pd


KEYS = [
    ("op_apt",       ["AIRCRAFT_OPERATOR_flt", "ADEP_mvt"]),
    ("op_apt_rwy",   ["AIRCRAFT_OPERATOR_flt", "ADEP_mvt", "RUNWAY_mvt"]),
    ("op_apt_hbin", ["AIRCRAFT_OPERATOR_flt", "ADEP_mvt", "_hour_bin"]),
    ("op",           ["AIRCRAFT_OPERATOR_flt"]),
    ("op_apt_stand", ["AIRCRAFT_OPERATOR_flt", "ADEP_mvt", "STAND_mvt"]),
]

MIN_COUNT = 20   # keys with fewer rows fall through to a coarser key


def _hour_bin(h: pd.Series) -> pd.Series:
    return pd.cut(h, bins=[-1, 5, 9, 13, 17, 21, 24],
                  labels=["night","earlyam","midam","midpm","evening","latenight"])


def fit_encoders(train: pd.DataFrame) -> dict:
    """Compute median + count per key on the training slice."""
    df = train.copy()
    df["_hour_bin"] = _hour_bin(df["hour"])
    y = "TAXITIME_SEC_mvt"
    encoders = {}
    for name, keys in KEYS:
        agg = df.groupby(keys, observed=True)[y].agg(["median", "count", "std"]).reset_index()
        agg = agg.rename(columns={"median": f"openc_{name}_median",
                                  "count":  f"openc_{name}_count",
                                  "std":    f"openc_{name}_std"})
        agg = agg[agg[f"openc_{name}_count"] >= MIN_COUNT]
        encoders[name] = (keys, agg)
    # global fallback
    encoders["_global"] = train[y].median()
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    out = df.copy()
    out["_hour_bin"] = _hour_bin(out["hour"])
    for name, val in encoders.items():
        if name == "_global":
            continue
        keys, tbl = val
        out = out.merge(tbl, on=keys, how="left")
    out = out.drop(columns=["_hour_bin"])
    # coalesce medians finest -> coarsest
    coarsen_order = ["op_apt_stand", "op_apt_rwy", "op_apt_hbin", "op_apt", "op"]
    out["openc_taxi_median"] = np.nan
    for name in coarsen_order:
        col = f"openc_{name}_median"
        if col in out.columns:
            out["openc_taxi_median"] = out["openc_taxi_median"].fillna(out[col])
    out["openc_taxi_median"] = out["openc_taxi_median"].fillna(encoders["_global"])
    return out


def operator_numeric_cols() -> list:
    cols = ["openc_taxi_median"]
    for name, _ in KEYS:
        cols.append(f"openc_{name}_median")
        cols.append(f"openc_{name}_count")
        cols.append(f"openc_{name}_std")
    return cols
