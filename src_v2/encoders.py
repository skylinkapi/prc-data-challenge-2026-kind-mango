"""A7. Robust target encoders on clean rows, fit months only.

Median and interquartile range of y per key, with a count column. `MIN_COUNT`
is a config value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src_v2 import config as C
from src_v2.labels import classify

ENCODE_KEYS = ("AIRCRAFT_OPERATOR_flt", "flt_prefix", "AIRCRAFT_TYPE_mvt",
               "STAND_mvt", "stand_prefix", "RUNWAY_mvt", "ADES_mvt")


def _fb_rate_key(y: pd.Series, sd: pd.Series) -> pd.Series:
    return ((y - sd).abs() <= C.FB_TOL).astype("float64")


def fit_encoders(dep: pd.DataFrame) -> dict:
    """Fit median, IQR and count per encoding key on clean, fit-month rows.
    Also fit a fallback-rate encoder per key on all fit-month rows."""
    fit_mask = dep["month"].isin(C.FIT_MONTHS)
    cls = classify(dep["TAXITIME_SEC_mvt"], dep["sd"])
    clean = fit_mask & (cls == "clean")
    y = dep.loc[clean, "TAXITIME_SEC_mvt"].astype("float64")

    fb = _fb_rate_key(dep.loc[fit_mask, "TAXITIME_SEC_mvt"].astype("float64"),
                      dep.loc[fit_mask, "sd"].astype("float64"))
    base_rate = float(fb.mean())

    enc = {"base_rate": base_rate, "keys": {}}
    for k in ENCODE_KEYS:
        g_clean = y.groupby(dep.loc[clean, k])
        med = g_clean.median()
        q1 = g_clean.quantile(0.25)
        q3 = g_clean.quantile(0.75)
        cnt = g_clean.count()
        keep = cnt >= C.MIN_COUNT
        med = med[keep]
        iqr = (q3 - q1)[keep]
        cnt = cnt[keep]

        g_fb = fb.groupby(dep.loc[fit_mask, k])
        fb_rate = g_fb.mean()
        fb_cnt = g_fb.count()
        smooth = (fb_rate * fb_cnt + base_rate * C.MIN_COUNT) / (fb_cnt + C.MIN_COUNT)
        smooth = smooth[fb_cnt >= C.MIN_COUNT]

        enc["keys"][k] = {
            "med": med.astype("float32"),
            "iqr": iqr.astype("float32"),
            "cnt": cnt.astype("int32"),
            "fb_rate": smooth.astype("float32"),
        }
    return enc


def apply_encoders(dep: pd.DataFrame, enc: dict) -> pd.DataFrame:
    """Add enc_<k>_med, enc_<k>_iqr, enc_<k>_cnt and enc_<k>_fbrate columns."""
    out = dep.copy()
    for k, m in enc["keys"].items():
        key_series = dep[k]
        out[f"enc_{k}_med"] = key_series.map(m["med"]).astype("float32")
        out[f"enc_{k}_iqr"] = key_series.map(m["iqr"]).astype("float32")
        out[f"enc_{k}_cnt"] = key_series.map(m["cnt"]).astype("Int32")
        out[f"enc_{k}_fbrate"] = key_series.map(m["fb_rate"]).astype("float32")
    return out
