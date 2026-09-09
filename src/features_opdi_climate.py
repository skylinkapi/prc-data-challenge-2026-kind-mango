"""Historical OPDI taxi climatology as features.

For each DEP row, joins in the median actual-taxi from 2022-2024 at the same
(airport, month, day-of-week, hour-bin) bucket. Falls back to coarser bucket
if the fine one is empty.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIM = os.path.join(ROOT, "external", "opdi", "climatology.parquet")

CLIM_NUM_COLS = ["clim_median_taxi", "clim_mean_taxi", "clim_count",
                 "clim_median_taxi_2", "clim_count_2"]


def _hour_bin(h: pd.Series) -> pd.Series:
    return pd.cut(h, bins=[-1, 5, 9, 13, 17, 21, 24],
                  labels=["night", "earlyam", "midam", "midpm", "evening", "latenight"])


def add_opdi_climate(dep: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(CLIM):
        out = dep.copy()
        for c in CLIM_NUM_COLS:
            out[c] = np.nan
        return out
    clim = pd.read_parquet(CLIM)
    out = dep.copy()
    out["hour_bin"] = _hour_bin(out["hour"])
    out = out.merge(clim, on=["ADEP_mvt", "month", "dow", "hour_bin"], how="left")
    return out.drop(columns=["hour_bin"])


if __name__ == "__main__":
    import glob
    from features_weather import TARGET_ICAOS
    p = os.path.join(ROOT, "training", "training_2025-06-01_2025-07-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].head(50000).copy()
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    df["hour"] = df["mvt_ts"].dt.hour
    df["month"] = df["mvt_ts"].dt.month
    df["dow"] = df["mvt_ts"].dt.dayofweek
    out = add_opdi_climate(df)
    print(f"Rows: {len(out):,}   climatology coverage:")
    for c in CLIM_NUM_COLS:
        cov = out[c].notna().mean() * 100
        print(f"  {c:25s} cov {cov:5.1f}%   sample mean {out[c].dropna().mean():.1f}")
    print()
    print("Per-airport coverage:")
    for a in sorted(out["ADEP_mvt"].unique()):
        sub = out[out["ADEP_mvt"] == a]
        cov = sub["clim_median_taxi"].notna().mean() * 100
        print(f"  {a}: {cov:5.1f}%")
