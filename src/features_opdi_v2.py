"""OPDI v2. Step 5 of MODEL_ANALYSIS.md.

Keeps only `entry-runway` counts (the one event that tracks real traffic at
every airport). Expresses each rolling count as a ratio to the trailing 28-day
median for the same airport and hour-of-week. That kills the OSM-tagging drift
between 2025 and 2026.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPDI_EVENTS = os.path.join(ROOT, "external", "opdi", "events_all.parquet")

WINDOWS_MIN = [15, 30, 60]

OPDI_V2_NUM_COLS = [f"opdi_runway_entries_ratio_prev_{w}m" for w in WINDOWS_MIN]


def _window_counts(query_ts, ref_ts, window_min):
    if len(ref_ts) == 0 or len(query_ts) == 0:
        return np.zeros(len(query_ts), dtype=np.int32)
    dt = np.timedelta64(window_min * 60, "s")
    lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
    hi = np.searchsorted(ref_ts, query_ts, side="right")
    return (hi - lo).astype(np.int32)


def _rolling_median_apt_how(counts_series: pd.Series, dates: np.ndarray,
                            hows: np.ndarray) -> np.ndarray:
    """For each row, median of counts over the trailing 28 days at same hour-of-week."""
    n = len(counts_series)
    out = np.full(n, np.nan, dtype=np.float32)
    df = pd.DataFrame({"date": dates, "how": hows, "c": counts_series.values})
    df["day"] = pd.to_datetime(df["date"]).dt.floor("D")
    # For each (how) group, compute rolling 28d median on daily median counts
    for how, g in df.groupby("how", sort=False):
        g_sorted = g.sort_values("day")
        daily_med = g_sorted.groupby("day")["c"].median()
        roll = daily_med.rolling("28D", closed="left").median()
        # Map back to original rows via day
        mapping = roll.to_dict()
        for idx, day in zip(g_sorted.index, g_sorted["day"].values):
            out[idx] = mapping.get(pd.Timestamp(day), np.nan)
    return out


def add_opdi_v2(dep: pd.DataFrame) -> pd.DataFrame:
    """dep needs: ADEP_mvt, mvt_ts. Adds OPDI entry-runway ratio features."""
    if not os.path.exists(OPDI_EVENTS):
        out = dep.copy()
        for c in OPDI_V2_NUM_COLS:
            out[c] = np.nan
        return out

    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    apt_arr = dep["ADEP_mvt"].values
    ts_arr = dep["mvt_ts"].values.astype("datetime64[us]")

    # Raw counts first
    raw = {w: np.full(n, np.nan, dtype=np.float32) for w in WINDOWS_MIN}

    import pyarrow.parquet as pq
    for icao in pd.unique(apt_arr):
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0:
            continue
        q_ts = ts_arr[pos]
        try:
            apt_ev = pq.read_table(
                OPDI_EVENTS,
                columns=["type", "event_time", "osm_airport"],
                filters=[("osm_airport", "=", str(icao)),
                         ("type", "=", "entry-runway")],
            ).to_pandas()
        except Exception:
            continue
        if len(apt_ev) == 0:
            continue
        e_ts = np.sort(apt_ev["event_time"].values.astype("datetime64[us]"))
        for w in WINDOWS_MIN:
            raw[w][pos] = _window_counts(q_ts, e_ts, w).astype(np.float32)

    # Rolling 28-day median of daily counts, keyed on (apt, hour-of-week)
    dep_dates = pd.to_datetime(dep["mvt_ts"]).dt.tz_convert(None) if pd.to_datetime(dep["mvt_ts"]).dt.tz is not None else pd.to_datetime(dep["mvt_ts"])
    dow = dep_dates.dt.dayofweek.values
    hour = dep_dates.dt.hour.values
    how = (dow * 24 + hour).astype(np.int32)
    days = dep_dates.dt.floor("D").values

    df_key = pd.DataFrame({"apt": apt_arr, "how": how, "day": days})
    for w in WINDOWS_MIN:
        col = f"opdi_runway_entries_ratio_prev_{w}m"
        tmp = df_key.copy()
        tmp["c"] = raw[w]
        daily = tmp.dropna(subset=["c"]).groupby(["apt", "how", "day"])["c"].median().reset_index()
        daily = daily.sort_values(["apt", "how", "day"])
        # Rolling 28d median per (apt, how) using transform
        roll_vals = []
        for (_, _), g in daily.groupby(["apt", "how"], sort=False):
            s = g.set_index("day")["c"].rolling("28D", closed="left").median()
            roll_vals.append(s.reset_index(drop=True))
        daily["roll_med"] = pd.concat(roll_vals, ignore_index=True).values
        ref = daily[["apt", "how", "day", "roll_med"]]
        merged = tmp.merge(ref, on=["apt", "how", "day"], how="left")
        denom = merged["roll_med"].values
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where((denom > 0) & np.isfinite(denom), raw[w] / denom, np.nan)
        dep[col] = ratio.astype(np.float32)

    return dep.set_index("_orig_idx").rename_axis(None)


if __name__ == "__main__":
    import glob
    from features_weather import TARGET_ICAOS
    p = os.path.join(ROOT, "training", "training_2025-06-01_2025-07-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].head(20000).copy()
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    out = add_opdi_v2(df)
    print(out[OPDI_V2_NUM_COLS].describe().round(3).to_string())
    print("\nRatio-1 distribution (should center on 1.0):")
    for c in OPDI_V2_NUM_COLS:
        v = out[c].dropna()
        print(f"  {c:45s} n={len(v):>6} mean={v.mean():.2f} p50={v.median():.2f}")
