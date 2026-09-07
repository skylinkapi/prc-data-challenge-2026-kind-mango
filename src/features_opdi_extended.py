"""Extended OPDI rolling-count features (6 event types, 3 windows = 18 cols).

Adds airport-state signals beyond the original 3-type version in features_opdi.py:
  exit-runway         landings clearing the runway
  exit-taxiway        aircraft moving from taxi network toward next stage
  entry-threshold     aircraft lining up for takeoff (imminent departure)
  entry-parking_position   arrivals reaching gate (gate demand)

Same design: leak-free (uses OTHER flights' events in past window at same airport).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPDI_EVENTS = os.path.join(ROOT, "external", "opdi", "events_all.parquet")

WINDOWS_MIN = [15, 30, 60]

EXT_TYPE_MAP = {
    "exit-runway":              "exit_runway",
    "exit-taxiway":             "exit_taxiway",
    "entry-threshold":          "entry_threshold",
    "entry-parking_position":   "entry_parking",
}

OPDI_EXT_NUM_COLS = []
for w in WINDOWS_MIN:
    for suffix in EXT_TYPE_MAP.values():
        OPDI_EXT_NUM_COLS.append(f"opdi_{suffix}_prev_{w}m")


def _window_counts(query_ts: np.ndarray, ref_ts: np.ndarray, window_min: int) -> np.ndarray:
    if len(ref_ts) == 0 or len(query_ts) == 0:
        return np.zeros(len(query_ts), dtype=np.int32)
    dt = np.timedelta64(window_min * 60, "s")
    lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
    hi = np.searchsorted(ref_ts, query_ts, side="right")
    return (hi - lo).astype(np.int32)


def add_opdi_ext(dep: pd.DataFrame) -> pd.DataFrame:
    """dep needs: ADEP_mvt, mvt_ts."""
    if not os.path.exists(OPDI_EVENTS):
        out = dep.copy()
        for c in OPDI_EXT_NUM_COLS:
            out[c] = np.nan
        return out

    import pyarrow.parquet as pq
    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    new_cols = {c: np.full(n, np.nan, dtype=np.float32) for c in OPDI_EXT_NUM_COLS}

    apt_arr = dep["ADEP_mvt"].values
    ts_arr = dep["mvt_ts"].values.astype("datetime64[us]")

    for icao in pd.unique(apt_arr):
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0:
            continue
        q_ts = ts_arr[pos]
        try:
            apt_ev = pq.read_table(
                OPDI_EVENTS,
                columns=["type", "event_time", "osm_airport"],
                filters=[("osm_airport", "=", str(icao))],
            ).to_pandas()
        except Exception:
            continue

        for evt_type, suffix in EXT_TYPE_MAP.items():
            sub = apt_ev[apt_ev["type"] == evt_type]
            if len(sub) == 0:
                continue
            e_ts = sub["event_time"].values.astype("datetime64[us]")
            e_ts.sort()
            for w in WINDOWS_MIN:
                col = f"opdi_{suffix}_prev_{w}m"
                new_cols[col][pos] = _window_counts(q_ts, e_ts, w).astype(np.float32)
        del apt_ev

    for k, v in new_cols.items():
        dep[k] = v
    return dep.set_index("_orig_idx").rename_axis(None)


if __name__ == "__main__":
    import glob
    from features_weather import TARGET_ICAOS
    p = os.path.join(ROOT, "training", "training_2025-06-01_2025-07-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].head(20000).copy()
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    out = add_opdi_ext(df)
    print(f"Rows: {len(out):,}   new cols: {len(OPDI_EXT_NUM_COLS)}")
    for c in OPDI_EXT_NUM_COLS[:6]:
        cov = out[c].notna().mean() * 100
        mn = out[c].mean()
        print(f"  {c:35s} cov {cov:5.1f}%  mean {mn:.2f}")
