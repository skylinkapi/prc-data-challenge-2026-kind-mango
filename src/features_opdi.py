"""OPDI-derived live ground-activity features.

Leak-free by construction: rolling counts of OTHER flights' events in a
backward time window at the same airport. Never uses the current flight's
own OPDI events.

Coverage varies by airport (OSM parking-position tagging is uneven):
  LSZH, EDDF: rich for all event types
  EHAM, EGLL, LFPG, LEBL, LEMD, LIRF, EDDM: entry-runway/entry-taxiway ok,
    exit-parking sparse
  LTFM: near-empty; features return NaN (LGBM handles natively)
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPDI_EVENTS = os.path.join(ROOT, "external", "opdi", "events_all.parquet")

WINDOWS_MIN = [15, 30, 60]

OPDI_NUM_COLS = []
for w in WINDOWS_MIN:
    OPDI_NUM_COLS.append(f"opdi_runway_entries_prev_{w}m")
    OPDI_NUM_COLS.append(f"opdi_taxiway_events_prev_{w}m")
    OPDI_NUM_COLS.append(f"opdi_pushback_events_prev_{w}m")


def _window_counts(query_ts: np.ndarray, ref_ts: np.ndarray, window_min: int) -> np.ndarray:
    """Count of ref_ts in [q - window, q] (backward). ref_ts pre-sorted."""
    if len(ref_ts) == 0 or len(query_ts) == 0:
        return np.zeros(len(query_ts), dtype=np.int32)
    dt = np.timedelta64(window_min * 60, "s")
    lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
    hi = np.searchsorted(ref_ts, query_ts, side="right")
    return (hi - lo).astype(np.int32)


def add_opdi(dep: pd.DataFrame) -> pd.DataFrame:
    """dep needs: ADEP_mvt, mvt_ts. Adds OPDI rolling-count features.

    For each DEP row at airport A, time t:
      opdi_runway_entries_prev_{w}m  = OPDI entry-runway events at A in [t-w, t]
      opdi_taxiway_events_prev_{w}m  = OPDI entry-taxiway events at A in [t-w, t]
      opdi_pushback_events_prev_{w}m = OPDI exit-parking_position events at A in [t-w, t]
    """
    if not os.path.exists(OPDI_EVENTS):
        out = dep.copy()
        for c in OPDI_NUM_COLS:
            out[c] = np.nan
        return out

    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    new_cols = {c: np.full(n, np.nan, dtype=np.float32) for c in OPDI_NUM_COLS}

    apt_arr = dep["ADEP_mvt"].values
    ts_arr = dep["mvt_ts"].values.astype("datetime64[us]")

    type_map = {
        "entry-runway": "runway_entries",
        "entry-taxiway": "taxiway_events",
        "exit-parking_position": "pushback_events",
    }

    # Stream events per airport using pyarrow filter push-down to keep RAM low.
    import pyarrow.parquet as pq
    unique_apts = pd.unique(apt_arr)
    for icao in unique_apts:
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0:
            continue
        q_ts = ts_arr[pos]

        # Only load event_time + type for this airport
        try:
            apt_ev = pq.read_table(
                OPDI_EVENTS,
                columns=["type", "event_time", "osm_airport"],
                filters=[("osm_airport", "=", str(icao))],
            ).to_pandas()
        except Exception:
            continue

        for evt_type, suffix in type_map.items():
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
    out = add_opdi(df)
    print(f"Rows: {len(out):,}")
    print("\nCoverage per airport (any non-null across new cols):")
    for a in sorted(out["ADEP_mvt"].unique()):
        sub = out[out["ADEP_mvt"] == a]
        for c in OPDI_NUM_COLS[:3]:   # 15m of each type
            cov = sub[c].notna().mean() * 100
            print(f"  {a} {c:40s} cov {cov:5.1f}%  mean {sub[c].mean():6.1f}")
