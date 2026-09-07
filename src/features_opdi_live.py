"""OPDI live-taxi rolling features (leak-free).

For each DEP row at airport A, time t: mean and median of ACTUAL taxi-out
(from OPDI events) for all OTHER flights whose runway-entry was in
[t - w, t]. Captures the airport's live operational tempo.

Uses external/opdi/taxi_out_v2.parquet (built by build_opdi_taxi_v2.py).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPDI_TAXI = os.path.join(ROOT, "external", "opdi", "taxi_out_v2.parquet")

WINDOWS_MIN = [30, 60, 120]

OPDI_LIVE_NUM_COLS = []
for w in WINDOWS_MIN:
    OPDI_LIVE_NUM_COLS.append(f"opdi_live_taxi_mean_prev_{w}m")
    OPDI_LIVE_NUM_COLS.append(f"opdi_live_taxi_median_prev_{w}m")
    OPDI_LIVE_NUM_COLS.append(f"opdi_live_taxi_count_prev_{w}m")


def _rolling_stats(query_ts: np.ndarray, ref_ts: np.ndarray,
                   ref_taxi: np.ndarray, window_min: int) -> tuple:
    """For each query, mean/median/count of ref_taxi where ref_ts in [q-window, q]."""
    n = len(query_ts)
    means = np.full(n, np.nan, dtype=np.float32)
    medians = np.full(n, np.nan, dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    if len(ref_ts) == 0:
        return means, medians, counts
    dt = np.timedelta64(window_min * 60, "s")
    # ref_ts already sorted
    lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
    hi = np.searchsorted(ref_ts, query_ts, side="right")
    for i in range(n):
        if hi[i] > lo[i]:
            vals = ref_taxi[lo[i]:hi[i]]
            counts[i] = len(vals)
            means[i] = float(vals.mean())
            medians[i] = float(np.median(vals))
    return means, medians, counts


def add_opdi_live(dep: pd.DataFrame) -> pd.DataFrame:
    """dep needs: ADEP_mvt, mvt_ts. Adds live-tempo features from OPDI taxi records."""
    if not os.path.exists(OPDI_TAXI):
        out = dep.copy()
        for c in OPDI_LIVE_NUM_COLS:
            out[c] = np.nan
        return out

    tx = pd.read_parquet(OPDI_TAXI)
    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    new_cols = {c: np.full(n, np.nan, dtype=np.float32) for c in OPDI_LIVE_NUM_COLS}

    apt_arr = dep["ADEP_mvt"].values
    ts_arr = dep["mvt_ts"].values.astype("datetime64[us]")

    for icao, apt_tx in tx.groupby("osm_airport", sort=False):
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0:
            continue
        q_ts = ts_arr[pos]
        # Sort ref by runway_entry_ts (that's when the taxi COMPLETED and became observable)
        apt_tx_sorted = apt_tx.sort_values("runway_entry_ts")
        ref_ts = apt_tx_sorted["runway_entry_ts"].values.astype("datetime64[us]")
        ref_taxi = apt_tx_sorted["actual_taxi_sec"].values.astype(np.float32)
        for w in WINDOWS_MIN:
            means, medians, counts = _rolling_stats(q_ts, ref_ts, ref_taxi, w)
            new_cols[f"opdi_live_taxi_mean_prev_{w}m"][pos] = means
            new_cols[f"opdi_live_taxi_median_prev_{w}m"][pos] = medians
            new_cols[f"opdi_live_taxi_count_prev_{w}m"][pos] = counts.astype(np.float32)

    for k, v in new_cols.items():
        dep[k] = v
    return dep.set_index("_orig_idx").rename_axis(None)


if __name__ == "__main__":
    import glob
    from features_weather import TARGET_ICAOS
    p = os.path.join(ROOT, "training", "training_2025-06-01_2025-07-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].head(50000).copy()
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    out = add_opdi_live(df)
    print(f"Rows: {len(out):,}")
    print("Coverage by airport (fraction with non-null 60m mean):")
    for a in sorted(out["ADEP_mvt"].unique()):
        sub = out[out["ADEP_mvt"] == a]
        cov = sub["opdi_live_taxi_mean_prev_60m"].notna().mean() * 100
        cnt = sub["opdi_live_taxi_count_prev_60m"].mean()
        mn = sub["opdi_live_taxi_mean_prev_60m"].mean()
        print(f"  {a}: coverage {cov:5.1f}%  avg n_records/query {cnt:5.1f}  avg mean taxi {mn:6.1f}s")
