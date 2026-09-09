"""OPDI live-taxi v2. Step 5 fix: NaN out airport-months with <300 records.

Same features as v1 but zeroes out the whole airport-month whenever the OPDI
taxi record count for that airport-month is below the drift threshold. The
model then learns the missing pattern instead of extrapolating on a feature
that is dense in training and empty on ranking.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPDI_TAXI = os.path.join(ROOT, "external", "opdi", "taxi_out_v2.parquet")

WINDOWS_MIN = [30, 60, 120]
MIN_APT_MONTH_RECORDS = 300

OPDI_LIVE_V2_NUM_COLS = []
for w in WINDOWS_MIN:
    OPDI_LIVE_V2_NUM_COLS.append(f"opdi_live_taxi_mean_prev_{w}m")
    OPDI_LIVE_V2_NUM_COLS.append(f"opdi_live_taxi_median_prev_{w}m")
    OPDI_LIVE_V2_NUM_COLS.append(f"opdi_live_taxi_count_prev_{w}m")


def _rolling_stats(query_ts, ref_ts, ref_taxi, window_min):
    n = len(query_ts)
    means = np.full(n, np.nan, dtype=np.float32)
    medians = np.full(n, np.nan, dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    if len(ref_ts) == 0:
        return means, medians, counts
    dt = np.timedelta64(window_min * 60, "s")
    lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
    hi = np.searchsorted(ref_ts, query_ts, side="right")
    for i in range(n):
        if hi[i] > lo[i]:
            vals = ref_taxi[lo[i]:hi[i]]
            counts[i] = len(vals)
            means[i] = float(vals.mean())
            medians[i] = float(np.median(vals))
    return means, medians, counts


def add_opdi_live_v2(dep: pd.DataFrame) -> pd.DataFrame:
    """As v1 but per-airport-month gate."""
    if not os.path.exists(OPDI_TAXI):
        out = dep.copy()
        for c in OPDI_LIVE_V2_NUM_COLS:
            out[c] = np.nan
        return out

    tx = pd.read_parquet(OPDI_TAXI)
    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    new_cols = {c: np.full(n, np.nan, dtype=np.float32) for c in OPDI_LIVE_V2_NUM_COLS}

    apt_arr = dep["ADEP_mvt"].values
    ts_arr = dep["mvt_ts"].values.astype("datetime64[us]")
    dep_month = pd.to_datetime(dep["mvt_ts"]).dt.to_period("M").astype(str).values

    # Count records per (airport, YYYY-MM)
    tx_ts = pd.to_datetime(tx["runway_entry_ts"])
    tx_month = tx_ts.dt.to_period("M").astype(str).values
    tx["_month"] = tx_month
    apt_month_counts = tx.groupby(["osm_airport", "_month"]).size()
    good_am = set(apt_month_counts[apt_month_counts >= MIN_APT_MONTH_RECORDS].index)

    for icao, apt_tx in tx.groupby("osm_airport", sort=False):
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0:
            continue
        # Filter position mask by (icao, month) presence
        row_months = dep_month[pos]
        keep = np.array([(icao, m) in good_am for m in row_months])
        pos_ok = pos[keep]
        if pos_ok.size == 0:
            continue
        q_ts = ts_arr[pos_ok]
        apt_tx_sorted = apt_tx.sort_values("runway_entry_ts")
        ref_ts = apt_tx_sorted["runway_entry_ts"].values.astype("datetime64[us]")
        ref_taxi = apt_tx_sorted["actual_taxi_sec"].values.astype(np.float32)
        for w in WINDOWS_MIN:
            means, medians, counts = _rolling_stats(q_ts, ref_ts, ref_taxi, w)
            new_cols[f"opdi_live_taxi_mean_prev_{w}m"][pos_ok] = means
            new_cols[f"opdi_live_taxi_median_prev_{w}m"][pos_ok] = medians
            new_cols[f"opdi_live_taxi_count_prev_{w}m"][pos_ok] = counts.astype(np.float32)

    for k, v in new_cols.items():
        dep[k] = v
    return dep.set_index("_orig_idx").rename_axis(None)
