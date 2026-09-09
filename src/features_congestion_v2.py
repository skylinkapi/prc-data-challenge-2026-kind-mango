"""Congestion features v2. Fixes the arrival-key bug from MODEL_ANALYSIS.md 3.3.

For each DEP row at airport A at time t, computes:
  - dep_load_prev_{15,30,60}m  (unchanged)
  - dep_same_rwy_prev_{15,30,60}m (unchanged)
  - dep_queue_next_10m (unchanged)
  - arr_load_prev_{15,30,60}m       — ARR rows with ADES_mvt == A
  - arr_same_rwy_prev_15m           — ARR rows on same runway
  - arr_same_rwy_next_10m           — ARR rows on same runway in next 10 min
  - arr_taxi_in_mean_60m            — mean TAXITIME_SEC_mvt of ARR rows in prev 60 min

Caller must pass `all_mvt` UNFILTERED by ADEP: needs both ADEP_mvt and ADES_mvt
and TAXITIME_SEC_mvt (for ARR taxi-in).
"""
import numpy as np
import pandas as pd

WINDOWS_MIN = [15, 30, 60]

CONG_V2_NUM_COLS = [
    "dep_load_prev_15m", "dep_load_prev_30m", "dep_load_prev_60m",
    "arr_load_prev_15m", "arr_load_prev_30m", "arr_load_prev_60m",
    "dep_same_rwy_prev_15m", "dep_same_rwy_prev_30m", "dep_same_rwy_prev_60m",
    "dep_queue_next_10m",
    "arr_same_rwy_prev_15m", "arr_same_rwy_next_10m",
    "arr_taxi_in_mean_60m",
]


def _count_in_window(query_ts, ref_ts, window_min, direction):
    dt = np.timedelta64(window_min * 60, "s")
    if direction == "backward":
        lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
        hi = np.searchsorted(ref_ts, query_ts, side="right")
    else:
        lo = np.searchsorted(ref_ts, query_ts, side="left")
        hi = np.searchsorted(ref_ts, query_ts + dt, side="right")
    return (hi - lo).astype(np.int32)


def _sum_in_window_backward(query_ts, ref_ts, ref_val, window_min):
    """Cumsum trick: sum(ref_val[i] for i where t-w <= ref_ts[i] < t)."""
    dt = np.timedelta64(window_min * 60, "s")
    cs = np.concatenate([[0.0], np.cumsum(ref_val)])
    lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
    hi = np.searchsorted(ref_ts, query_ts, side="right")
    return cs[hi] - cs[lo], (hi - lo)


def add_congestion_v2(dep: pd.DataFrame, all_mvt: pd.DataFrame) -> pd.DataFrame:
    """dep: DEP subset with ADEP_mvt, mvt_ts, RUNWAY_mvt.
    all_mvt: full corpus with ADEP_mvt, ADES_mvt, PHASE_mvt, mvt_ts, RUNWAY_mvt,
             TAXITIME_SEC_mvt (needed for ARR taxi-in mean).
    """
    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    new_cols = {c: np.zeros(n, dtype=np.float32) for c in CONG_V2_NUM_COLS}
    for c in ["dep_queue_next_10m", "arr_same_rwy_next_10m",
              "dep_load_prev_15m", "dep_load_prev_30m", "dep_load_prev_60m",
              "arr_load_prev_15m", "arr_load_prev_30m", "arr_load_prev_60m",
              "dep_same_rwy_prev_15m", "dep_same_rwy_prev_30m", "dep_same_rwy_prev_60m",
              "arr_same_rwy_prev_15m"]:
        new_cols[c] = new_cols[c].astype(np.int32)

    apt_arr = dep["ADEP_mvt"].values
    rwy_arr = dep["RUNWAY_mvt"].values
    ts_arr  = dep["mvt_ts"].values.astype("datetime64[us]")

    # Precompute apt_key on all_mvt: DEP -> ADEP, ARR -> ADES
    is_arr = (all_mvt["PHASE_mvt"] == "ARR").values
    apt_key = np.where(is_arr,
                       all_mvt["ADES_mvt"].values,
                       all_mvt["ADEP_mvt"].values)
    all_mvt = all_mvt.assign(_apt_key=apt_key)

    for icao, grp in all_mvt.groupby("_apt_key", sort=False):
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0:
            continue
        q_ts = ts_arr[pos]

        dep_ts = np.sort(grp.loc[grp["PHASE_mvt"] == "DEP", "mvt_ts"]
                         .values.astype("datetime64[us]"))
        arr_rows = grp.loc[grp["PHASE_mvt"] == "ARR"].sort_values("mvt_ts")
        arr_ts = arr_rows["mvt_ts"].values.astype("datetime64[us]")
        arr_taxi = arr_rows["TAXITIME_SEC_mvt"].astype(float).fillna(0).values
        arr_rwy = arr_rows["RUNWAY_mvt"].values

        for w in WINDOWS_MIN:
            new_cols[f"dep_load_prev_{w}m"][pos] = _count_in_window(q_ts, dep_ts, w, "backward")
            new_cols[f"arr_load_prev_{w}m"][pos] = _count_in_window(q_ts, arr_ts, w, "backward")
        new_cols["dep_queue_next_10m"][pos] = _count_in_window(q_ts, dep_ts, 10, "forward")

        # ARR taxi-in mean, prev 60 min
        s, c = _sum_in_window_backward(q_ts, arr_ts, arr_taxi, 60)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(c > 0, s / c, np.nan)
        new_cols["arr_taxi_in_mean_60m"][pos] = m.astype(np.float32)

        # Per-runway DEP + ARR
        for rwy, sub in grp[grp["PHASE_mvt"] == "DEP"].groupby("RUNWAY_mvt"):
            rwy_ts = np.sort(sub["mvt_ts"].values.astype("datetime64[us]"))
            rwy_pos = pos[rwy_arr[pos] == rwy]
            if rwy_pos.size == 0: continue
            q_rwy = ts_arr[rwy_pos]
            for w in WINDOWS_MIN:
                new_cols[f"dep_same_rwy_prev_{w}m"][rwy_pos] = _count_in_window(q_rwy, rwy_ts, w, "backward")

        for rwy in np.unique(arr_rwy[arr_rwy != None]):  # noqa
            if pd.isna(rwy): continue
            mask = arr_rwy == rwy
            rwy_ts = arr_ts[mask]
            rwy_pos = pos[rwy_arr[pos] == rwy]
            if rwy_pos.size == 0: continue
            q_rwy = ts_arr[rwy_pos]
            new_cols["arr_same_rwy_prev_15m"][rwy_pos] = _count_in_window(q_rwy, rwy_ts, 15, "backward")
            new_cols["arr_same_rwy_next_10m"][rwy_pos] = _count_in_window(q_rwy, rwy_ts, 10, "forward")

    for k, v in new_cols.items():
        dep[k] = v
    return dep.set_index("_orig_idx").rename_axis(None)


if __name__ == "__main__":
    import os, glob
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = pd.read_parquet(os.path.join(ROOT, "training", "training_2025-07-01_2025-08-01.parquet"))
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    tgt = {"EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"}
    # Keep rows where either endpoint is a target airport
    m = m[m["ADEP_mvt"].isin(tgt) | m["ADES_mvt"].isin(tgt)].copy()
    dep = m[(m["PHASE_mvt"] == "DEP") & m["ADEP_mvt"].isin(tgt)].head(20000).copy()
    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt","TAXITIME_SEC_mvt"]]
    out = add_congestion_v2(dep, ctx)
    print("Sample cols:")
    print(out[CONG_V2_NUM_COLS].describe().round(2).to_string())
    print("\nARR coverage (v1 filter dropped ~85%):")
    for c in ["arr_load_prev_15m", "arr_load_prev_30m", "arr_same_rwy_prev_15m",
              "arr_same_rwy_next_10m", "arr_taxi_in_mean_60m"]:
        nz = (out[c] > 0).mean() if c != "arr_taxi_in_mean_60m" else (~out[c].isna()).mean()
        print(f"  {c:30s} nonzero/notna {nz*100:5.1f}%")
