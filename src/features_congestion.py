"""Congestion features from movement records themselves.

For each DEP row at airport A and take-off time t, computes rolling counts of
DEP and ARR movements around t. Available at inference because MVT_TIME_UTC_mvt
is present on ranking DEP rows and all ARR rows carry taxi-in + timestamps.
"""
import numpy as np
import pandas as pd

WINDOWS_MIN = [15, 30, 60]


def _search_window_counts(query_ts: np.ndarray, ref_ts: np.ndarray,
                          window_min: int, direction: str) -> np.ndarray:
    """Counts of ref_ts inside [query - window, query] (backward) or
    [query, query + window] (forward). ref_ts must be sorted."""
    dt = np.timedelta64(window_min * 60, "s")
    if direction == "backward":
        lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
        hi = np.searchsorted(ref_ts, query_ts, side="right")
    else:
        lo = np.searchsorted(ref_ts, query_ts, side="left")
        hi = np.searchsorted(ref_ts, query_ts + dt, side="right")
    return (hi - lo).astype(np.int32)


def add_congestion(dep: pd.DataFrame, all_mvt: pd.DataFrame) -> pd.DataFrame:
    """dep: DEP subset with columns ADEP_mvt, mvt_ts, RUNWAY_mvt.
    all_mvt: full movement corpus (both DEP+ARR).
    Adds count features. Preserves dep's original index.
    """
    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    new_cols = {}
    for w in WINDOWS_MIN:
        new_cols[f"dep_load_prev_{w}m"] = np.zeros(n, dtype=np.int32)
        new_cols[f"arr_load_prev_{w}m"] = np.zeros(n, dtype=np.int32)
        new_cols[f"dep_same_rwy_prev_{w}m"] = np.zeros(n, dtype=np.int32)
    new_cols["dep_queue_next_10m"] = np.zeros(n, dtype=np.int32)

    apt_arr = dep["ADEP_mvt"].values
    rwy_arr = dep["RUNWAY_mvt"].values
    ts_arr = dep["mvt_ts"].values.astype("datetime64[us]")

    for icao, grp in all_mvt.groupby("ADEP_mvt", sort=False):
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0:
            continue
        q_ts = ts_arr[pos]

        dep_ts = np.sort(grp.loc[grp["PHASE_mvt"] == "DEP", "mvt_ts"]
                         .values.astype("datetime64[us]"))
        arr_ts = np.sort(grp.loc[grp["PHASE_mvt"] == "ARR", "mvt_ts"]
                         .values.astype("datetime64[us]"))

        for w in WINDOWS_MIN:
            new_cols[f"dep_load_prev_{w}m"][pos] = \
                _search_window_counts(q_ts, dep_ts, w, "backward")
            new_cols[f"arr_load_prev_{w}m"][pos] = \
                _search_window_counts(q_ts, arr_ts, w, "backward")
        new_cols["dep_queue_next_10m"][pos] = \
            _search_window_counts(q_ts, dep_ts, 10, "forward")

        for rwy, sub in grp[grp["PHASE_mvt"] == "DEP"].groupby("RUNWAY_mvt"):
            rwy_ts = np.sort(sub["mvt_ts"].values.astype("datetime64[us]"))
            rwy_pos = pos[rwy_arr[pos] == rwy]
            if rwy_pos.size == 0:
                continue
            q_rwy = ts_arr[rwy_pos]
            for w in WINDOWS_MIN:
                new_cols[f"dep_same_rwy_prev_{w}m"][rwy_pos] = \
                    _search_window_counts(q_rwy, rwy_ts, w, "backward")

    for k, v in new_cols.items():
        dep[k] = v
    return dep.set_index("_orig_idx").rename_axis(None)


if __name__ == "__main__":
    import os, glob
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(ROOT, "training", "training_2025-01-01_2025-02-01.parquet")
    m = pd.read_parquet(p)
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    tgt = {"EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
           "LFPG", "LIRF", "LTFM", "LSZH"}
    m = m[m["ADEP_mvt"].isin(tgt)]
    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    out = add_congestion(dep, m)
    cols = ["ADEP_mvt", "mvt_ts",
            "dep_load_prev_15m", "dep_load_prev_30m", "dep_load_prev_60m",
            "arr_load_prev_15m", "dep_same_rwy_prev_15m", "dep_queue_next_10m"]
    print(out[cols].head(15).to_string())
    print("\nStats per airport (mean load in prev 15 min):")
    print(out.groupby("ADEP_mvt")["dep_load_prev_15m"].agg(["mean", "max"]).round(1).to_string())
