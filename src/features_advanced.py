"""Advanced physically-motivated features for taxi-out prediction.

All computed from MVT_TIME_UTC_mvt and SCHED_TIME_UTC_mvt, both present
on ranking DEP rows. No leakage.

1. sched_pushback_load_15m : count of DEPs whose SCHEDULED time falls in
   [t-15m, t] at the same airport. Physical proxy for concurrent pushback
   demand from the schedule.
2. runway_diversity_prev_30m : distinct runways used for DEPs in the prev
   30 min at the airport. Captures runway-configuration state.
3. secs_since_last_dep_same_rwy : seconds since the previous DEP on the
   same runway at this airport. Runway occupancy rate proxy.
4. rwy_bank_intensity_5m : DEPs on same runway in [t-5m, t] as a rate
   (count / 5 min). Fine-grained queue-at-runway proxy.
5. ades_arr_atfm_delay_today : for each flight's destination, daily
   arrival ATFM delay total (from Eurocontrol daily). NaN when the
   destination is outside the 10 target airports.
"""
import numpy as np
import pandas as pd


ADV_NUM_COLS = [
    "sched_pushback_load_15m",
    "runway_diversity_prev_30m",
    "secs_since_last_dep_same_rwy",
    "rwy_bank_intensity_5m",
    "ades_arr_atfm_delay_today",
]


def _count_in_window_backward(query_ts: np.ndarray, ref_ts: np.ndarray,
                              window_min: int) -> np.ndarray:
    """Number of ref_ts in [query - window, query]. ref_ts pre-sorted."""
    dt = np.timedelta64(window_min * 60, "s")
    lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
    hi = np.searchsorted(ref_ts, query_ts, side="right")
    return (hi - lo).astype(np.int32)


def add_advanced(dep: pd.DataFrame, all_mvt: pd.DataFrame,
                 daily_ec: pd.DataFrame = None) -> pd.DataFrame:
    """dep needs: ADEP_mvt, mvt_ts, sched_ts, RUNWAY_mvt, ADES_mvt.
    all_mvt needs: ADEP_mvt, PHASE_mvt, mvt_ts, sched_ts, RUNWAY_mvt.
    daily_ec (optional): Eurocontrol daily table with columns FLT_DATE,
      APT_ICAO, airport_arrival_atfm_delay__DLY_APT_ARR_1.
    """
    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    apt_arr = dep["ADEP_mvt"].values
    rwy_arr = dep["RUNWAY_mvt"].values
    mvt_ts_arr = dep["mvt_ts"].values.astype("datetime64[us]")
    sch_ts_arr = dep["sched_ts"].values.astype("datetime64[us]")

    out_cols = {c: np.zeros(n, dtype=np.float32) for c in ADV_NUM_COLS[:4]}

    for icao, grp in all_mvt.groupby("ADEP_mvt", sort=False):
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0:
            continue
        q_mvt = mvt_ts_arr[pos]
        q_sch = sch_ts_arr[pos]

        # 1) sched pushback load: DEPs by SCHEDULED time
        dep_sched = grp.loc[grp["PHASE_mvt"] == "DEP", "sched_ts"] \
            .dropna().values.astype("datetime64[us]")
        dep_sched.sort()
        out_cols["sched_pushback_load_15m"][pos] = \
            _count_in_window_backward(q_sch, dep_sched, 15)

        # 2) runway diversity: unique RUNWAY_mvt in prev 30 min at airport
        dep_rows = grp[grp["PHASE_mvt"] == "DEP"].copy()
        dep_rows["_ts"] = dep_rows["mvt_ts"].values.astype("datetime64[us]")
        # For each query, count distinct runways in [t-30m, t]
        # Efficient path: pre-sort by _ts and iterate with a rolling window
        dr_sorted = dep_rows.sort_values("_ts")
        dr_ts = dr_sorted["_ts"].values
        dr_rwy = dr_sorted["RUNWAY_mvt"].fillna("_NA").values
        dt30 = np.timedelta64(30 * 60, "s")
        for k, qt in enumerate(q_mvt):
            lo = np.searchsorted(dr_ts, qt - dt30, side="left")
            hi = np.searchsorted(dr_ts, qt, side="right")
            if hi > lo:
                out_cols["runway_diversity_prev_30m"][pos[k]] = \
                    len(np.unique(dr_rwy[lo:hi]))

        # Per-runway indices
        for rwy, sub in dep_rows.groupby("RUNWAY_mvt"):
            rwy_ts = np.sort(sub["_ts"].values)
            mask = rwy_arr[pos] == rwy
            if not mask.any():
                continue
            rwy_pos = pos[mask]
            q_rwy = mvt_ts_arr[rwy_pos]

            # 3) secs since last DEP on same runway (before this one, strictly)
            idx = np.searchsorted(rwy_ts, q_rwy, side="left")
            prev = np.where(idx > 0, rwy_ts[np.clip(idx - 1, 0, len(rwy_ts) - 1)],
                            q_rwy - np.timedelta64(3600, "s"))
            secs = (q_rwy - prev).astype("timedelta64[s]").astype(np.int32)
            out_cols["secs_since_last_dep_same_rwy"][rwy_pos] = np.clip(secs, 0, 3600)

            # 4) rwy bank intensity in prev 5 min
            out_cols["rwy_bank_intensity_5m"][rwy_pos] = \
                _count_in_window_backward(q_rwy, rwy_ts, 5).astype(np.float32)

    for c, v in out_cols.items():
        dep[c] = v

    # 5) ADES arrival ATFM delay today
    if daily_ec is not None and "airport_arrival_atfm_delay__DLY_APT_ARR_1" in daily_ec.columns:
        d = daily_ec[["FLT_DATE", "APT_ICAO", "airport_arrival_atfm_delay__DLY_APT_ARR_1"]].copy()
        d["FLT_DATE"] = pd.to_datetime(d["FLT_DATE"])
        d = d.rename(columns={"FLT_DATE": "_date",
                              "APT_ICAO": "ADES_mvt",
                              "airport_arrival_atfm_delay__DLY_APT_ARR_1": "ades_arr_atfm_delay_today"})
        dep["_date"] = pd.to_datetime(dep["mvt_ts"].dt.date)
        dep = dep.merge(d, on=["ADES_mvt", "_date"], how="left").drop(columns=["_date"])
    else:
        dep["ades_arr_atfm_delay_today"] = np.nan

    return dep.set_index("_orig_idx").rename_axis(None)


if __name__ == "__main__":
    import glob, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(ROOT, "training", "training_2025-06-01_2025-07-01.parquet")
    m = pd.read_parquet(p)
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    tgt = {"EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"}
    m = m[m["ADEP_mvt"].isin(tgt)]
    dep = m[m["PHASE_mvt"] == "DEP"].head(30000).copy()
    ec = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    out = add_advanced(dep, m, daily_ec=ec)
    print(f"Rows: {len(out):,}")
    print(out[ADV_NUM_COLS].describe().round(2).to_string())
    print(f"\nCoverage:\n{(~out[ADV_NUM_COLS].isna()).mean().round(3).to_string()}")
