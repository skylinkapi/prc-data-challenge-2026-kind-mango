"""Section 5.2 turnaround features: aircraft on the same stand.

For each departure at airport A with `MVT_TIME` = t, find the last arrival at
airport A on the same `STAND_mvt` whose in-block is before t and within 12 h.
Emit 6 features:

  ground_time     : MVT - inblk_of_linked_arrival
  slack_eobt      : EOBT_1 - inblk
  slack_sched     : SCHED  - inblk
  in_delay        : inblk  - SCHED_arr
  taxi_in_prev    : inblk  - landing_of_linked_arrival
  link_ambiguous  : another departure from the same stand between inblk and MVT

Caller must pass all_mvt with ADES_mvt (for arrival airport key), STAND_mvt,
MVT_TIME_UTC_mvt (landing time for ARR, MVT for DEP), SCHED_TIME_UTC_mvt,
TAXITIME_SEC_mvt, PHASE_mvt, ADEP_mvt.
"""
import numpy as np
import pandas as pd

TURN_NUM_COLS = [
    "ground_time", "slack_eobt", "slack_sched",
    "in_delay", "taxi_in_prev", "link_ambiguous",
]


def add_turnaround(dep: pd.DataFrame, all_mvt: pd.DataFrame) -> pd.DataFrame:
    """dep needs ADEP_mvt, STAND_mvt, mvt_ts, eobt1_ts, sched_ts.
    all_mvt needs ADEP_mvt, ADES_mvt, PHASE_mvt, mvt_ts, sched_ts, TAXITIME_SEC_mvt, STAND_mvt.
    """
    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    arr = all_mvt[all_mvt["PHASE_mvt"] == "ARR"].copy()
    arr["airport"] = arr["ADES_mvt"]
    arr["landing_ts"] = arr["mvt_ts"].values.astype("datetime64[us]")
    # Fix 3.5 (MODEL_ANALYSIS.md 2026-09-09): read BLOCK_TIME_UTC_mvt directly.
    # The identity `TAXITIME = BLOCK - MVT` holds on every ARR row with 0 s diff.
    # Leave missing block times missing.
    if "BLOCK_TIME_UTC_mvt" in arr.columns:
        arr["inblk_ts"] = pd.to_datetime(arr["BLOCK_TIME_UTC_mvt"], errors="coerce").values.astype("datetime64[us]")
    else:
        tx = arr["TAXITIME_SEC_mvt"].astype(float).fillna(0).clip(0, 7200)
        arr["inblk_ts"] = (arr["landing_ts"] + tx.astype(np.int64).values.astype("timedelta64[s]")).astype("datetime64[us]")
    arr["sched_arr_ts"] = arr["sched_ts"].values.astype("datetime64[us]")
    arr = arr[["airport", "STAND_mvt", "inblk_ts", "landing_ts", "sched_arr_ts"]]
    arr = arr.dropna(subset=["inblk_ts"])

    dep["airport"] = dep["ADEP_mvt"]
    dep["mvt_ts_us"] = dep["mvt_ts"].values.astype("datetime64[us]")
    dep["eobt1_ts_us"] = dep["eobt1_ts"].values.astype("datetime64[us]") if "eobt1_ts" in dep.columns else pd.NaT
    dep["sched_ts_us"] = dep["sched_ts"].values.astype("datetime64[us]")

    tol = pd.Timedelta("12h")
    # merge_asof on inblk_ts <= mvt_ts within airport+stand+tolerance
    arr_sorted = arr.sort_values("inblk_ts").reset_index(drop=True)
    dep_sorted = dep.sort_values("mvt_ts_us").reset_index(drop=True)
    linked = pd.merge_asof(
        dep_sorted, arr_sorted,
        left_on="mvt_ts_us", right_on="inblk_ts",
        by=["airport", "STAND_mvt"],
        tolerance=tol, direction="backward", allow_exact_matches=False,
    )
    # Compute features
    linked["ground_time"] = (linked["mvt_ts_us"] - linked["inblk_ts"]).dt.total_seconds()
    linked["slack_eobt"] = (linked["eobt1_ts_us"] - linked["inblk_ts"]).dt.total_seconds()
    linked["slack_sched"] = (linked["sched_ts_us"] - linked["inblk_ts"]).dt.total_seconds()
    linked["in_delay"] = (linked["inblk_ts"] - linked["sched_arr_ts"]).dt.total_seconds()
    linked["taxi_in_prev"] = (linked["inblk_ts"] - linked["landing_ts"]).dt.total_seconds()

    # link_ambiguous: was there ANOTHER DEP from the same stand between inblk and this MVT?
    dep_all = all_mvt[all_mvt["PHASE_mvt"] == "DEP"].copy()
    dep_all["airport"] = dep_all["ADEP_mvt"]
    dep_all["dep_ts"] = dep_all["mvt_ts"].values.astype("datetime64[us]")
    dep_all = dep_all[["airport", "STAND_mvt", "dep_ts"]].dropna(subset=["dep_ts"])

    amb = np.zeros(len(linked), dtype=np.int8)
    for (apt, stand), grp in dep_all.groupby(["airport", "STAND_mvt"], sort=False):
        stand_dep_ts = np.sort(grp["dep_ts"].values.astype("datetime64[us]"))
        sub = linked[(linked["airport"] == apt) & (linked["STAND_mvt"] == stand) &
                     linked["inblk_ts"].notna()]
        if len(sub) == 0: continue
        for k in sub.index.values:
            in_ts = linked.at[k, "inblk_ts"]
            m_ts = linked.at[k, "mvt_ts_us"]
            if pd.isna(in_ts): continue
            in_np = np.datetime64(in_ts, "us")
            m_np = np.datetime64(m_ts, "us")
            lo = np.searchsorted(stand_dep_ts, in_np, side="left")
            hi = np.searchsorted(stand_dep_ts, m_np, side="right")
            # exclude own row (dep_ts == mvt_ts)
            n_between = int(hi - lo - 1) if hi > lo else 0
            amb[k] = 1 if n_between > 0 else 0
    linked["link_ambiguous"] = amb

    # Return in original order
    out = linked.sort_values("_orig_idx").set_index("_orig_idx").rename_axis(None)
    keep = [c for c in dep.columns if c not in
            {"airport", "mvt_ts_us", "eobt1_ts_us", "sched_ts_us", "_orig_idx"}]
    return out[keep + TURN_NUM_COLS]


if __name__ == "__main__":
    import os, glob, time
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TARGET = {"EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"}
    m = pd.read_parquet(os.path.join(ROOT, "training", "training_2025-07-01_2025-08-01.parquet"))
    m = m[m["ADEP_mvt"].isin(TARGET) | m["ADES_mvt"].isin(TARGET)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    dep = m[(m["PHASE_mvt"] == "DEP") & m["ADEP_mvt"].isin(TARGET)].head(30000).copy()
    dep["eobt1_ts"] = pd.to_datetime(dep["EOBT_1_flt"], errors="coerce")
    t0 = time.time()
    out = add_turnaround(dep, m)
    print(f"n={len(out):,}   wall {time.time()-t0:.1f}s")
    print(f"Coverage of linked arrival: {out['ground_time'].notna().mean()*100:.1f}%")
    print(out[TURN_NUM_COLS].describe().round(1).to_string())
