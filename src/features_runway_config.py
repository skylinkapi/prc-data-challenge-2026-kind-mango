"""Step 7 of MODEL_ANALYSIS.md: runway configuration categorical.

For each DEP row at airport A, time t, computes the sorted set of RUNWAY_mvt
values used by any DEP or ARR at A in [t - 30 min, t]. Encoded as a "|"-joined
string so LightGBM can treat it as a category. The taxi path length depends on
the airport's active runway configuration, not on the departure runway alone.

Caller must pass all_mvt UNFILTERED by ADEP: needs both ADEP_mvt and ADES_mvt
(so ARR rows at target airports are visible) plus PHASE_mvt, mvt_ts, RUNWAY_mvt.
"""
import numpy as np
import pandas as pd

WINDOW_MIN = 30

RWY_CONFIG_CAT_COLS = ["rwy_config_prev_30m"]


def add_runway_config(dep: pd.DataFrame, all_mvt: pd.DataFrame) -> pd.DataFrame:
    """dep: DEP subset with ADEP_mvt, mvt_ts.
    all_mvt: DEP+ARR with ADEP_mvt, ADES_mvt, PHASE_mvt, mvt_ts, RUNWAY_mvt.
    Adds `rwy_config_prev_30m` (object; caller casts to Categorical).
    """
    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    out = np.full(n, "", dtype=object)

    apt_arr = dep["ADEP_mvt"].values
    ts_arr = dep["mvt_ts"].values.astype("datetime64[us]")

    is_arr = (all_mvt["PHASE_mvt"] == "ARR").values
    apt_key = np.where(is_arr, all_mvt["ADES_mvt"].values, all_mvt["ADEP_mvt"].values)
    all_mvt = all_mvt.assign(_apt_key=apt_key)

    dt = np.timedelta64(WINDOW_MIN * 60, "s")
    for icao, grp in all_mvt.groupby("_apt_key", sort=False):
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0: continue
        q_ts = ts_arr[pos]

        sub = grp[grp["RUNWAY_mvt"].notna()].copy()
        if len(sub) == 0: continue
        sub["_ts"] = sub["mvt_ts"].values.astype("datetime64[us]")
        sub = sub.sort_values("_ts")
        s_ts = sub["_ts"].values
        s_rwy = sub["RUNWAY_mvt"].values.astype(str)

        lo = np.searchsorted(s_ts, q_ts - dt, side="left")
        hi = np.searchsorted(s_ts, q_ts, side="right")
        for k in range(len(pos)):
            if hi[k] > lo[k]:
                uniq = sorted(set(s_rwy[lo[k]:hi[k]]))
                out[pos[k]] = "|".join(uniq)

    dep["rwy_config_prev_30m"] = out
    dep.loc[dep["rwy_config_prev_30m"] == "", "rwy_config_prev_30m"] = np.nan
    return dep.set_index("_orig_idx").rename_axis(None)


if __name__ == "__main__":
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = pd.read_parquet(os.path.join(ROOT, "training", "training_2025-07-01_2025-08-01.parquet"))
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    tgt = {"EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"}
    m = m[m["ADEP_mvt"].isin(tgt) | m["ADES_mvt"].isin(tgt)].copy()
    dep = m[(m["PHASE_mvt"] == "DEP") & m["ADEP_mvt"].isin(tgt)].head(20000).copy()
    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]]
    out = add_runway_config(dep, ctx)
    print(f"Rows: {len(out):,}   NaN config: {out['rwy_config_prev_30m'].isna().sum():,}")
    print("\nTop 15 configs per airport (July 2025 sample):")
    for a in sorted(out["ADEP_mvt"].unique()):
        top = out[out["ADEP_mvt"] == a]["rwy_config_prev_30m"].value_counts().head(3)
        print(f"\n  {a}:")
        for cfg, n in top.items():
            print(f"    {cfg:40s}  n={n}")
