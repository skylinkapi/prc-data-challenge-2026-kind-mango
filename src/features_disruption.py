"""Section 5.2 disruption meters: rolling 60-min window before MVT.

For each departure at airport A at time t, compute over OTHER visible rows at A:
  dep_sd_mean_60m  : mean of clip(sd, -1800, 14400) for departures in [t-60m, t)
  dep_fnull_60m    : share of departures in the window with null FLIGHT_ID_mvt
  arr_txi_mean_60m : mean taxi-in of arrivals in the window (30<=y<=7200)
  arr_txi_p90_60m  : 90th percentile of the same

Caller must pass all_mvt with ADEP_mvt, ADES_mvt, PHASE_mvt, mvt_ts, sched_ts,
FLIGHT_ID_mvt, TAXITIME_SEC_mvt.
"""
import numpy as np
import pandas as pd

DISR_NUM_COLS = [
    "dep_sd_mean_60m", "dep_fnull_60m",
    "arr_txi_mean_60m", "arr_txi_p90_60m",
]

WINDOW_MIN = 60


def _mean_in_window(query_ts, ref_ts, ref_val, w_min, exclude_self=False):
    """For each query t, mean of ref_val where ref_ts in [t-w, t) (or t]."""
    dt = np.timedelta64(w_min * 60, "s")
    lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
    hi = np.searchsorted(ref_ts, query_ts, side=("left" if exclude_self else "right"))
    cs = np.concatenate([[0.0], np.cumsum(ref_val)])
    tot = cs[hi] - cs[lo]
    cnt = hi - lo
    return np.where(cnt > 0, tot / cnt, np.nan), cnt


def _p90_in_window(query_ts, ref_ts, ref_val, w_min):
    """P90 in window. Slower path; ref_ts pre-sorted."""
    dt = np.timedelta64(w_min * 60, "s")
    lo = np.searchsorted(ref_ts, query_ts - dt, side="left")
    hi = np.searchsorted(ref_ts, query_ts, side="right")
    out = np.full(len(query_ts), np.nan, dtype=np.float32)
    for i in range(len(query_ts)):
        if hi[i] > lo[i]:
            out[i] = float(np.percentile(ref_val[lo[i]:hi[i]], 90))
    return out


def add_disruption(dep: pd.DataFrame, all_mvt: pd.DataFrame) -> pd.DataFrame:
    """dep: DEP rows with ADEP_mvt, mvt_ts, sched_ts.
    all_mvt: full corpus with ADES_mvt too and FLIGHT_ID_mvt + TAXITIME_SEC_mvt.
    """
    dep = dep.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(dep)
    for c in DISR_NUM_COLS:
        dep[c] = np.nan

    apt_arr = dep["ADEP_mvt"].values
    ts_arr = dep["mvt_ts"].values.astype("datetime64[us]")

    # Build airport-keyed views. For DEP: airport = ADEP_mvt. For ARR: airport = ADES_mvt.
    is_arr = (all_mvt["PHASE_mvt"] == "ARR").values
    apt_key = np.where(is_arr, all_mvt["ADES_mvt"].values, all_mvt["ADEP_mvt"].values)
    ctx = all_mvt.assign(_apt=apt_key)

    for icao in pd.unique(apt_arr):
        pos = np.where(apt_arr == icao)[0]
        if pos.size == 0: continue
        q = ts_arr[pos]

        # Departures at the airport
        d = ctx[(ctx["_apt"] == icao) & (ctx["PHASE_mvt"] == "DEP")].copy()
        if len(d) > 0:
            d["_ts"] = d["mvt_ts"].values.astype("datetime64[us]")
            d = d.sort_values("_ts")
            d_ts = d["_ts"].values
            sd_raw = (d["mvt_ts"] - d["sched_ts"]).dt.total_seconds().clip(-1800, 14400).fillna(0).values
            null_ind = d["FLIGHT_ID_mvt"].isna().astype(np.float32).values
            m_sd, _ = _mean_in_window(q, d_ts, sd_raw, WINDOW_MIN, exclude_self=True)
            m_null, _ = _mean_in_window(q, d_ts, null_ind, WINDOW_MIN, exclude_self=True)
            dep.loc[pos, "dep_sd_mean_60m"] = m_sd.astype(np.float32)
            dep.loc[pos, "dep_fnull_60m"] = m_null.astype(np.float32)

        # Arrivals at the airport
        a = ctx[(ctx["_apt"] == icao) & (ctx["PHASE_mvt"] == "ARR")].copy()
        if len(a) > 0:
            a["_ts"] = a["mvt_ts"].values.astype("datetime64[us]")
            a["_txi"] = a["TAXITIME_SEC_mvt"].astype(float)
            a = a[(a["_txi"] >= 30) & (a["_txi"] <= 7200)]
            if len(a) > 0:
                a = a.sort_values("_ts")
                a_ts = a["_ts"].values
                a_val = a["_txi"].values.astype(np.float32)
                m_a, _ = _mean_in_window(q, a_ts, a_val, WINDOW_MIN, exclude_self=False)
                p90 = _p90_in_window(q, a_ts, a_val, WINDOW_MIN)
                dep.loc[pos, "arr_txi_mean_60m"] = m_a.astype(np.float32)
                dep.loc[pos, "arr_txi_p90_60m"] = p90.astype(np.float32)

    return dep.set_index("_orig_idx").rename_axis(None)


if __name__ == "__main__":
    import os, time
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TARGET = {"EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"}
    m = pd.read_parquet(os.path.join(ROOT, "training", "training_2025-07-01_2025-08-01.parquet"))
    m = m[m["ADEP_mvt"].isin(TARGET) | m["ADES_mvt"].isin(TARGET)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    dep = m[(m["PHASE_mvt"] == "DEP") & m["ADEP_mvt"].isin(TARGET)].head(30000).copy()
    t0 = time.time()
    out = add_disruption(dep, m)
    print(f"n={len(out):,}   wall {time.time()-t0:.1f}s")
    for c in DISR_NUM_COLS:
        cov = out[c].notna().mean() * 100
        print(f"  {c:20s} cov {cov:5.1f}%  mean {out[c].mean():.2f}")
