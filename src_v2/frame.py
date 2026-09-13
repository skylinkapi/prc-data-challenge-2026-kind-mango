"""Single frame builder for 2025 training rows and 2026 ranking rows (P1).

Reads the 12 monthly training parquets and the ranking parquet from the same
paths every stage uses. Adds the anchor deltas, the neighbour tempo,
take-off order, stand gap, queue-between-anchors, plan-taxi residual and
weather. Writes one parquet per year to models/v2/cache/.

Missing values stay NaN (A6). No zero fills, no default constants.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src_v2 import config as C

log = logging.getLogger(__name__)

RAW_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt",
    "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt",
    "AIRCRAFT_OPERATOR_flt",
]

_NS_MIN = np.int64(60) * 1_000_000_000


def _norm_ts(s: pd.Series) -> pd.Series:
    """Coerce to tz-aware UTC ns. Training parquets ship datetime64[us]."""
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.dtype.unit != "ns":
        ts = ts.astype("datetime64[ns, UTC]")
    return ts


def _to_ns(ts: pd.Series) -> np.ndarray:
    return ts.astype("datetime64[ns, UTC]").astype("int64").values


def _read_training(train_dir: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(train_dir / "*.parquet"))):
        parts.append(pd.read_parquet(f, columns=RAW_COLS))
    return pd.concat(parts, ignore_index=True)


def _read_ranking(rank_path: Path) -> pd.DataFrame:
    return pd.read_parquet(rank_path, columns=RAW_COLS)


def _anchor_deltas(df: pd.DataFrame) -> pd.DataFrame:
    mvt = _norm_ts(df["MVT_TIME_UTC_mvt"])
    sched = _norm_ts(df["SCHED_TIME_UTC_mvt"])
    eobt1 = _norm_ts(df["EOBT_1_flt"])
    iobt = _norm_ts(df["IOBT_flt"])
    lobt = _norm_ts(df["LOBT_flt"])
    arvt1 = _norm_ts(df["ARVT_1_flt"])

    df["mvt_ts"] = mvt
    df["sched_ts"] = sched
    df["eobt1_ts"] = eobt1
    df["iobt_ts"] = iobt
    df["arvt1_ts"] = arvt1
    df["lobt_ts"] = lobt

    df["sd"] = (mvt - sched).dt.total_seconds()
    df["mvt_eobt1"] = (mvt - eobt1).dt.total_seconds()
    df["mvt_iobt"] = (mvt - iobt).dt.total_seconds()
    df["mvt_lobt"] = (mvt - lobt).dt.total_seconds()
    df["eobt1_minus_iobt"] = (eobt1 - iobt).dt.total_seconds()
    df["sd_minus_mvt_eobt1"] = df["sd"] - df["mvt_eobt1"]
    df["eobt1_equals_sched"] = ((eobt1 - sched).dt.total_seconds().abs() <= 1).astype("Int8")
    df["flt_null"] = df["AIRCRAFT_OPERATOR_flt"].isna().astype("Int8")

    df["hour"] = mvt.dt.hour.astype("Int16")
    df["dow"] = mvt.dt.dayofweek.astype("Int16")
    df["month"] = mvt.dt.month.astype("Int16")
    df["sd_mod_86400"] = df["sd"].mod(86400).where(df["sd"].notna())
    return df


def _grouped_window_stats(
    dep: pd.DataFrame,
    ctx_dep: pd.DataFrame,
    keys: tuple[str, ...],
    window_min: int,
    suffix: str,
) -> pd.DataFrame:
    """B1. For every dep row, aggregate mvt_eobt1 over ctx_dep rows sharing
    the group keys with mvt_ts in [row.mvt_ts - window, row.mvt_ts).
    Uses per-group cumulative sums for O(N log N)."""
    win_ns = np.int64(window_min) * _NS_MIN
    right = ctx_dep.dropna(subset=["mvt_eobt1"]).copy()
    right["mvt_ns"] = right["mvt_ts"].astype("int64")
    right = right.sort_values([*keys, "mvt_ns"]).reset_index(drop=True)

    med_col = f"nb_eobt_med{suffix}"
    mean_col = f"nb_eobt_mean{suffix}"
    cnt_col = f"nb_eobt_cnt{suffix}"

    med_arr = np.full(len(dep), np.nan)
    mean_arr = np.full(len(dep), np.nan)
    cnt_arr = np.full(len(dep), np.nan)

    left = dep[["MVT_ID_mvt", "mvt_ts", *keys]].copy()
    left["mvt_ns"] = left["mvt_ts"].astype("int64")
    left["_orig_idx"] = np.arange(len(left))
    left = left.sort_values([*keys, "mvt_ns"]).reset_index(drop=True)

    for k, right_g in right.groupby(list(keys) if len(keys) > 1 else keys[0], sort=False, dropna=True):
        r_ts = right_g["mvt_ns"].values
        r_val = right_g["mvt_eobt1"].values
        if len(keys) == 1:
            mask = (left[keys[0]].values == k)
        else:
            mask = np.ones(len(left), dtype=bool)
            for kn, kv in zip(keys, k):
                mask &= (left[kn].values == kv)
        left_ts = left.loc[mask, "mvt_ns"].values
        orig_idx = left.loc[mask, "_orig_idx"].values
        if len(left_ts) == 0:
            continue
        hi = np.searchsorted(r_ts, left_ts, side="left")
        lo = np.searchsorted(r_ts, left_ts - win_ns, side="left")
        n = hi - lo
        cumsum = np.concatenate(([0.0], np.cumsum(r_val)))
        s = cumsum[hi] - cumsum[lo]
        mean_g = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        # Median per row: use rolling reindex trick via Python loop only when n > 0
        med_g = np.full(len(left_ts), np.nan)
        for i in np.where(n > 0)[0]:
            med_g[i] = np.median(r_val[lo[i]:hi[i]])
        mean_arr[orig_idx] = mean_g
        med_arr[orig_idx] = med_g
        cnt_arr[orig_idx] = n.astype(float)
    return pd.DataFrame({
        "MVT_ID_mvt": dep["MVT_ID_mvt"].values,
        med_col: med_arr,
        mean_col: mean_arr,
        cnt_col: cnt_arr,
    })


def _order_backward(dep: pd.DataFrame, ctx_dep: pd.DataFrame,
                    window_min: int) -> pd.DataFrame:
    """B2. Same-airport backward window at 30 min; count with a later EOBT_1."""
    win_ns = np.int64(window_min) * _NS_MIN
    right = ctx_dep.dropna(subset=["eobt1_ts"]).copy()
    right["mvt_ns"] = right["mvt_ts"].astype("int64")
    right["eobt_ns"] = right["eobt1_ts"].astype("int64")
    right = right.sort_values(["ADEP_mvt", "mvt_ns"]).reset_index(drop=True)

    later = np.full(len(dep), np.nan)
    total = np.full(len(dep), np.nan)

    left = dep[["MVT_ID_mvt", "ADEP_mvt", "mvt_ts", "eobt1_ts"]].copy()
    left["mvt_ns"] = left["mvt_ts"].astype("int64")
    left["eobt_ns"] = left["eobt1_ts"].astype("int64")
    left["_orig_idx"] = np.arange(len(left))
    left = left.sort_values(["ADEP_mvt", "mvt_ns"]).reset_index(drop=True)

    for a, right_g in right.groupby("ADEP_mvt", sort=False):
        r_mvt = right_g["mvt_ns"].values
        r_eobt = right_g["eobt_ns"].values
        mask = (left["ADEP_mvt"].values == a)
        if not mask.any():
            continue
        left_mvt = left.loc[mask, "mvt_ns"].values
        left_eobt = left.loc[mask, "eobt_ns"].values
        orig_idx = left.loc[mask, "_orig_idx"].values
        hi = np.searchsorted(r_mvt, left_mvt, side="left")
        lo = np.searchsorted(r_mvt, left_mvt - win_ns, side="left")
        n = hi - lo
        later_g = np.zeros(len(left_mvt), dtype=np.float64)
        for i in np.where(n > 0)[0]:
            if left_eobt[i] != np.iinfo(np.int64).min:
                later_g[i] = int(np.sum(r_eobt[lo[i]:hi[i]] > left_eobt[i]))
            else:
                later_g[i] = np.nan
        total_g = n.astype(float)
        total_g[total_g == 0] = np.nan
        later[orig_idx] = later_g
        total[orig_idx] = total_g
    return pd.DataFrame({
        "MVT_ID_mvt": dep["MVT_ID_mvt"].values,
        "order_later_eobt_30": later,
        "order_total_30": total,
    })


def _stand_gap(dep: pd.DataFrame, ctx_all: pd.DataFrame) -> pd.DataFrame:
    """B3. Seconds since the last ARR BLOCK on the same stand at the same
    operational airport (ADES for ARR = the airport the flight landed at).
    Kept only if in (0, 1500) s."""
    arr = ctx_all[(ctx_all["PHASE_mvt"] == "ARR") &
                  ctx_all["BLOCK_TIME_UTC_mvt"].notna() &
                  ctx_all["STAND_mvt"].notna() &
                  ctx_all["ADES_mvt"].notna()].copy()
    arr["block_ts"] = _norm_ts(arr["BLOCK_TIME_UTC_mvt"])
    arr = arr.rename(columns={"ADES_mvt": "AIRPORT"})
    right = arr[["AIRPORT", "STAND_mvt", "block_ts"]].dropna(subset=["block_ts"])
    right = right.sort_values("block_ts").reset_index(drop=True)
    left = dep[["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "mvt_ts"]].copy()
    left = left.rename(columns={"ADEP_mvt": "AIRPORT"})
    left = left[left["STAND_mvt"].notna() & left["AIRPORT"].notna()]
    left = left.sort_values("mvt_ts").reset_index(drop=True)
    m = pd.merge_asof(
        left, right, left_on="mvt_ts", right_on="block_ts",
        by=["AIRPORT", "STAND_mvt"], direction="backward",
        allow_exact_matches=False,
    )
    gap = (m["mvt_ts"] - m["block_ts"]).dt.total_seconds()
    gap = gap.where((gap > 0) & (gap < 1500))
    return pd.DataFrame({"MVT_ID_mvt": m["MVT_ID_mvt"].values, "stand_gap": gap.values})


def _queue_between(dep: pd.DataFrame, ctx_all: pd.DataFrame) -> pd.DataFrame:
    """B4. Airport departures with MVT in (EOBT_1, MVT] of the row."""
    dep_ctx = ctx_all[(ctx_all["PHASE_mvt"] == "DEP") & ctx_all["ADEP_mvt"].notna()].copy()
    dep_ctx["mvt_ts"] = _norm_ts(dep_ctx["MVT_TIME_UTC_mvt"])
    dep_ctx = dep_ctx.dropna(subset=["mvt_ts"])
    dep_ctx["mvt_ns"] = dep_ctx["mvt_ts"].astype("int64")
    dep_ctx = dep_ctx.sort_values(["ADEP_mvt", "mvt_ns"]).reset_index(drop=True)

    out = np.full(len(dep), np.nan)
    dep_key = dep[["MVT_ID_mvt", "ADEP_mvt", "mvt_ts", "eobt1_ts"]].copy()
    dep_key["mvt_ns"] = dep_key["mvt_ts"].astype("int64")
    dep_key["eobt_ns"] = dep_key["eobt1_ts"].astype("int64")
    dep_key["_orig_idx"] = np.arange(len(dep_key))

    airports = dep_ctx["ADEP_mvt"].values
    # Per-airport searchsorted, then vectorised subtraction.
    for a in np.unique(dep_key["ADEP_mvt"].dropna().values):
        r = dep_ctx.loc[dep_ctx["ADEP_mvt"].values == a, "mvt_ns"].values
        mask = (dep_key["ADEP_mvt"].values == a)
        if not mask.any() or r.size == 0:
            continue
        left_hi = dep_key.loc[mask, "mvt_ns"].values
        left_lo = dep_key.loc[mask, "eobt_ns"].values
        orig_idx = dep_key.loc[mask, "_orig_idx"].values
        hi_idx = np.searchsorted(r, left_hi, side="right")
        lo_idx = np.searchsorted(r, left_lo, side="right")
        cnt = (hi_idx - lo_idx).astype(np.float64)
        cnt[left_lo == np.iinfo(np.int64).min] = np.nan
        out[orig_idx] = cnt
    return pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].values, "queue_eobt_mvt": out})


def _stand_prefix(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.extract(r"^([A-Za-z]+)")[0]
    return x.fillna("UNK")


def _flt_prefix(s: pd.Series) -> pd.Series:
    return s.astype("string").str[:3].fillna("UNK")


def compute_route_medians(dep: pd.DataFrame) -> pd.Series:
    """Median of MVT - ARVT_1 by route on clean rows, fit months only."""
    m = (dep["month"].isin(C.FIT_MONTHS) &
         dep["TAXITIME_SEC_mvt"].between(C.Y_MIN, C.Y_CLEAN_MAX) &
         (dep["TAXITIME_SEC_mvt"] - dep["sd"]).abs().gt(C.FB_TOL))
    plan_block = (dep["mvt_ts"] - dep["arvt1_ts"]).dt.total_seconds()
    route = dep["ADEP_mvt"].astype(str) + "_" + dep["ADES_mvt"].astype(str)
    return plan_block[m].groupby(route[m]).median()


def add_plan_residual(dep: pd.DataFrame, route_medians: pd.Series) -> pd.DataFrame:
    plan_block = (dep["mvt_ts"] - dep["arvt1_ts"]).dt.total_seconds()
    route = dep["ADEP_mvt"].astype(str) + "_" + dep["ADES_mvt"].astype(str)
    med = route.map(route_medians)
    dep["plan_block"] = plan_block
    dep["plan_taxi_res"] = (plan_block - med).clip(lower=-3600, upper=3600)
    return dep


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    icaos = pd.Series(df["ADEP_mvt"].unique()).dropna().tolist()
    frames = []
    for icao in icaos:
        p = C.METAR_DIR / f"{icao}.csv"
        if not p.exists():
            continue
        m = pd.read_csv(p, low_memory=False, na_values=["M", ""])
        m["ts"] = pd.to_datetime(m["valid"], utc=True, errors="coerce")
        for col in ("tmpf", "sknt", "gust", "vsby"):
            m[col] = pd.to_numeric(m[col], errors="coerce")
        m["tmpc"] = (m["tmpf"] - 32) * 5 / 9
        m["vis_km"] = m["vsby"] * 1.609344
        m["wind_kt"] = m["sknt"]
        m["gust_kt"] = m["gust"]
        m["ADEP_mvt"] = icao
        frames.append(m[["ADEP_mvt", "ts", "tmpc", "vis_km", "wind_kt", "gust_kt"]])
    if not frames:
        for c in ("tmpc", "vis_km", "wind_kt", "gust_kt"):
            df[c] = np.nan
        return df
    metar = pd.concat(frames, ignore_index=True).sort_values(["ADEP_mvt", "ts"])
    tol = pd.Timedelta(minutes=45)
    left = df[["MVT_ID_mvt", "ADEP_mvt", "mvt_ts"]].copy().sort_values(["ADEP_mvt", "mvt_ts"])
    out = []
    for a, g in left.groupby("ADEP_mvt", sort=False):
        r = metar[metar["ADEP_mvt"] == a][["ts", "tmpc", "vis_km", "wind_kt", "gust_kt"]].sort_values("ts")
        if r.empty:
            continue
        m = pd.merge_asof(g.sort_values("mvt_ts"), r,
                          left_on="mvt_ts", right_on="ts",
                          direction="nearest", tolerance=tol)
        out.append(m[["MVT_ID_mvt", "tmpc", "vis_km", "wind_kt", "gust_kt"]])
    if not out:
        for c in ("tmpc", "vis_km", "wind_kt", "gust_kt"):
            df[c] = np.nan
        return df
    w = pd.concat(out, ignore_index=True)
    return df.merge(w, on="MVT_ID_mvt", how="left")


def build_year(source: str, out_path: Path) -> pd.DataFrame:
    if out_path.exists():
        log.info("cache hit: %s", out_path)
        return pd.read_parquet(out_path)

    log.info("read %s", source)
    if source == "train":
        raw = _read_training(C.TRAIN_DIR)
    elif source == "rank":
        raw = _read_ranking(C.RANK)
    else:
        raise ValueError(source)

    raw["mvt_ts"] = _norm_ts(raw["MVT_TIME_UTC_mvt"])
    # Keep DEP rows at target airports and ARR rows that land at target
    # airports; the ARR rows are the right side of stand_gap (B3).
    keep = ((raw["PHASE_mvt"] == "DEP") & raw["ADEP_mvt"].isin(C.TARGETS)) | \
           ((raw["PHASE_mvt"] == "ARR") & raw["ADES_mvt"].isin(C.TARGETS))
    raw = raw[keep].copy()

    dep = raw[raw["PHASE_mvt"] == "DEP"].copy().reset_index(drop=True)
    dep = _anchor_deltas(dep)
    if source == "train":
        dep = dep[dep["TAXITIME_SEC_mvt"].astype("float64") > 0].copy().reset_index(drop=True)

    dep["flt_prefix"] = _flt_prefix(dep["FLIGHT_mvt"])
    dep["stand_prefix"] = _stand_prefix(dep["STAND_mvt"])

    ctx_dep = raw[raw["PHASE_mvt"] == "DEP"][
        ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "EOBT_1_flt"]].copy()
    ctx_dep["mvt_ts"] = _norm_ts(ctx_dep["MVT_TIME_UTC_mvt"])
    ctx_dep["eobt1_ts"] = _norm_ts(ctx_dep["EOBT_1_flt"])
    ctx_dep["mvt_eobt1"] = (ctx_dep["mvt_ts"] - ctx_dep["eobt1_ts"]).dt.total_seconds()

    log.info("neighbour tempo")
    for win, keys, suf in [
        (30, ("ADEP_mvt",), "_apt30"),
        (60, ("ADEP_mvt",), "_apt60"),
        (30, ("ADEP_mvt", "RUNWAY_mvt"), "_rwy30"),
    ]:
        dep = dep.merge(_grouped_window_stats(dep, ctx_dep, keys, win, suf),
                        on="MVT_ID_mvt", how="left")
    log.info("take-off order")
    dep = dep.merge(_order_backward(dep, ctx_dep, 30), on="MVT_ID_mvt", how="left")

    ctx_all = raw[["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
                   "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt",
                   "RUNWAY_mvt"]].copy()
    log.info("stand gap")
    dep = dep.merge(_stand_gap(dep, ctx_all), on="MVT_ID_mvt", how="left")
    log.info("queue between anchors")
    dep = dep.merge(_queue_between(dep, ctx_all), on="MVT_ID_mvt", how="left")

    log.info("weather")
    dep = add_weather(dep)

    for c in ("sched_ts", "eobt1_ts", "iobt_ts", "arvt1_ts", "lobt_ts"):
        if c in dep.columns:
            dep = dep.drop(columns=[c])

    dep.to_parquet(out_path)
    log.info("wrote %s (%d rows, %d cols)", out_path, len(dep), len(dep.columns))
    return dep


def load_or_build(source: str) -> pd.DataFrame:
    path = C.CACHE / f"frame_{source}.parquet"
    return build_year(source, path)
