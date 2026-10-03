"""WINNING_PLAN A2: ADS-B pushback clocks of the scored departure, keyed on `MVT_ID_mvt`.

The callsign links a departure to the ADSB.lol aircraft (`icao`) near its
`MVT`. The builder then reads every ground point of that aircraft, not
only the points that carry the callsign: the callsign often arrives after
the pushback. A gap above `ADSB_SEGMENT_GAP_S` ends the taxi segment,
walking back from the take-off. The columns give the stack model the
candidate clocks and their quality; the model learns which one to trust.
The 2025 training days and the 2026 ranking days go through the same code.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep

POINTS = C.ADSB_DIR / "points"
STANDS = C.ROOT / "external" / "osm" / "stands.parquet"
OUT_TRAIN = C.ADSB_DIR / "adsb_clock_train.parquet"
OUT_RANK = C.ADSB_DIR / "adsb_clock_rank.parquet"
M_PER_DEG = 111_320.0
log = logging.getLogger(__name__)


def stand_distance(p: pd.DataFrame) -> np.ndarray:
    """Metres from each point to the nearest OSM stand node of its airport."""
    st = pd.read_parquet(STANDS)
    out = np.full(len(p), np.nan)
    for apt, s in st.groupby("ADEP_mvt"):
        m = (p["apt"] == apt).values
        if m.any():
            k = np.cos(np.radians(s["lat"].mean()))
            tree = cKDTree(np.c_[s["lat"].values, s["lon"].values * k] * M_PER_DEG)
            out[m] = tree.query(np.c_[p.loc[m, "lat"].values, p.loc[m, "lon"].values * k] * M_PER_DEG)[0]
    return out


def own_stand(dep: pd.DataFrame) -> pd.DataFrame:
    """`dep` with the mean position of the OSM nodes of its `STAND_mvt`; NaN where OSM has no such ref."""
    def key(s: pd.Series) -> pd.Series:
        return s.astype(str).str.upper().map(lambda v: re.sub(r"[^A-Z0-9]", "", v))
    st = pd.read_parquet(STANDS)
    pos = st.assign(k=key(st["stand_ref"])).groupby(["ADEP_mvt", "k"])[["lat", "lon"]].mean()
    pos.columns = ["own_lat", "own_lon"]
    return dep.assign(k=key(dep["STAND_mvt"])).merge(pos, left_on=["ADEP_mvt", "k"], right_index=True,
                                                     how="left").drop(columns="k")


def metres(lat0, lon0, lat1, lon1) -> np.ndarray:
    return np.hypot(lat1 - lat0, (lon1 - lon0) * np.cos(np.radians(lat0))) * M_PER_DEG


def load_points(day: pd.Timestamp) -> pd.DataFrame:
    """Points of `day` and of the evening before; empty if neither day is on disk."""
    files = [POINTS / f"{d:%Y.%m.%d}.parquet" for d in (day - pd.Timedelta(days=1), day)]
    parts = [pd.read_parquet(f) for f in files if f.exists()]
    p = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return p[p["ts"] >= day - pd.Timedelta(seconds=C.ADSB_LOOKBACK_S)] if len(p) else p


def link(dep: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """One `icao` per departure: the callsign match with the most points near `MVT`."""
    m = dep[["MVT_ID_mvt", "ADEP_mvt", "CALLSIGN_flt", "mvt"]].merge(
        p[["apt", "flight", "icao", "ts"]], left_on=["ADEP_mvt", "CALLSIGN_flt"], right_on=["apt", "flight"])
    dt = (m["mvt"] - m["ts"]).dt.total_seconds()
    m = m[(dt >= -C.ADSB_AFTER_S) & (dt <= C.ADSB_LOOKBACK_S)]
    n = m.groupby(["MVT_ID_mvt", "icao"]).size().rename("n").reset_index()
    n = n.sort_values("n", ascending=False).drop_duplicates("MVT_ID_mvt")
    return n.drop(columns="n").merge(dep[["MVT_ID_mvt", "ADEP_mvt", "mvt", "own_lat", "own_lon"]], on="MVT_ID_mvt")


def segment(dep: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Ground points of each linked aircraft in its last taxi segment before `MVT`, time-sorted."""
    g = link(dep, p).merge(p[p["alt"] == C.ADSB_GROUND], left_on=["icao", "ADEP_mvt"], right_on=["icao", "apt"])
    g["dt"] = (g["mvt"] - g["ts"]).dt.total_seconds()
    g = g[(g["dt"] >= -C.ADSB_AFTER_S) & (g["dt"] <= C.ADSB_LOOKBACK_S)].sort_values(["MVT_ID_mvt", "dt"])
    gap = g.groupby("MVT_ID_mvt")["dt"].diff().fillna(g["dt"])  # the first gap runs from `MVT`
    g = g[(gap > C.ADSB_SEGMENT_GAP_S).groupby(g["MVT_ID_mvt"]).cumsum() == 0].copy()
    g["gap"] = gap.loc[g.index].clip(lower=0)
    return g.iloc[::-1].reset_index(drop=True)


def first_move_after(g: pd.DataFrame, stops: pd.DataFrame) -> pd.Series:
    """Per departure, `dt` of the first moving point after the last run in `stops`."""
    end = stops.groupby("MVT_ID_mvt")["end"].min().rename("stop_end")
    after = g[g["moving"]].merge(end, left_on="MVT_ID_mvt", right_index=True)
    return after[after["dt"] < after["stop_end"]].groupby("MVT_ID_mvt")["dt"].max()


def clock_columns(g: pd.DataFrame) -> pd.DataFrame:
    """Candidate clocks and quality columns per departure, in seconds before `MVT`."""
    g["dist"] = stand_distance(g)
    g["moving"] = g["gs"] >= C.ADSB_MOVING_KT
    g["run"] = ((g["moving"] != g["moving"].shift()) | (g["MVT_ID_mvt"] != g["MVT_ID_mvt"].shift())).cumsum()
    g["own"] = metres(g["own_lat"], g["own_lon"], g["lat"], g["lon"])
    runs = g[~g["moving"]].groupby("run").agg(MVT_ID_mvt=("MVT_ID_mvt", "first"), start=("dt", "max"),
                                              end=("dt", "min"), dist=("dist", "min"), own=("own", "min"))
    runs["dur"] = runs["start"] - runs["end"]
    by = g.groupby("MVT_ID_mvt", sort=False)
    first = by.first()
    out = pd.DataFrame({"adsb_seg": first["dt"], "adsb_gs0": first["gs"], "adsb_dist0": first["dist"],
                        "adsb_end": by["dt"].min(), "adsb_n": by.size(), "adsb_gap_max": by["gap"].max()})
    out["adsb_move"] = g[g["moving"]].groupby("MVT_ID_mvt")["dt"].max()
    gate = runs[(runs["dur"] >= C.ADSB_GATE_STOP_S) & (runs["dist"] <= C.ADSB_STAND_M)]
    out["adsb_push"] = first_move_after(g, gate)
    out["adsb_push_dur"] = gate.groupby("MVT_ID_mvt")["dur"].last()
    out["adsb_parked"] = first_move_after(g, runs[runs["dur"] >= C.ADSB_PARKED_S])
    near = g[g["own"] <= C.ADSB_OWN_M]
    own = g.groupby("MVT_ID_mvt", sort=False)["own"]
    out["adsb_own_d0"] = own.first()
    out["adsb_own_dmin"] = own.min()
    out["adsb_own_first"] = near.groupby("MVT_ID_mvt")["dt"].max()
    out["adsb_own_last"] = near.groupby("MVT_ID_mvt")["dt"].min()
    out["adsb_own_move"] = near[near["moving"]].groupby("MVT_ID_mvt")["dt"].max()
    out["adsb_own_parked"] = first_move_after(g, runs[(runs["dur"] >= C.ADSB_PARKED_S) & (runs["own"] <= C.ADSB_OWN_M)])
    top = runs.sort_values("dur", ascending=False).groupby("MVT_ID_mvt").head(2)
    top = top.assign(k=top.groupby("MVT_ID_mvt").cumcount() + 1).pivot(index="MVT_ID_mvt", columns="k",
                                                                      values=["end", "dur", "dist"])
    top.columns = [f"adsb_stop{k}_{c}" for c, k in top.columns]
    return out.join(top).reset_index()


def build(dep: pd.DataFrame) -> pd.DataFrame:
    """ADS-B columns for every departure; `adsb_day_loaded` is 0 where the day is not on disk."""
    dep = own_stand(dep.assign(mvt=dep["MVT_TIME_UTC_mvt"].astype("datetime64[us, UTC]"),
                               CALLSIGN_flt=dep["CALLSIGN_flt"].str.strip()))
    day = dep["mvt"].dt.floor("D")
    parts = []
    for d, rows in dep.groupby(day):
        loaded = (POINTS / f"{d:%Y.%m.%d}.parquet").exists()
        f = clock_columns(segment(rows, load_points(d))) if loaded else pd.DataFrame(columns=["MVT_ID_mvt"])
        parts.append(rows[["MVT_ID_mvt"]].merge(f, on="MVT_ID_mvt", how="left").assign(adsb_day_loaded=int(loaded)))
    out = pd.concat(parts, ignore_index=True)
    on = out["adsb_day_loaded"] == 1
    log.info("%d rows, %d on loaded days, clock share there %.3f", len(out), on.sum(), out.loc[on, "adsb_seg"].notna().mean())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    extra = ["CALLSIGN_flt", "STAND_mvt"]
    build(load_dep(sorted(C.TRAIN_DIR.glob("training_*.parquet")), extra)).to_parquet(OUT_TRAIN)
    build(load_dep([C.RANK], extra)).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
