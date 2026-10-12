"""WINNING_PLAN A2: ADS-B pushback clocks of the scored departure, keyed on `MVT_ID_mvt`.

The callsign links a departure to the ADSB.lol aircraft (`icao`) near its
`MVT`. The builder then reads every ground point of that aircraft, not
only the points that carry the callsign: the callsign often arrives after
the pushback. A gap above `ADSB_SEGMENT_GAP_S` ends the taxi segment,
walking back from the take-off. The columns give the stack model the
candidate clocks and their quality; the model learns which one to trust.
The 2025 training days and the 2026 ranking days go through the same code.
A departure without an NM record has no callsign. With `--link-flight` it links by `FLIGHT_mvt`; an IATA flight
number takes the ICAO airline code that the record rows pair with it (`LY358` is `ELY358`). With `--link-takeoff`,
a departure that no callsign links takes the aircraft of its type that lifts off at `MVT` plus the airport offset;
many airlines broadcast a callsign that differs from the flight number (`EZY12AB`). With `--keep DIR`, rows with
an NM callsign take the columns of the tables in DIR, so only the newly linked rows change; `--keep-linked` also
keeps the rows that have a segment in DIR (v130: the flight-number links of v127 stay).
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

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
IATA_FLIGHT = r"^([A-Z0-9]{2})(\d{1,4}[A-Z]?)$"   # airline code and number of an IATA flight number
ICAO_FLIGHT = r"^([A-Z]{3})(\d{1,4}[A-Z]?)$"      # the same for an ICAO callsign
log = logging.getLogger(__name__)


def stand_distance(p: pd.DataFrame, stands=STANDS) -> np.ndarray:
    """Metres from each point to the nearest stand node of its airport."""
    st = pd.read_parquet(stands)
    out = np.full(len(p), np.nan)
    for apt, s in st.groupby("ADEP_mvt"):
        m = (p["apt"] == apt).values
        if m.any():
            k = np.cos(np.radians(s["lat"].mean()))
            tree = cKDTree(np.c_[s["lat"].values, s["lon"].values * k] * M_PER_DEG)
            out[m] = tree.query(np.c_[p.loc[m, "lat"].values, p.loc[m, "lon"].values * k] * M_PER_DEG)[0]
    return out


def own_stand(dep: pd.DataFrame, stands=STANDS) -> pd.DataFrame:
    """`dep` with the mean position of the stand nodes of its `STAND_mvt`; NaN where the stand table has no such ref."""
    def key(s: pd.Series) -> pd.Series:
        return s.astype(str).str.upper().map(lambda v: re.sub(r"[^A-Z0-9]", "", v))
    st = pd.read_parquet(stands)
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


def liftoffs(p: pd.DataFrame) -> pd.DataFrame:
    """Lift-off events: the first airborne point of an aircraft within `ADSB_LIFT_GAP_S` of a ground point."""
    p = p.sort_values(["icao", "ts"])
    after_ground = ((p["icao"].shift() == p["icao"]) & (p["alt"].shift() == C.ADSB_GROUND)
                    & ((p["ts"] - p["ts"].shift()).dt.total_seconds() <= C.ADSB_LIFT_GAP_S))
    return p.loc[(p["alt"] != C.ADSB_GROUND) & after_ground, ["icao", "apt", "ts", "type"]]


def link_liftoff(dep: pd.DataFrame, p: pd.DataFrame, linked: pd.DataFrame) -> pd.DataFrame:
    """One `icao` per departure that `linked` misses: the closest lift-off of its aircraft type within
    `ADSB_LIFT_TOL_S` of `MVT` plus the airport offset, skipping lift-offs of aircraft that a callsign links."""
    todo = dep.loc[~dep["MVT_ID_mvt"].isin(linked["MVT_ID_mvt"]), ["MVT_ID_mvt", "ADEP_mvt", "AIRCRAFT_TYPE_mvt", "mvt"]]
    m = todo.merge(liftoffs(p), left_on=["ADEP_mvt", "AIRCRAFT_TYPE_mvt"], right_on=["apt", "type"])
    m["r"] = ((m["ts"] - m["mvt"]).dt.total_seconds() - m["ADEP_mvt"].map(C.ADSB_LIFT_OFFSET_S)).abs()
    m = m[m["r"] <= C.ADSB_LIFT_TOL_S]
    taken = linked[["icao", "ADEP_mvt", "mvt"]].merge(m[["icao", "ADEP_mvt", "ts"]], on=["icao", "ADEP_mvt"])
    taken = taken.loc[(taken["ts"] - taken["mvt"]).dt.total_seconds().abs() <= C.ADSB_AFTER_S, ["icao", "ts"]]
    m = m.merge(taken.drop_duplicates().assign(taken=True), on=["icao", "ts"], how="left")
    m = m[m["taken"].isna()].sort_values("r").drop_duplicates("MVT_ID_mvt").drop_duplicates(["icao", "ts"])
    return m[["MVT_ID_mvt", "icao"]].merge(dep[["MVT_ID_mvt", "ADEP_mvt", "mvt", "own_lat", "own_lon"]], on="MVT_ID_mvt")


def segment(dep: pd.DataFrame, p: pd.DataFrame, use_liftoff: bool = False) -> pd.DataFrame:
    """Ground points of each linked aircraft in its last taxi segment before `MVT`, time-sorted."""
    linked = link(dep, p)
    if use_liftoff:
        linked = pd.concat([linked, link_liftoff(dep, p, linked)], ignore_index=True)
    g = linked.merge(p[p["alt"] == C.ADSB_GROUND], left_on=["icao", "ADEP_mvt"], right_on=["icao", "apt"])
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


def clock_columns(g: pd.DataFrame, stands=STANDS) -> pd.DataFrame:
    """Candidate clocks and quality columns per departure, in seconds before `MVT`."""
    g["dist"] = stand_distance(g, stands)
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


def airline_map(dep: pd.DataFrame) -> dict:
    """IATA airline code of `FLIGHT_mvt` to the ICAO code of `CALLSIGN_flt`, from record rows with the same number."""
    f = dep["FLIGHT_mvt"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.extract(IATA_FLIGHT)
    c = dep["CALLSIGN_flt"].str.upper().str.strip().str.extract(ICAO_FLIGHT)
    pairs = pd.DataFrame({"iata": f[0], "icao": c[0]})[f[0].notna() & c[0].notna() & (f[1] == c[1])]
    n = pairs.groupby(["iata", "icao"]).size().sort_values(ascending=False)
    top = n.groupby(level="iata").head(1).reset_index(level="icao")
    top["purity"] = top[0] / n.groupby(level="iata").sum()
    top = top[(top[0] >= C.ADSB_IATA_MIN_PAIRS) & (top["purity"] >= C.ADSB_IATA_PURITY)]
    log.info("airline map: %d IATA codes", len(top))
    return top["icao"].to_dict()


def callsigns(dep: pd.DataFrame, amap: dict) -> pd.Series:
    """`CALLSIGN_flt`; where it is missing, `FLIGHT_mvt` with an IATA airline code replaced through `amap`."""
    flight = dep["FLIGHT_mvt"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    parts = flight.str.extract(IATA_FLIGHT)
    icao = parts[0].map(amap)
    fallback = flight.where(icao.isna(), icao + parts[1])
    return dep["CALLSIGN_flt"].str.strip().fillna(fallback)


def build(dep: pd.DataFrame, stands=STANDS, use_liftoff: bool = False) -> pd.DataFrame:
    """ADS-B columns for every departure; `adsb_day_loaded` is 0 where the day is not on disk."""
    dep = own_stand(dep.assign(mvt=dep["MVT_TIME_UTC_mvt"].astype("datetime64[us, UTC]"),
                               CALLSIGN_flt=dep["CALLSIGN_flt"].str.strip()), stands)
    day = dep["mvt"].dt.floor("D")
    parts = []
    for d, rows in dep.groupby(day):
        loaded = (POINTS / f"{d:%Y.%m.%d}.parquet").exists()
        f = (clock_columns(segment(rows, load_points(d), use_liftoff), stands) if loaded
             else pd.DataFrame(columns=["MVT_ID_mvt"]))
        parts.append(rows[["MVT_ID_mvt"]].merge(f, on="MVT_ID_mvt", how="left").assign(adsb_day_loaded=int(loaded)))
    out = pd.concat(parts, ignore_index=True)
    on = out["adsb_day_loaded"] == 1
    log.info("%d rows, %d on loaded days, clock share there %.3f", len(out), on.sum(), out.loc[on, "adsb_seg"].notna().mean())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stands", type=Path, default=STANDS, help="stand table; v115: OSM plus X-Plane (`build_xplane_stands`).")
    ap.add_argument("--out-dir", type=Path, default=C.ADSB_DIR, help="folder of the two clock tables.")
    ap.add_argument("--link-flight", action="store_true", help="link rows without an NM callsign by `FLIGHT_mvt`.")
    ap.add_argument("--link-takeoff", action="store_true", help="link rows that no callsign links by their lift-off (v130).")
    ap.add_argument("--keep", type=Path, help="folder of clock tables whose rows with an NM callsign are kept unchanged.")
    ap.add_argument("--keep-linked", action="store_true", help="with --keep: also keep the rows with a segment in that folder.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    extra = ["CALLSIGN_flt", "STAND_mvt", "FLIGHT_mvt", "AIRCRAFT_TYPE_mvt"]
    train, rank = load_dep(sorted(C.TRAIN_DIR.glob("training_*.parquet")), extra), load_dep([C.RANK], extra)
    amap = airline_map(train) if args.link_flight else {}
    for dep, out in ((train, OUT_TRAIN), (rank, OUT_RANK)):
        has_callsign = dep["CALLSIGN_flt"].notna()
        table = build(dep.assign(CALLSIGN_flt=callsigns(dep, amap)) if args.link_flight else dep, args.stands,
                      args.link_takeoff)
        if args.keep:
            kept = pd.read_parquet(args.keep / out.name)
            ids = dep.loc[has_callsign, "MVT_ID_mvt"]
            if args.keep_linked:
                ids = pd.concat([ids, kept.loc[kept["adsb_seg"].notna(), "MVT_ID_mvt"]])
            table = pd.concat([kept[kept["MVT_ID_mvt"].isin(ids)], table[~table["MVT_ID_mvt"].isin(ids)]], ignore_index=True)
        table.to_parquet(args.out_dir / out.name)
        log.info("%s: segment on %d rows without an NM callsign", out.name,
                 int(table.loc[table["MVT_ID_mvt"].isin(dep.loc[~has_callsign, "MVT_ID_mvt"]), "adsb_seg"].notna().sum()))


if __name__ == "__main__":
    main()
