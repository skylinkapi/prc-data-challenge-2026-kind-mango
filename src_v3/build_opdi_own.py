"""WINNING_PLAN A1: OPDI pushback clocks of the scored departure, keyed on `MVT_ID_mvt`.

The link is the airport, the callsign (`CALLSIGN_flt` equals the OPDI
`osn_flight_id`) and the OPDI runway entry closest to `MVT` inside
`[MVT - OPDI_RWY_BEFORE_S, MVT + OPDI_RWY_AFTER_S]`. One OPDI flight serves
one departure. Three candidate clocks come from the OPDI stand exits
before the runway entry: the first one, the last one, and the last one
at the stand that `STAND_mvt` names.
"""
from __future__ import annotations

import logging

import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep

EVENTS = C.ROOT / "external" / "opdi" / "events_all.parquet"
OUT_TRAIN = C.ROOT / "models" / "opdi_own_train_v90.parquet"
OUT_RANK = C.ROOT / "models" / "opdi_own_rank_v90.parquet"
log = logging.getLogger(__name__)


def load_events() -> pd.DataFrame:
    e = pd.read_parquet(EVENTS, columns=["flight_id", "type", "event_time", "osm_airport", "osn_flight_id", "osm_ref"])
    e = e[e["type"].isin(["entry-runway", "exit-parking_position"])]
    return e.assign(event_time=e["event_time"].astype("datetime64[us, UTC]"))


def link(dep: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    rw = e.loc[e["type"] == "entry-runway", ["flight_id", "event_time", "osm_airport", "osn_flight_id"]]
    m = dep.merge(rw.rename(columns={"event_time": "rwy"}), left_on=["ADEP_mvt", "CALLSIGN_flt"],
                  right_on=["osm_airport", "osn_flight_id"])
    m["opdi_dt"] = (m["mvt"] - m["rwy"]).dt.total_seconds()
    m = m[m["opdi_dt"].between(-C.OPDI_RWY_AFTER_S, C.OPDI_RWY_BEFORE_S)].sort_values("opdi_dt")
    return m.drop_duplicates("MVT_ID_mvt").drop_duplicates(["flight_id", "rwy"])


def clocks(dep: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    """Candidate clocks in seconds before `MVT`, one row per departure."""
    dep = dep.assign(mvt=dep["MVT_TIME_UTC_mvt"].astype("datetime64[us, UTC]"))
    m = link(dep, e)
    pk = e.loc[e["type"] == "exit-parking_position", ["flight_id", "event_time", "osm_ref"]]
    c = m[["MVT_ID_mvt", "flight_id", "rwy", "mvt", "STAND_mvt"]].merge(pk, on="flight_id")
    c = c[(c["event_time"] < c["rwy"]) & (c["event_time"] > c["rwy"] - pd.Timedelta(seconds=C.OPDI_PUSH_MAX_S))]
    c = c.assign(sec=(c["mvt"] - c["event_time"]).dt.total_seconds()).sort_values("sec")
    at_stand = [str(r).upper().endswith(str(s).upper()) for r, s in zip(c["osm_ref"], c["STAND_mvt"])]
    out = dep[["MVT_ID_mvt"]].merge(m[["MVT_ID_mvt", "opdi_dt"]], on="MVT_ID_mvt", how="left")
    for name, rows, keep in (("opdi_first", c, "last"), ("opdi_last", c, "first"), ("opdi_stand", c[at_stand], "first")):
        out = out.merge(rows.drop_duplicates("MVT_ID_mvt", keep=keep)[["MVT_ID_mvt", "sec"]].rename(columns={"sec": name}),
                        on="MVT_ID_mvt", how="left")
    log.info("%d rows, OPDI clock share %.3f", len(out), out["opdi_first"].notna().mean())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    e = load_events()
    extra = ["CALLSIGN_flt", "STAND_mvt"]
    clocks(load_dep(sorted(C.TRAIN_DIR.glob("training_*.parquet")), extra), e).to_parquet(OUT_TRAIN)
    clocks(load_dep([C.RANK], extra), e).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
