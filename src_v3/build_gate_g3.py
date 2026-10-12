"""Column table of the LIRF gate g3 (v104): the G1 arrival shares, the v83 arrival link and the operator shares.

`art_op_day` and `art_op_month` are the share of the operator's arrivals with in-block within `FB_TOL_DEFAULT` of
the schedule, per day and per month, smoothed toward the airport share. Only arrival labels enter. Run
`build_arr_artefact` and `build_arr_link_v83` first; this script joins their tables.
"""
from __future__ import annotations

import logging

import pandas as pd

from src_v3 import config as C
from src_v3.build_arr_artefact import OUT_RANK as G1_RANK
from src_v3.build_arr_artefact import OUT_TRAIN as G1_TRAIN
from src_v3.build_arr_link_v83 import OUT_RANK as LINK_RANK
from src_v3.build_arr_link_v83 import OUT_TRAIN as LINK_TRAIN

OUT_TRAIN = C.ROOT / "models" / "arr_art_op_link_train_g3.parquet"
OUT_RANK = C.ROOT / "models" / "arr_art_op_link_rank_g3.parquet"
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
log = logging.getLogger(__name__)


def op_shares(paths: list) -> pd.DataFrame:
    """`art_op_day` and `art_op_month` of every departure in `paths`, keyed on `MVT_ID_mvt`."""
    m = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    m["apt"] = m["ADEP_mvt"].where(m["PHASE_mvt"] == "DEP", m["ADES_mvt"])
    m["op"] = m["AIRCRAFT_OPERATOR_flt"].astype(str)
    m["day"], m["month"] = m["MVT_TIME_UTC_mvt"].dt.floor("D"), m["MVT_TIME_UTC_mvt"].dt.month
    arr = m[(m["PHASE_mvt"] == "ARR") & m["BLOCK_TIME_UTC_mvt"].notna()].copy()
    arr["art"] = (arr["BLOCK_TIME_UTC_mvt"] - arr["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs() <= C.FB_TOL_DEFAULT
    dep = m.loc[m["PHASE_mvt"] == "DEP", ["MVT_ID_mvt", "apt", "op", "day", "month"]]
    k = C.ARTEFACT_SMOOTH_K
    for per, name in (("day", "art_op_day"), ("month", "art_op_month")):
        prior = arr.groupby(["apt", per])["art"].mean().rename("p").reset_index()
        g = arr.groupby(["apt", "op", per])["art"].agg(["sum", "size"]).reset_index().merge(prior, on=["apt", per])
        g[name] = (g["sum"] + k * g["p"]) / (g["size"] + k)
        dep = dep.merge(g[["apt", "op", per, name]], on=["apt", "op", per], how="left") \
                 .merge(prior, on=["apt", per], how="left")
        dep[name] = dep[name].fillna(dep.pop("p"))
    return dep[["MVT_ID_mvt", "art_op_day", "art_op_month"]]


def table(g1, link, paths: list) -> pd.DataFrame:
    return pd.read_parquet(g1).merge(pd.read_parquet(link), on="MVT_ID_mvt", how="left", validate="1:1") \
                              .merge(op_shares(paths), on="MVT_ID_mvt", how="left", validate="1:1")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for g1, link, paths, out in ((G1_TRAIN, LINK_TRAIN, sorted(C.TRAIN_DIR.glob("training_*.parquet")), OUT_TRAIN),
                                 (G1_RANK, LINK_RANK, [C.RANK], OUT_RANK)):
        t = table(g1, link, paths)
        t.to_parquet(out)
        log.info("%s: %d rows, %d columns", out.name, len(t), t.shape[1])


if __name__ == "__main__":
    main()
