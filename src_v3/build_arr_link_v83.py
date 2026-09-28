"""Arrival artefact link (v83): the schedule-grid state of the arrivals around each departure.

At LIRF some handlers report the scheduled time as the block time, on
arrivals and on departures alike. When the previous arrival at the
departure's stand (usually the same aircraft's inbound) blocked in within
60 s of its schedule, 36 % of 2025 LIRF departures are fallback rows,
against 18 % otherwise. The columns: that arrival's grid flag, its
in-block minus schedule, its age at the departure, and the share of the
airport's arrivals on the grid that day (L7). Arrival block times are in
the 2025 training files and the 2026 ranking file.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_stand_seq_v82 import seconds

OUT_TRAIN = C.ROOT / "models" / "arr_link_train_v83.parquet"
OUT_RANK = C.ROOT / "models" / "arr_link_rank_v83.parquet"
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]
FEATURES = ["prv_arr_grid", "prv_arr_off_sched", "prv_arr_age", "day_arr_grid_share"]
GRID_TOL_S, MAX_AGE_S, NO_AOBT_LOOKBACK_S = 60, 12 * 3600, 1200
log = logging.getLogger(__name__)


def arr_link(paths: list) -> pd.DataFrame:
    """The four link columns of every departure in `paths`, keyed on `MVT_ID_mvt`."""
    m = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    arr = m[(m["PHASE_mvt"] == "ARR") & m["BLOCK_TIME_UTC_mvt"].notna()].copy()
    arr["off_sched"] = (arr["BLOCK_TIME_UTC_mvt"] - arr["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    arr["grid"] = (arr["off_sched"].abs() < GRID_TOL_S).astype(float)
    arr["tb"], arr["day"] = seconds(arr["BLOCK_TIME_UTC_mvt"]), arr["BLOCK_TIME_UTC_mvt"].dt.date
    share = arr.groupby(["ADES_mvt", "day"])["grid"].mean().rename("day_arr_grid_share")
    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["q"] = seconds(dep["AOBT_3_flt"]).fillna(seconds(dep["MVT_TIME_UTC_mvt"]) - NO_AOBT_LOOKBACK_S)
    dep["day"] = dep["MVT_TIME_UTC_mvt"].dt.date
    dep = dep.merge(share, left_on=["ADEP_mvt", "day"], right_index=True, how="left")
    dep["key"] = dep["ADEP_mvt"] + "|" + dep["STAND_mvt"].astype(str)
    arr["key"] = arr["ADES_mvt"] + "|" + arr["STAND_mvt"].astype(str)
    ok = dep["STAND_mvt"].notna() & dep["q"].notna()
    j = pd.merge_asof(dep[ok].sort_values("q"), arr.loc[arr["STAND_mvt"].notna(), ["key", "tb", "grid", "off_sched"]]
                      .sort_values("tb"), left_on="q", right_on="tb", by="key", direction="backward")
    recent = (j["q"] - j["tb"]) < MAX_AGE_S
    out = pd.DataFrame({"MVT_ID_mvt": j["MVT_ID_mvt"].values,
                        "prv_arr_grid": j["grid"].where(recent).values,
                        "prv_arr_off_sched": j["off_sched"].where(recent).clip(-C.NM_CLOCK_CLIP_S, C.NM_CLOCK_CLIP_S).values,
                        "prv_arr_age": (j["q"] - j["tb"]).where(recent).values})
    out = dep[["MVT_ID_mvt", "day_arr_grid_share"]].merge(out, on="MVT_ID_mvt", how="left", validate="1:1")
    log.info("%d departures; prv_arr_grid coverage %.3f, mean %.3f", len(out),
             out["prv_arr_grid"].notna().mean(), out["prv_arr_grid"].mean())
    return out[["MVT_ID_mvt", *FEATURES]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arr_link(sorted(C.TRAIN_DIR.glob("training_*.parquet"))).to_parquet(OUT_TRAIN)
    arr_link([C.RANK]).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
