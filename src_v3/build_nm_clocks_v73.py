"""WINNING_PLAN L1 (track B): NM clocks of the scored flight.

`AOBT_3_flt` is NM's off-block time of the flown trajectory, at minute
resolution. `MVT - AOBT_3` is a noisy taxi-out clock: on 2025 departures
its median error against the label is 175 s, and 21 % of rows lie within
60 s. The trees learn its bias and slope per airport. `LOBT_flt` is the
last known off-block time. The 2025 training files and the 2026 ranking
file go through the same code.
"""
from __future__ import annotations

import logging

import pandas as pd

from src_v3 import config as C

OUT_TRAIN = C.ROOT / "models" / "nm_clocks_train_v73.parquet"
OUT_RANK = C.ROOT / "models" / "nm_clocks_rank_v73.parquet"
TIMES = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt"]
FEATURES = ["mvt_aobt3", "aobt3_eobt1", "aobt3_sched", "mvt_lobt", "lobt_aobt3"]
CLIPPED = ("mvt_aobt3", "mvt_lobt", "lobt_aobt3")
log = logging.getLogger(__name__)


def load_dep(paths: list, extra: list[str] = ()) -> pd.DataFrame:
    """Departure rows with their id, airport, runway, the NM times and the `extra` columns."""
    f = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                               "RUNWAY_mvt", *TIMES, *extra]) for p in paths],
                  ignore_index=True)
    f = f[f["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    if not f["MVT_ID_mvt"].is_unique:
        raise ValueError("MVT_ID_mvt is not unique on the departure rows. "
                         "Check the input files for duplicates, then rerun.")
    return f


def clocks(dep: pd.DataFrame) -> pd.DataFrame:
    """The five clock columns in seconds, keyed on `MVT_ID_mvt`."""
    def sec(a: str, b: str) -> pd.Series:
        return (dep[a] - dep[b]).dt.total_seconds()
    out = pd.DataFrame({
        "MVT_ID_mvt": dep["MVT_ID_mvt"].values,
        "mvt_aobt3": sec("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
        "aobt3_eobt1": sec("AOBT_3_flt", "EOBT_1_flt"),
        "aobt3_sched": sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt"),
        "mvt_lobt": sec("MVT_TIME_UTC_mvt", "LOBT_flt"),
        "lobt_aobt3": sec("LOBT_flt", "AOBT_3_flt"),
    })
    for c in CLIPPED:
        out[c] = out[c].clip(-C.NM_CLOCK_CLIP_S, C.NM_CLOCK_CLIP_S)
    log.info("coverage %.4f on %d rows; mvt_aobt3 median %.0f s",
             out["mvt_aobt3"].notna().mean(), len(out), out["mvt_aobt3"].median())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    clocks(load_dep(sorted(C.TRAIN_DIR.glob("training_*.parquet")))).to_parquet(OUT_TRAIN)
    clocks(load_dep([C.RANK])).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
