"""Structure terms of the record for the neural members (twentieth and twenty-first passes).

Four columns that describe how a departure row was recorded, not the flight itself:

- `mvt_sec_dist`: seconds from `MVT` to the nearest whole minute. At seven airports every time is a whole minute
  plus organiser noise of at most `MVT_GRID_JITTER_S`; a larger value marks a take-off time from another source (P1).
- `aobt_default_share`: the 2025 share of the airport's departures whose `MVT - AOBT_3`, rounded to the minute,
  equals this row's. A high share marks an NM default taxi time rather than a reported off-block (F20.4).
- `dup_prev_gap`, `dup_next_gap`: seconds from the previous and to the next departure of the same flight number at
  the same airport, within `DUP_WINDOW_S`. A second departure within 1 h has a long label (F21.3).

The 2025 training files and the ranking file go through the same code; the default shares come from 2025 only.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep

OUT_TRAIN = C.ROOT / "models" / "struct_terms_train.parquet"
OUT_RANK = C.ROOT / "models" / "struct_terms_rank.parquet"
log = logging.getLogger(__name__)


def default_shares(dep: pd.DataFrame) -> pd.Series:
    """Share of each airport's departures per `MVT - AOBT_3` rounded to the minute, indexed by (airport, minutes)."""
    key = pd.DataFrame({"apt": dep["ADEP_mvt"].values, "min": minutes(dep)}).dropna()
    return key.groupby(["apt", "min"]).size() / key.groupby("apt").size()


def minutes(dep: pd.DataFrame) -> np.ndarray:
    return np.round((dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds().values / 60)


def dup_gaps(dep: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Seconds from the previous and to the next departure of the same flight number at the same airport."""
    d = dep[["ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt"]].copy()
    d["i"] = np.arange(len(d))
    d = d.dropna(subset=["FLIGHT_mvt"]).sort_values(["ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt"])
    g = d.groupby(["ADEP_mvt", "FLIGHT_mvt"])["MVT_TIME_UTC_mvt"]
    prev, nxt = np.full(len(dep), np.nan), np.full(len(dep), np.nan)
    prev[d["i"].values] = (d["MVT_TIME_UTC_mvt"] - g.shift()).dt.total_seconds().values
    nxt[d["i"].values] = (g.shift(-1) - d["MVT_TIME_UTC_mvt"]).dt.total_seconds().values
    return (np.where(prev <= C.DUP_WINDOW_S, prev, np.nan), np.where(nxt <= C.DUP_WINDOW_S, nxt, np.nan))


def terms(dep: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """The four structure columns of every departure in `dep`, keyed on `MVT_ID_mvt`."""
    s = dep["MVT_TIME_UTC_mvt"].dt.second.values
    idx = pd.MultiIndex.from_arrays([dep["ADEP_mvt"].values, minutes(dep)])
    prev, nxt = dup_gaps(dep)
    return pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].values, "mvt_sec_dist": np.minimum(s, 60 - s).astype(float),
                         "aobt_default_share": shares.reindex(idx).values, "dup_prev_gap": prev, "dup_next_gap": nxt})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    extra = ["FLIGHT_mvt"]
    train = load_dep(sorted(C.TRAIN_DIR.glob("training_*.parquet")), extra)
    shares = default_shares(train)
    for dep, out in ((train, OUT_TRAIN), (load_dep([C.RANK], extra), OUT_RANK)):
        t = terms(dep, shares)
        t.to_parquet(out)
        log.info("%s: %d rows; mvt_sec_dist > %d s on %d rows; default share > 0.2 on %.3f; second departure within 1 h on %d",
                 out.name, len(t), C.MVT_GRID_JITTER_S, int((t["mvt_sec_dist"] > C.MVT_GRID_JITTER_S).sum()),
                 float((t["aobt_default_share"] > 0.2).mean()), int((t["dup_prev_gap"] <= 3600).sum()))


if __name__ == "__main__":
    main()
