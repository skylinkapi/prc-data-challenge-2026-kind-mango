"""WINNING_PLAN second edition A4: pushback clocks of the other departures at the airport.

For each departure, the mean and the count of the other departures'
pushback taxi (`MVT` minus the best OPDI or ADS-B off-block) with a
take-off in `[t - W, t)` or in `(t, t + W]`, the windows of v85. A second
mean holds the pushback taxi minus the NM clock `MVT - AOBT_3`: the
current bias of the NM clock at the airport. Rows on days without ADS-B
data see OPDI neighbours only. The stack reads these columns next to the
row's own clocks.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_adsb_clock import OUT_RANK as ADSB_RANK, OUT_TRAIN as ADSB_TRAIN
from src_v3.build_nm_clocks_v73 import OUT_RANK as NM_RANK, OUT_TRAIN as NM_TRAIN, load_dep
from src_v3.build_nm_neighbours_v73 import window_stats
from src_v3.build_opdi_own import OUT_RANK as OPDI_RANK, OUT_TRAIN as OPDI_TRAIN

OUT_TRAIN = C.ADSB_DIR / "pushback_neighbours_train.parquet"
OUT_RANK = C.ADSB_DIR / "pushback_neighbours_rank.parquet"
log = logging.getLogger(__name__)


def best_clock(f: pd.DataFrame) -> pd.Series:
    """The first clock present in `C.PUSH_NB_CLOCKS`, in seconds before `MVT`, kept in 0 to 7,200 s."""
    v = pd.concat([f[c] for c in C.PUSH_NB_CLOCKS if c in f], axis=1).bfill(axis=1).iloc[:, 0]
    return v.where(v.between(0, C.NM_CLOCK_CLIP_S))


def neighbours(dep: pd.DataFrame, paths: tuple) -> pd.DataFrame:
    """The neighbour columns keyed on `MVT_ID_mvt`; negated times turn `[s - w, s)` into `(t, t + w]`."""
    for p in paths:
        dep = dep.merge(pd.read_parquet(p), on="MVT_ID_mvt", how="left", validate="1:1")
    d = dep.assign(t=(dep["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1),
                   v=best_clock(dep)).dropna(subset=["t"])
    d["dv"] = d["v"] - d["mvt_aobt3"]
    out = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].values}, index=d.index)
    for _, grp in d.groupby("ADEP_mvt", sort=False):
        grp = grp.sort_values("t", kind="stable")
        for name, sign, windows in (("bk", 1, C.PUSH_NB_BACK_S), ("fw", -1, C.NM_FORWARD_WINDOWS_S)):
            order = slice(None) if sign == 1 else slice(None, None, -1)
            g = grp.iloc[order]
            s = sign * g["t"].values.astype(np.int64)
            for w in windows:
                mean, n = window_stats(s, g["v"].values.astype(float), w)
                out.loc[g.index, f"pnb_{name}{w // 60}_mean"] = mean
                out.loc[g.index, f"pnb_{name}{w // 60}_n"] = n
                out.loc[g.index, f"pnb_{name}{w // 60}_dnm"] = window_stats(s, g["dv"].values.astype(float), w)[0]
    log.info("%d rows; share with a backward neighbour %.3f", len(out), (out[f"pnb_bk{C.PUSH_NB_BACK_S[0] // 60}_n"] > 0).mean())
    return out.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    neighbours(load_dep(sorted(C.TRAIN_DIR.glob("training_*.parquet"))), (NM_TRAIN, OPDI_TRAIN, ADSB_TRAIN)).to_parquet(OUT_TRAIN)
    neighbours(load_dep([C.RANK]), (NM_RANK, OPDI_RANK, ADSB_RANK)).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
