"""WINNING_PLAN L2 (track M): NM clocks of the other departures, backward windows.

For each departure, the mean and the count of `MVT - AOBT_3` over the
other departures at the same airport, and on the same runway, with a
take-off in `[t - W, t)`. Only clocks in 0 to 7,200 s count. The window
ends before the row's own take-off, so the row's own clock never enters.
This is a live measure of the airport's taxi-out times.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep

OUT_TRAIN = C.ROOT / "models" / "nm_neighbours_train_v73.parquet"
OUT_RANK = C.ROOT / "models" / "nm_neighbours_rank_v73.parquet"
GROUPS = {"apt": ["ADEP_mvt"], "rwy": ["ADEP_mvt", "RUNWAY_mvt"]}
FEATURES = [f"nb_{g}{w // 60}_{s}" for g in GROUPS for w in C.NM_NEIGHBOUR_WINDOWS_S
            for s in ("mean", "n")]
log = logging.getLogger(__name__)


def window_stats(t: np.ndarray, v: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of the finite `v` with time in `[t - w, t)`; `t` is sorted."""
    ok = np.isfinite(v)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    lo, hi = np.searchsorted(t, t - w, "left"), np.searchsorted(t, t, "left")
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan), n


def neighbours(dep: pd.DataFrame) -> pd.DataFrame:
    """The neighbour columns keyed on `MVT_ID_mvt`."""
    d = dep.assign(t=(dep["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1),
                   v=(dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds())
    d["v"] = d["v"].where(d["v"].between(0, C.NM_CLOCK_CLIP_S))
    d = d.dropna(subset=["t"]).sort_values("t", kind="stable")
    out = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].values}, index=d.index)
    for g, keys in GROUPS.items():
        for _, grp in d.groupby(keys, sort=False, dropna=True):
            t, v = grp["t"].values.astype(np.int64), grp["v"].values.astype(float)
            for w in C.NM_NEIGHBOUR_WINDOWS_S:
                mean, n = window_stats(t, v, w)
                out.loc[grp.index, f"nb_{g}{w // 60}_mean"] = mean
                out.loc[grp.index, f"nb_{g}{w // 60}_n"] = n
    log.info("coverage of nb_apt30_mean %.4f on %d rows", out["nb_apt30_mean"].notna().mean(), len(out))
    return out.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    neighbours(load_dep(sorted(C.TRAIN_DIR.glob("training_*.parquet")))).to_parquet(OUT_TRAIN)
    neighbours(load_dep([C.RANK])).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
