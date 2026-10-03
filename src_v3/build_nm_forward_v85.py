"""WINNING_PLAN L8 (v85): NM clocks of the departures that take off after the row.

For each departure, the mean and the count of `MVT - AOBT_3` over the
other departures at the same airport, and on the same runway, with a
take-off in `(t, t + W]`. These flights queued with the row, so their
clocks describe the queue the row was in. The organiser allows features
from other movements ("what is occupied, what is coming", 2026-09-09);
the team stance MF6 kept forward windows out until track B.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep
from src_v3.build_nm_neighbours_v73 import GROUPS, window_stats

OUT_TRAIN = C.ROOT / "models" / "nm_forward_train_v86.parquet"
OUT_RANK = C.ROOT / "models" / "nm_forward_rank_v86.parquet"
log = logging.getLogger(__name__)


def forward(dep: pd.DataFrame, windows: tuple[int, ...] = C.NM_FORWARD_WINDOWS_S) -> pd.DataFrame:
    """The forward-window columns keyed on `MVT_ID_mvt`; negated times turn `[s - w, s)` into `(t, t + w]`."""
    d = dep.assign(t=(dep["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1),
                   v=(dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds())
    d["v"] = d["v"].where(d["v"].between(0, C.NM_CLOCK_CLIP_S))
    d = d.dropna(subset=["t"]).sort_values("t", ascending=False, kind="stable")
    out = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].values}, index=d.index)
    for g, keys in GROUPS.items():
        for _, grp in d.groupby(keys, sort=False, dropna=True):
            s, v = -grp["t"].values.astype(np.int64), grp["v"].values.astype(float)
            for w in windows:
                mean, n = window_stats(s, v, w)
                out.loc[grp.index, f"fw_{g}{w // 60}_mean"] = mean
                out.loc[grp.index, f"fw_{g}{w // 60}_n"] = n
    log.info("forward windows %s on %d rows", windows, len(out))
    return out.reset_index(drop=True)


def queue_counts(a: np.ndarray, t: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Departures with off-block before `at` and take-off after `at`; `a` and `t` are sorted copies."""
    return np.searchsorted(a, at, "left") - np.searchsorted(t, at, "right")


def queues(dep: pd.DataFrame) -> pd.DataFrame:
    """Queue length at the row's off-block and at its take-off, keyed on `MVT_ID_mvt`."""
    sec = lambda ts: ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype(float)
    d = dep.assign(t=sec(dep["MVT_TIME_UTC_mvt"]), a=sec(dep["AOBT_3_flt"])).dropna(subset=["t", "a"])
    out = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].values}, index=d.index)
    for g, keys in GROUPS.items():
        for _, grp in d.groupby(keys, sort=False, dropna=True):
            a, t = np.sort(grp["a"].values), np.sort(grp["t"].values)
            own_a, own_t = grp["a"].values, grp["t"].values
            out.loc[grp.index, f"q_push_{g}"] = queue_counts(a, t, own_a)
            out.loc[grp.index, f"q_take_{g}"] = queue_counts(a, t, own_t)
    return out.reset_index(drop=True)


def build(dep: pd.DataFrame) -> pd.DataFrame:
    return forward(dep).merge(queues(dep), on="MVT_ID_mvt", how="left", validate="1:1")


def main() -> None:
    """Write the v86 files (forward windows and queues) and the served v85 files (forward windows only)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for paths, out in ((sorted(C.TRAIN_DIR.glob("training_*.parquet")), OUT_TRAIN), ([C.RANK], OUT_RANK)):
        f = build(load_dep(paths))
        f.to_parquet(out)
        f[[c for c in f if not c.startswith("q_")]].to_parquet(str(out).replace("_v86", "_v85"))


if __name__ == "__main__":
    main()
