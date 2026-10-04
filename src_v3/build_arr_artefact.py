"""Third-edition G1: the arrival artefact shares of each departure's airline, day and month.

At LIRF the departure fallback rate (block on the schedule) follows the
airline's arrival rate of in-block on the schedule: correlation 0.77 over
the airline-month cells of 2025, 0.58 within an airline over months
(`measure_lirf_gap.py`). The ranking file keeps arrival block times, so the
2026 shares exist. Only arrival labels enter; no departure label does.
The 2025 training files and the 2026 ranking file go through the same code.
"""
from __future__ import annotations

import logging

import pandas as pd

from src_v3 import config as C

OUT_TRAIN = C.ROOT / "models" / "arr_art_train_g1.parquet"
OUT_RANK = C.ROOT / "models" / "arr_art_rank_g1.parquet"
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
FEATURES = ["art_pre_day", "art_pre_day_n", "art_pre_month", "art_day"]
log = logging.getLogger(__name__)


def shares(arr: pd.DataFrame, keys: list[str], prior: pd.Series) -> pd.DataFrame:
    """Artefact share per key, smoothed toward `prior` (indexed on the airport and period keys)."""
    g = arr.groupby(keys)["art"].agg(["sum", "size"]).reset_index()
    p = g[prior.index.names].merge(prior.rename("prior").reset_index(), on=list(prior.index.names), how="left")["prior"].values
    k = C.ARTEFACT_SMOOTH_K
    return g.assign(share=(g["sum"] + k * p) / (g["size"] + k))


def arr_artefact(paths: list) -> pd.DataFrame:
    """The four G1 columns of every departure in `paths`, keyed on `MVT_ID_mvt`."""
    m = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    m["apt"] = m["ADEP_mvt"].where(m["PHASE_mvt"] == "DEP", m["ADES_mvt"])
    m["pre"] = m["FLIGHT_mvt"].astype(str).str[:3]
    m["day"], m["month"] = m["MVT_TIME_UTC_mvt"].dt.floor("D"), m["MVT_TIME_UTC_mvt"].dt.month
    arr = m[(m["PHASE_mvt"] == "ARR") & m["BLOCK_TIME_UTC_mvt"].notna()].copy()
    arr["art"] = (arr["BLOCK_TIME_UTC_mvt"] - arr["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs() <= C.FB_TOL_DEFAULT
    day = arr.groupby(["apt", "day"])["art"].mean()
    month = arr.groupby(["apt", "month"])["art"].mean()
    pd_ = shares(arr, ["apt", "pre", "day"], day).rename(columns={"share": "art_pre_day", "size": "art_pre_day_n"})
    pm = shares(arr, ["apt", "pre", "month"], month).rename(columns={"share": "art_pre_month"})
    dep = m[m["PHASE_mvt"] == "DEP"]
    out = (dep[["MVT_ID_mvt", "apt", "pre", "day", "month"]]
           .merge(pd_[["apt", "pre", "day", "art_pre_day", "art_pre_day_n"]], on=["apt", "pre", "day"], how="left")
           .merge(pm[["apt", "pre", "month", "art_pre_month"]], on=["apt", "pre", "month"], how="left")
           .merge(day.rename("art_day").reset_index(), on=["apt", "day"], how="left"))
    out["art_pre_day"] = out["art_pre_day"].fillna(out["art_day"])
    out["art_pre_day_n"] = out["art_pre_day_n"].fillna(0)
    if not out["MVT_ID_mvt"].is_unique:
        raise ValueError("MVT_ID_mvt repeats in the G1 table. Check the merges for duplicate keys, then rerun.")
    log.info("%d departures; art_pre_day mean %.3f, rows with same-day airline arrivals %.3f", len(out),
             out["art_pre_day"].mean(), (out["art_pre_day_n"] > 0).mean())
    return out[["MVT_ID_mvt", *FEATURES]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arr_artefact(sorted(C.TRAIN_DIR.glob("training_*.parquet"))).to_parquet(OUT_TRAIN)
    arr_artefact([C.RANK]).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
