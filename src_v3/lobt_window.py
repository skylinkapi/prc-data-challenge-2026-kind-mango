"""J1: the NM join window. Every 2025 departure with an NM record has `|BLOCK - LOBT| <= 3,606` s.

The organiser joined a movement to its NM flight only inside that window (2,062,577 rows of 2025, no
exception). So the label of a departure with `LOBT` lies in `MVT - LOBT` plus or minus `LOBT_WINDOW_S`,
and a schedule copy (`BLOCK = SCHED`) is impossible where `|SCHED - LOBT|` is larger. `no_copy` sets the
LIRF gate to 0 on those rows; `predict_v67 --copy-window` passes it to the LIRF head. `main` projects a
served file onto the window. It runs last, after the stack, the A5 bound and `step_a_date`. Rows without
`LOBT` keep their value. With `--copy-linked`, J2 also sets the gate to 0 on LIRF departures without a record
whose arrival row at another of the ten airports gives a `LOBT` within the window of `SCHED`. With `--anchor`,
J3 first serves `MVT - LOBT` minus the 2025 median `BLOCK - LOBT` on rows whose `LOBT` lies 2 to 4 h before the
take-off, where the clipped clock columns of the trees stop.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C

TIMES = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt"]
log = logging.getLogger(__name__)


def offsets(ids: pd.Series) -> pd.DataFrame:
    """`MVT - LOBT` and `SCHED - LOBT` in seconds of the ranking departures `ids`, in their order; NaN without `LOBT`."""
    r = pd.read_parquet(C.RANK, columns=["PHASE_mvt", *TIMES])
    r = r[r["PHASE_mvt"] == "DEP"]
    t = pd.DataFrame({"MVT_ID_mvt": ids.values}).merge(r, on="MVT_ID_mvt", how="left", validate="1:1")
    return pd.DataFrame({"ml": (t["MVT_TIME_UTC_mvt"] - t["LOBT_flt"]).dt.total_seconds().values,
                         "sl": (t["SCHED_TIME_UTC_mvt"] - t["LOBT_flt"]).dt.total_seconds().values})


def window(pred: np.ndarray, ml: np.ndarray) -> np.ndarray:
    """`pred` projected onto `ml` plus or minus `LOBT_WINDOW_S` where `ml` is known."""
    return np.where(np.isnan(ml), pred, np.clip(pred, ml - C.LOBT_WINDOW_S, ml + C.LOBT_WINDOW_S))


def anchor(pred: np.ndarray, ml: np.ndarray, medians: tuple = C.LOBT_ANCHOR_U_S) -> np.ndarray:
    """J3: `ml` minus the median `BLOCK - LOBT` of its band where `ml` lies in `LOBT_ANCHOR_EDGES_S`."""
    band = np.searchsorted(C.LOBT_ANCHOR_EDGES_S, np.nan_to_num(ml, nan=-1.0), side="right") - 1
    inside = (band >= 0) & (band < len(medians))
    u = np.asarray(medians)[np.clip(band, 0, len(medians) - 1)]
    return np.where(inside, ml - u, pred)


def arrival_lobt(paths: list) -> pd.DataFrame:
    """`LOBT` of the arrival row of the same flight number at the destination, for departures without an NM record."""
    cols = ["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt", "ADES_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]
    m = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    dep = m[(m["PHASE_mvt"] == "DEP") & m["FLIGHT_ID_mvt"].isna()]
    arr = m[(m["PHASE_mvt"] == "ARR") & m["FLIGHT_ID_mvt"].notna()][["FLIGHT_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]]
    j = dep[["MVT_ID_mvt", "FLIGHT_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]].merge(arr, on=["FLIGHT_mvt", "ADES_mvt"], suffixes=("", "_arr"))
    j["lag"] = (j["MVT_TIME_UTC_mvt_arr"] - j["MVT_TIME_UTC_mvt"]).dt.total_seconds()
    j = j[j["lag"].between(*C.ARR_LINK_S, inclusive="neither")].sort_values("lag").drop_duplicates("MVT_ID_mvt")
    return j[["MVT_ID_mvt", "LOBT_flt"]].rename(columns={"LOBT_flt": "lobt_arr"})


def copy_would_join(ids: pd.Series) -> np.ndarray:
    """J2: departures without a record whose arrival-side `LOBT` lies within the window of `SCHED`.

    Such a departure failed the join, so its reported off-block is not `SCHED`: a copy would have joined."""
    r = pd.read_parquet(C.RANK, columns=["MVT_ID_mvt", "PHASE_mvt", "SCHED_TIME_UTC_mvt"])
    t = pd.DataFrame({"MVT_ID_mvt": ids.values}).merge(r[r["PHASE_mvt"] == "DEP"], on="MVT_ID_mvt", how="left", validate="1:1")
    t = t.merge(arrival_lobt([C.RANK]), on="MVT_ID_mvt", how="left", validate="1:1")
    return np.abs((t["SCHED_TIME_UTC_mvt"] - t["lobt_arr"]).dt.total_seconds().values) <= C.LOBT_WINDOW_S


def no_copy(frame: pd.DataFrame, p: np.ndarray, linked: bool = False) -> np.ndarray:
    """LIRF gate `p` with 0 where a schedule copy is impossible; `frame` holds the departures of `p`.

    J1: the copy would break the window of a record row. J2 (`linked`): the copy would have joined."""
    ids = frame["MVT_ID_mvt"]
    blocked = np.abs(offsets(ids)["sl"].values) > C.LOBT_WINDOW_S
    if linked:
        blocked |= copy_would_join(ids)
    hit = blocked & (frame["ADEP_mvt"].astype(str) == "LIRF").values
    log.info("copy test: LIRF gate set to 0 on %d rows; their mean gate was %.4f", int(hit.sum()),
             float(p[hit].mean()) if hit.any() else 0.0)
    return np.where(hit, 0.0, p)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="served file in submission/")
    ap.add_argument("out", help="output file in submission/")
    ap.add_argument("--anchor", action="store_true", help="J3: anchor rows with LOBT 2 to 4 h before take-off first.")
    args = ap.parse_args()
    sub = pd.read_parquet(C.ROOT / "submission" / args.src)
    ml = offsets(sub["MVT_ID_mvt"])["ml"].values
    pred = sub["TAXITIME_SEC_mvt"].values.astype(float)
    if args.anchor:
        anchored = anchor(pred, ml)
        log.info("anchor: %d rows moved, mean move %.0f s", int((anchored != pred).sum()), float((anchored - pred)[anchored != pred].mean()))
        pred = anchored
    new = window(pred, ml)
    sub.assign(TAXITIME_SEC_mvt=new.astype("float64")).to_parquet(C.ROOT / "submission" / args.out, index=False)
    log.info("window: %d rows moved (%d up, %d down); live gain at least %.1f MSE", int((new != pred).sum()),
             int((new > pred).sum()), int((new < pred).sum()), float(((new - pred) ** 2).sum() / C.RANK_ROWS))


if __name__ == "__main__":
    main()
