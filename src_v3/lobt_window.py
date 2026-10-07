"""J1: the NM join window. Every 2025 departure with an NM record has `|BLOCK - LOBT| <= 3,606` s.

The organiser joined a movement to its NM flight only inside that window (2,062,577 rows of 2025, no
exception). So the label of a departure with `LOBT` lies in `MVT - LOBT` plus or minus `LOBT_WINDOW_S`,
and a schedule copy (`BLOCK = SCHED`) is impossible where `|SCHED - LOBT|` is larger. `no_copy` sets the
LIRF gate to 0 on those rows; `predict_v67 --copy-window` passes it to the LIRF head. `main` projects a
served file onto the window. It runs last, after the stack, the A5 bound and `step_a_date`. Rows without
`LOBT` keep their value.
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


def no_copy(frame: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """LIRF gate `p` with 0 where a schedule copy would break the window; `frame` holds the departures of `p`."""
    far = np.abs(offsets(frame["MVT_ID_mvt"])["sl"].values) > C.LOBT_WINDOW_S
    hit = far & (frame["ADEP_mvt"].astype(str) == "LIRF").values
    log.info("copy test: LIRF gate set to 0 on %d rows; their mean gate was %.4f", int(hit.sum()),
             float(p[hit].mean()) if hit.any() else 0.0)
    return np.where(hit, 0.0, p)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="served file in submission/")
    ap.add_argument("out", help="output file in submission/")
    args = ap.parse_args()
    sub = pd.read_parquet(C.ROOT / "submission" / args.src)
    pred = sub["TAXITIME_SEC_mvt"].values.astype(float)
    new = window(pred, offsets(sub["MVT_ID_mvt"])["ml"].values)
    sub.assign(TAXITIME_SEC_mvt=new.astype("float64")).to_parquet(C.ROOT / "submission" / args.out, index=False)
    log.info("window: %d rows moved (%d up, %d down); live gain at least %.1f MSE", int((new != pred).sum()),
             int((new > pred).sum()), int((new < pred).sum()), float(((new - pred) ** 2).sum() / C.RANK_ROWS))


if __name__ == "__main__":
    main()
