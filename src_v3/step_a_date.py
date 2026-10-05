"""Step A without the 24-h term where the local dates do not cross.

The 24-h artefact at LIRF stores the off-block as the local date of the schedule plus the local clock
time of the real off-block, so the label is 86,400 s times the local day difference plus the taxi. With
no local date crossing between the schedule and the take-off, the artefact cannot occur: on the 2025
null-record LIRF rows above the Step A gate, 0 of 64 such rows carry it, against 11 of 63 rows with a
crossing. The band table (`lirf_band_table_v30.json`) ignores the dates. This module serves its fallback
and genuine terms only on those rows. It reads the genuine term back from the served value, so it needs
no model. Every other row keeps its value.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C

BAND_TABLE = C.ROOT / "models" / "lirf_band_table_v30.json"
LIRF_TZ = "Europe/Rome"
log = logging.getLogger(__name__)


def day_crossing(dep: pd.DataFrame) -> np.ndarray:
    """Local calendar days from the schedule to the take-off."""
    m = dep["MVT_TIME_UTC_mvt"].dt.tz_convert(LIRF_TZ).dt.normalize()
    s = dep["SCHED_TIME_UTC_mvt"].dt.tz_convert(LIRF_TZ).dt.normalize()
    return (m - s).dt.days.values


def adjust(dep: pd.DataFrame, pred: np.ndarray, band: dict) -> np.ndarray:
    """`pred` with the 24-h term of the Step A mixture removed on Step A rows without a local date crossing."""
    sd = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds().values
    cell = ((dep["ADEP_mvt"] == "LIRF") & dep["FLIGHT_ID_mvt"].isna()).values & (sd > band["gate_sd"])
    out = pred.copy()
    for i in np.where(cell & (day_crossing(dep) == 0))[0]:
        row = next((band["table"].get(f"{lo}_{hi}") for lo, hi in band["bands"] if lo < sd[i] <= hi), None)
        if row is None or row["p_24h"] == 0:
            continue
        p_fb, p_24 = row["p_fb"], row["p_24h"]
        p_norm = max(0.0, 1.0 - p_fb - p_24)
        extra = row["mean_24h_extra"] if np.isfinite(row["mean_24h_extra"]) else C.STEP_A_24H_EXTRA_S
        genuine = (pred[i] - p_fb * sd[i] - p_24 * (86400.0 + extra)) / p_norm if p_norm > 0 else 0.0
        out[i] = sd[i] if p_fb + p_norm == 0 else (p_fb * sd[i] + p_norm * genuine) / (p_fb + p_norm)
        log.info("Step A row %s: sd %.0f s, %.0f -> %.0f s", dep["FLIGHT_mvt"].iloc[i], sd[i], pred[i], out[i])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="served file in submission/")
    ap.add_argument("out", help="output file in submission/")
    args = ap.parse_args()
    sub = pd.read_parquet(C.ROOT / "submission" / args.src)
    rank = pd.read_parquet(C.RANK, columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt",
                                            "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    dep = sub[["MVT_ID_mvt"]].merge(rank, on="MVT_ID_mvt", how="left", validate="1:1")
    pred = adjust(dep, sub["TAXITIME_SEC_mvt"].values.astype(float), json.loads(BAND_TABLE.read_text()))
    sub.assign(TAXITIME_SEC_mvt=pred.astype("float64")).to_parquet(C.ROOT / "submission" / args.out, index=False)
    log.info("rows moved: %d", int((pred != sub["TAXITIME_SEC_mvt"].values).sum()))


if __name__ == "__main__":
    main()
