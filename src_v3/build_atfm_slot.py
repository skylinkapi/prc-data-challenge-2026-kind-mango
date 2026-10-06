"""ATFM slot adherence of the departure airport on the day of the take-off, keyed on `MVT_ID_mvt`.

EUROCONTROL publishes per airport and UTC day the departures, the regulated departures and how many of
those took off before, inside and after their slot window (ansperformance.eu, `atfm_slot_adherence`). The
counts use take-off and slot times, not the off-block, so they carry no label. A day with many regulated
departures holds aircraft at the stand or in the queue. Columns: `slot_dep`, `slot_reg_share`,
`slot_early_share`, `slot_late_share` (the last two as shares of the regulated departures).
Data (c) EUROCONTROL, used with attribution.
"""
from __future__ import annotations

import logging
import urllib.request

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep

OUT_TRAIN = C.ROOT / "models" / "atfm_slot_train.parquet"
OUT_RANK = C.ROOT / "models" / "atfm_slot_rank.parquet"
log = logging.getLogger(__name__)


def daily() -> pd.DataFrame:
    """Slot columns per airport and UTC day, from the yearly CSVs (downloaded once)."""
    parts = []
    for year in C.ATFM_SLOT_YEARS:
        path = C.ATFM_SLOT_DIR / f"atfm_slot_adherence_{year}.csv"
        if not path.exists():
            urllib.request.urlretrieve(C.EC_CSV_URL.format("atfm_slot_adherence", year), path)
        parts.append(pd.read_csv(path))
    d = pd.concat(parts, ignore_index=True)
    d = d[d["APT_ICAO"].isin(C.TARGETS)]
    reg = d["FLT_DEP_REG_1"].replace(0, np.nan)
    return pd.DataFrame({"ADEP_mvt": d["APT_ICAO"], "day": pd.to_datetime(d["FLT_DATE"]).dt.date,
                         "slot_dep": d["FLT_DEP_1"], "slot_reg_share": d["FLT_DEP_REG_1"] / d["FLT_DEP_1"],
                         "slot_early_share": d["FLT_DEP_OUT_EARLY_1"] / reg, "slot_late_share": d["FLT_DEP_OUT_LATE_1"] / reg})


def columns(dep: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    dep = dep.assign(day=dep["MVT_TIME_UTC_mvt"].dt.tz_convert("UTC").dt.date)
    out = dep[["MVT_ID_mvt", "ADEP_mvt", "day"]].merge(d, on=["ADEP_mvt", "day"], how="left", validate="m:1")
    log.info("slot coverage %.3f", out["slot_dep"].notna().mean())
    return out.drop(columns=["ADEP_mvt", "day"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = daily()
    columns(load_dep(sorted(C.TRAIN_DIR.glob("training_*.parquet"))), d).to_parquet(OUT_TRAIN)
    columns(load_dep([C.RANK]), d).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
