"""MB8 for the plan_taxi_res route medians.

Same computation as `src/build_plan_taxi_res.py` but the median per route
is taken over ALL 12 months of 2025 clean rows (not the 10 fit months).
The residual value for each row is then computed against this all-year
median. Writes:

  models/plan_route_medians_v57.parquet
  models/plan_taxi_res_train_v57.parquet
  models/plan_taxi_res_rank_v57.parquet
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src_v3 import config as C

ROOT = C.ROOT
FRAME = ROOT / "models" / "v2" / "cache" / "frame_train.parquet"
RANK_FRAME = ROOT / "models" / "v2" / "cache" / "frame_rank.parquet"
OUT_TRAIN = ROOT / "models" / "plan_taxi_res_train_v57.parquet"
OUT_RANK = ROOT / "models" / "plan_taxi_res_rank_v57.parquet"
MED_OUT = ROOT / "models" / "plan_route_medians_v57.parquet"
CLIP = 3600
FB_TOL = 5
log = logging.getLogger(__name__)


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt",
                                          "mvt_ts", "ARVT_1_flt", "sd",
                                          "TAXITIME_SEC_mvt", "month"])


def _medians_all_months(train: pd.DataFrame) -> pd.Series:
    arvt = pd.to_datetime(train["ARVT_1_flt"], utc=True, errors="coerce")
    mvt = pd.to_datetime(train["mvt_ts"], utc=True, errors="coerce")
    plan_block = (mvt - arvt).dt.total_seconds()
    y = train["TAXITIME_SEC_mvt"].astype(float)
    sd = train["sd"].astype(float)
    # Clean rows over all 12 months, non-fallback (|y - sd| > 5)
    clean = y.between(30, 7200) & ((y - sd).abs() > FB_TOL)
    route = train["ADEP_mvt"].astype(str) + "_" + train["ADES_mvt"].astype(str)
    med = plan_block[clean].groupby(route[clean]).median().dropna()
    log.info("all-months route medians: %d routes (median %.0f, p10 %.0f, p90 %.0f)",
             len(med), med.median(), med.quantile(0.1), med.quantile(0.9))
    return med


def _residual(frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    arvt = pd.to_datetime(frame["ARVT_1_flt"], utc=True, errors="coerce")
    mvt = pd.to_datetime(frame["mvt_ts"], utc=True, errors="coerce")
    plan_block = (mvt - arvt).dt.total_seconds()
    route = frame["ADEP_mvt"].astype(str) + "_" + frame["ADES_mvt"].astype(str)
    res = (plan_block - route.map(medians)).clip(-CLIP, CLIP)
    return pd.DataFrame({"MVT_ID_mvt": frame["MVT_ID_mvt"].values,
                         "plan_taxi_res": res.values})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info("loading training frame")
    train = _load(FRAME)
    medians = _medians_all_months(train)
    medians.to_frame("plan_block_median").to_parquet(MED_OUT)
    log.info("wrote %s", MED_OUT)

    log.info("computing plan_taxi_res on training rows against new median")
    res_train = _residual(train, medians)
    cov = res_train["plan_taxi_res"].notna().mean()
    res_train.to_parquet(OUT_TRAIN)
    log.info("wrote %s (%d rows, coverage %.3f)", OUT_TRAIN, len(res_train), cov)

    log.info("loading ranking frame")
    rank = _load(RANK_FRAME)
    res_rank = _residual(rank, medians)
    cov = res_rank["plan_taxi_res"].notna().mean()
    res_rank.to_parquet(OUT_RANK)
    log.info("wrote %s (%d rows, coverage %.3f)", OUT_RANK, len(res_rank), cov)


if __name__ == "__main__":
    main()
