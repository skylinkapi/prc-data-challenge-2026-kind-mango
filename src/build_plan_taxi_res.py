"""L9: plan-taxi residual, clipped to +/- 3,600 s.

Follows src_v2/frame.py: route = ADEP_mvt + "_" + ADES_mvt. `plan_block` is
`mvt_ts - arvt1_ts` in seconds. Route medians come from the 2025 clean
fit-month rows only (|y - sd| > 5, 30 <= y <= 7200, months 2..12 \ {1, 7}).
Writes plan_taxi_res_{train,rank}.parquet and plan_route_medians.parquet.
"""
import logging
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_train.parquet")
RANK_FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_rank.parquet")
OUT_TRAIN = os.path.join(ROOT, "models", "plan_taxi_res_train.parquet")
OUT_RANK = os.path.join(ROOT, "models", "plan_taxi_res_rank.parquet")
ROUTE_MED = os.path.join(ROOT, "models", "plan_route_medians.parquet")
CLIP = 3600
FIT_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
FB_TOL = 5
log = logging.getLogger(__name__)


def load_frame(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt",
                                          "mvt_ts", "ARVT_1_flt", "sd",
                                          "TAXITIME_SEC_mvt", "month"])


def compute_medians(train: pd.DataFrame) -> pd.Series:
    arvt = pd.to_datetime(train["ARVT_1_flt"], utc=True, errors="coerce")
    mvt = pd.to_datetime(train["mvt_ts"], utc=True, errors="coerce")
    plan_block = (mvt - arvt).dt.total_seconds()
    y = train["TAXITIME_SEC_mvt"].astype(float)
    sd = train["sd"].astype(float)
    clean = train["month"].isin(FIT_MONTHS) & y.between(30, 7200) & ((y - sd).abs() > FB_TOL)
    route = train["ADEP_mvt"].astype(str) + "_" + train["ADES_mvt"].astype(str)
    med = plan_block[clean].groupby(route[clean]).median().dropna()
    log.info("route medians: %d routes, median %.0f, p10 %.0f, p90 %.0f",
             len(med), med.median(), med.quantile(0.1), med.quantile(0.9))
    return med


def apply_residual(frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    arvt = pd.to_datetime(frame["ARVT_1_flt"], utc=True, errors="coerce")
    mvt = pd.to_datetime(frame["mvt_ts"], utc=True, errors="coerce")
    plan_block = (mvt - arvt).dt.total_seconds()
    route = frame["ADEP_mvt"].astype(str) + "_" + frame["ADES_mvt"].astype(str)
    res = (plan_block - route.map(medians)).clip(-CLIP, CLIP)
    return pd.DataFrame({"MVT_ID_mvt": frame["MVT_ID_mvt"].values,
                         "plan_taxi_res": res.values})


def main() -> None:
    train = load_frame(FRAME)
    log.info("train %d rows", len(train))
    medians = compute_medians(train)
    medians.to_frame("plan_block_median").to_parquet(ROUTE_MED)
    log.info("wrote %s", ROUTE_MED)

    res_train = apply_residual(train, medians)
    cov = res_train["plan_taxi_res"].notna().mean()
    res_train.to_parquet(OUT_TRAIN)
    log.info("wrote %s (%d rows, coverage %.3f)", OUT_TRAIN, len(res_train), cov)

    rank = load_frame(RANK_FRAME)
    res_rank = apply_residual(rank, medians)
    cov = res_rank["plan_taxi_res"].notna().mean()
    res_rank.to_parquet(OUT_RANK)
    log.info("wrote %s (%d rows, coverage %.3f)", OUT_RANK, len(res_rank), cov)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
