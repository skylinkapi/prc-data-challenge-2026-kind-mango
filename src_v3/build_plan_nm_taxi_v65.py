"""L9 (WINNING_PLAN R5, MF5): planned-taxi proxy from the filed NM plan.

`ARVT_1 - EOBT_1` is the filed block time: planned taxi-out plus flight
time. Within one (ADEP, ADES, aircraft type) key the flight time is near
constant, so the residual against the key's 12-month median isolates the
planned taxi-out. The served `plan_taxi_res` mixes take-off delay into it
(finding C2). Clean rows and the clip follow the v57 route medians.
"""
from __future__ import annotations

import logging

import pandas as pd

from src_v3 import config as C
from src_v3.build_plan_taxi_res_v57 import CLIP, FB_TOL, FRAME, RANK_FRAME

KEY = ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"]
OUT_TRAIN = C.ROOT / "models" / "plan_nm_taxi_train_v65.parquet"
OUT_RANK = C.ROOT / "models" / "plan_nm_taxi_rank_v65.parquet"
log = logging.getLogger(__name__)


def load_plan(path) -> pd.DataFrame:
    """Rows with their key, label, `sd` and filed block time in seconds."""
    f = pd.read_parquet(path, columns=["MVT_ID_mvt", *KEY, "EOBT_1_flt", "ARVT_1_flt",
                                       "sd", "TAXITIME_SEC_mvt"])
    f["plan_block"] = (pd.to_datetime(f["ARVT_1_flt"], utc=True, errors="coerce")
                       - pd.to_datetime(f["EOBT_1_flt"], utc=True, errors="coerce")
                       ).dt.total_seconds()
    f["key"] = f[KEY].astype(str).agg("_".join, axis=1)
    return f


def residual(f: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    res = (f["plan_block"] - f["key"].map(medians)).clip(-CLIP, CLIP)
    log.info("coverage %.3f on %d rows", res.notna().mean(), len(f))
    return pd.DataFrame({"MVT_ID_mvt": f["MVT_ID_mvt"].values, "plan_nm_taxi": res.values})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train = load_plan(FRAME)
    y = train["TAXITIME_SEC_mvt"].astype(float)
    clean = y.between(30, 7200) & ((y - train["sd"].astype(float)).abs() > FB_TOL)
    medians = train.loc[clean, "plan_block"].groupby(train.loc[clean, "key"]).median().dropna()
    log.info("%d keys with a median", len(medians))
    residual(train, medians).to_parquet(OUT_TRAIN)
    residual(load_plan(RANK_FRAME), medians).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
