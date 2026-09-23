"""L11 (WINNING_PLAN R6, MF1): v65 base recipe plus METAR at EOBT_1.

The five weather columns come from `src/build_weather_eobt.py`
(`models/weather_eobt_{train,rank}.parquet`).
"""
from __future__ import annotations

import logging

import pandas as pd

from src_v3 import config as C
from src_v3.build_plan_nm_taxi_v65 import OUT_TRAIN as PLAN_NM_TRAIN
from src_v3.frames import v47_feature_names
from src_v3.train_v57_base import _load_frame_v57, train_base

WX_TRAIN = C.ROOT / "models" / "weather_eobt_train.parquet"
WX_COLS = ["tmpc_eobt", "vis_km_eobt", "wind_kt_eobt", "wx_precip_eobt", "deicing_gate_eobt"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep = _load_frame_v57()
    for path in (PLAN_NM_TRAIN, WX_TRAIN):
        dep = dep.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    train_base(dep, v47_feature_names() + ["plan_nm_taxi", *WX_COLS], "v66",
               "v65 recipe plus weather at EOBT_1 (WINNING_PLAN L11, MF1)")


if __name__ == "__main__":
    main()
