"""WINNING_PLAN L1 (track B): v65 base recipe plus the five NM clock columns (`build_nm_clocks_v73.py`)."""
from __future__ import annotations

import logging

import pandas as pd

from src_v3.build_nm_clocks_v73 import FEATURES as CLOCKS, OUT_TRAIN as CLOCKS_TRAIN
from src_v3.build_plan_nm_taxi_v65 import OUT_TRAIN as PLAN_NM_TRAIN
from src_v3.frames import v47_feature_names
from src_v3.train_v57_base import _load_frame_v57, train_base


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep = _load_frame_v57()
    for path in (PLAN_NM_TRAIN, CLOCKS_TRAIN):
        dep = dep.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    train_base(dep, v47_feature_names() + ["plan_nm_taxi", *CLOCKS], "v73",
               "v65 recipe plus the NM clock columns (WINNING_PLAN L1, track B)")


if __name__ == "__main__":
    main()
