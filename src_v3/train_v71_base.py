"""MF2: v65 base recipe plus the seven runway-queue columns (`build_runway_queue_v71.py`)."""
from __future__ import annotations

import logging

import pandas as pd

from src_v3.build_plan_nm_taxi_v65 import OUT_TRAIN as PLAN_NM_TRAIN
from src_v3.build_runway_queue_v71 import FEATURES as MF2, OUT_TRAIN as MF2_TRAIN
from src_v3.frames import v47_feature_names
from src_v3.train_v57_base import _load_frame_v57, train_base


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep = _load_frame_v57()
    for path in (PLAN_NM_TRAIN, MF2_TRAIN):
        dep = dep.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    train_base(dep, v47_feature_names() + ["plan_nm_taxi", *MF2], "v71",
               "v65 recipe plus the MF2 runway-queue columns")


if __name__ == "__main__":
    main()
