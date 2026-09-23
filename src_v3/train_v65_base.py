"""L9 (WINNING_PLAN R5, MF5): v57 base recipe plus the `plan_nm_taxi` column."""
from __future__ import annotations

import logging

import pandas as pd

from src_v3.build_plan_nm_taxi_v65 import OUT_TRAIN
from src_v3.frames import v47_feature_names
from src_v3.train_v57_base import _load_frame_v57, train_base


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep = _load_frame_v57().merge(pd.read_parquet(OUT_TRAIN), on="MVT_ID_mvt",
                                  how="left", validate="m:1")
    train_base(dep, v47_feature_names() + ["plan_nm_taxi"], "v65",
               "v57 recipe plus plan_nm_taxi (WINNING_PLAN L9, MF5)")


if __name__ == "__main__":
    main()
