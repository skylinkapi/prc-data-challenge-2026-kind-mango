"""Frame loader that reuses the existing v47 feature caches.

Full MP4 requires refitting every artefact per fold. That is Phase 0's
biggest remaining piece and rewrites the encoders, rate maps and route
medians per fold. This module is the light path: it loads the frozen v47
feature frame (h1 cache + v40 tempo + v44 quartiles + v45 plan) so a fold
retrain of the v47 recipe can measure the identical-retrain noise (MP3)
without redoing the feature engineering. MP4 layers on top later.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.merge import assert_row_count

MODELS_OLD = C.ROOT / "models"
H1 = MODELS_OLD / "v36_tune_cache.parquet"
H1_FEAT = MODELS_OLD / "v36_tune_cache.parquet.feat.txt"
H1_IDS = MODELS_OLD / "v36_tune_cache.ids.parquet"
V2_TRAIN = MODELS_OLD / "v2" / "cache" / "frame_train.parquet"
TEMPO_P2575 = MODELS_OLD / "tempo_p2575_train.parquet"
PLAN_TRAIN = MODELS_OLD / "plan_taxi_res_train.parquet"

V40_TEMPO = ["nb_eobt_med_apt30", "nb_eobt_mean_apt30", "nb_eobt_cnt_apt30",
             "nb_eobt_med_apt60", "nb_eobt_mean_apt60", "nb_eobt_cnt_apt60",
             "nb_eobt_med_rwy30", "nb_eobt_mean_rwy30", "nb_eobt_cnt_rwy30",
             "order_later_eobt_30", "order_total_30", "stand_gap",
             "queue_eobt_mvt"]
V44_QUARTILES = ["nb_eobt_p25_apt30", "nb_eobt_p75_apt30",
                 "nb_eobt_p25_apt60", "nb_eobt_p75_apt60",
                 "nb_eobt_p25_rwy30", "nb_eobt_p75_rwy30"]
V45_PLAN = ["plan_taxi_res"]


def v47_feature_names() -> list[str]:
    with open(H1_FEAT) as f:
        base = f.read().split()
    return base + V40_TEMPO + V44_QUARTILES + V45_PLAN


def load_v47_frame() -> pd.DataFrame:
    """Reproduces the v47 feature frame used by train_r_all_v47.py.

    Column order matches `lgbm_r_all_v47.features.txt` on load. The base
    columns come from the H1 cache; the v40 tempo, v44 quartile and v45
    plan-residual columns come from their own caches keyed on MVT_ID_mvt.
    """
    dep = pd.read_parquet(H1)
    dep["MVT_ID_mvt"] = pd.read_parquet(H1_IDS)["MVT_ID_mvt"].values
    for extra, cols in (
        (V2_TRAIN, V40_TEMPO),
        (TEMPO_P2575, V44_QUARTILES),
        (PLAN_TRAIN, V45_PLAN),
    ):
        merged = pd.read_parquet(extra, columns=["MVT_ID_mvt", *cols])
        n_before = len(dep)
        dep = dep.merge(merged, on="MVT_ID_mvt", how="left")
        assert_row_count(dep, n_before, f"merge {extra.name}")
    return dep


def fold_masks(month: pd.Series, hold: tuple[int, int],
               early_stop: tuple[int, int]
               ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Boolean masks for train, early-stop and hold-out rows.

    `train` is every month outside `hold` and `early_stop`.
    """
    m = month.astype(int)
    hold_mask = m.isin(hold)
    stop_mask = m.isin(early_stop)
    train_mask = ~hold_mask & ~stop_mask
    return train_mask, stop_mask, hold_mask
