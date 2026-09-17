"""v45 = v44 pipeline with plan_taxi_res added to the base.

The v40 tempo, v44 eobt quartile and v45 plan-taxi-residual columns join into
`extra_columns` of predict_v30.main. Base members are lgbm_r_all_v45_s{42,43,44}.
LIRF head, Step A, ITY340 and post-processing stay on v41.
"""
import os
import sys

import pandas as pd

from predict_v30 import main
from predict_v40 import RANK_FRAME
from train_r_all_v40 import MODELS, TEMPO_COLS
from train_r_all_v44 import EXTRA_COLS as EOBT_P2575
from train_r_all_v45plan import EXTRA_COLS as PLAN_COLS
from train_r_norm_lirf_seeds import SEEDS

R_NORM_FILES = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in SEEDS]
P2575_RANK = os.path.join(MODELS, "tempo_p2575_rank.parquet")
PLAN_RANK = os.path.join(MODELS, "plan_taxi_res_rank.parquet")


def build_extra() -> pd.DataFrame:
    tempo = pd.read_parquet(RANK_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    p2575 = pd.read_parquet(P2575_RANK, columns=["MVT_ID_mvt", *EOBT_P2575])
    plan = pd.read_parquet(PLAN_RANK, columns=["MVT_ID_mvt", *PLAN_COLS])
    return tempo.merge(p2575, on="MVT_ID_mvt", how="left").merge(plan, on="MVT_ID_mvt", how="left")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v45.parquet"
    main(name, r_norm_files=R_NORM_FILES, base_model="lgbm_r_all_v45",
         extra_columns=build_extra(),
         r_norm_features="lirf_regime_v41.features.txt", r_norm_clip=None)
