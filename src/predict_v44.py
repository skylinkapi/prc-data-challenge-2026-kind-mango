"""v44 = v41 pipeline with p25/p75 neighbour EOBT columns on the base.

The v40 tempo columns and the six new p25/p75 columns join to `extra_columns`
of predict_v30.main. The base members are lgbm_r_all_v44_s{42,43,44} from
train_r_all_v44.py. LIRF head, Step A, ITY340 and post-processing stay on v41.
"""
import os
import sys

import pandas as pd

from predict_v30 import main
from predict_v40 import RANK_FRAME
from train_r_all_v40 import MODELS, TEMPO_COLS
from train_r_all_v44 import EXTRA_COLS
from train_r_norm_lirf_seeds import SEEDS

R_NORM_FILES = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in SEEDS]
P2575_RANK = os.path.join(MODELS, "tempo_p2575_rank.parquet")


def build_extra() -> pd.DataFrame:
    tempo = pd.read_parquet(RANK_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    p2575 = pd.read_parquet(P2575_RANK, columns=["MVT_ID_mvt", *EXTRA_COLS])
    return tempo.merge(p2575, on="MVT_ID_mvt", how="left")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v44.parquet"
    main(name, r_norm_files=R_NORM_FILES, base_model="lgbm_r_all_v44",
         extra_columns=build_extra(),
         r_norm_features="lirf_regime_v41.features.txt", r_norm_clip=None)
