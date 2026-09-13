"""v41 = v40 pipeline with a new R_norm_LIRF term: five members trained with the
13 tempo columns (train_r_norm_lirf_v41.py) and no 4,431 s cap. Base, gate,
Step A, ITY340 and post-processing stay at v40.
"""
import sys

import pandas as pd

from predict_v30 import main
from predict_v40 import RANK_FRAME
from train_r_all_v40 import TEMPO_COLS
from train_r_norm_lirf_seeds import SEEDS

R_NORM_FILES = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in SEEDS]

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v41.parquet"
    extra = pd.read_parquet(RANK_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    main(name, r_norm_files=R_NORM_FILES, base_model="lgbm_r_all_v40", extra_columns=extra,
         r_norm_features="lirf_regime_v41.features.txt", r_norm_clip=None)
