"""v40 = v33 pipeline with the tempo, order, stand-gap and queue columns on the base.

The 13 columns come from the src_v2 ranking frame, joined by movement id. The
base members are lgbm_r_all_v40_s{42,43,44} from train_r_all_v40.py. The LIRF
head, Step A, ITY340 and post-processing stay on v33 defaults.
"""
import os
import sys

import pandas as pd

from predict_v30 import MODELS, main
from predict_v33 import R_NORM_FILES
from train_r_all_v40 import TEMPO_COLS, V2_FRAME

RANK_FRAME = os.path.join(os.path.dirname(V2_FRAME), "frame_rank.parquet")

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v40.parquet"
    extra = pd.read_parquet(RANK_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    main(name, r_norm_files=R_NORM_FILES, base_model="lgbm_r_all_v40", extra_columns=extra)
