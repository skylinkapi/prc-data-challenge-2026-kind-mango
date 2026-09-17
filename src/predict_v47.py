"""v47 = v46 pipeline with a 12-month refit of the base.

The base members are lgbm_r_all_v47_s{42,43,44} from train_r_all_v47.py,
trained on ALL 12 months with the retuned params and scaled iteration counts.
All other components (LIRF head, Step A, ITY340, extra columns) stay on v46.
"""
import sys

from predict_v30 import main
from predict_v45 import R_NORM_FILES, build_extra

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v47.parquet"
    main(name, r_norm_files=R_NORM_FILES, base_model="lgbm_r_all_v47",
         extra_columns=build_extra(),
         r_norm_features="lirf_regime_v41.features.txt", r_norm_clip=None)
