"""v46 = v45 pipeline with the retuned base (`train_r_all_v46.py`).

Same features as v45 (v40 tempo + v44 eobt quartiles + v45 plan_taxi_res)
but the base members carry the Optuna-tuned hyperparameters. LIRF head,
Step A, ITY340 and post-processing stay on v41.
"""
import sys

from predict_v30 import main
from predict_v45 import R_NORM_FILES, build_extra

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v46.parquet"
    main(name, r_norm_files=R_NORM_FILES, base_model="lgbm_r_all_v46",
         extra_columns=build_extra(),
         r_norm_features="lirf_regime_v41.features.txt", r_norm_clip=None)
