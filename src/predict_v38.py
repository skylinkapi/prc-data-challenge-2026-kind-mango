"""v38 = v33 pipeline with the base retrained on the ARVT_1 planned-time features.

The LIRF head, Step A, ITY340 and post-processing stay on v33 defaults, so only
non-LIRF rows move (docs/MODEL_ANALYSIS.md F18 and section 4).
"""
import sys
from predict_v30 import main

R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in (42, 43, 44, 45, 46)]

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v38.parquet"
    main(name, r_norm_files=R_NORM_FILES, base_model="lgbm_r_all_v38", use_plan_features=True)
