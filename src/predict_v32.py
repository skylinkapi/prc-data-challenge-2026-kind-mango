"""v32 = v30 with two variance cuts (sixth audit, section 1.3 step 4).

  1. Base ensemble grows from 3 seeds to 7 (seeds 42..48).
  2. R_norm_LIRF grows from 1 booster to a 5-member mean, clipped at 0 per member
     and at 4,431 s on the mean.

No rule change. Feature list, parameters, stop sets and Step A / ITY340 rules are
identical to v30.
"""
import sys
from predict_v30 import main

BASE_SEEDS = [42, 43, 44, 45, 46, 47, 48]
R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in [42, 43, 44, 45, 46]]


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v32.parquet"
    main(name, base_seeds=BASE_SEEDS, r_norm_files=R_NORM_FILES)
