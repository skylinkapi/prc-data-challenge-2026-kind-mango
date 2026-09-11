"""v33 = v30 pipeline + 5-member R_norm_LIRF mean, keeping the 3-seed base.

The v32 debrief (docs/MODEL_ANALYSIS.md seventh audit) shows the 74-MSE
R_norm_LIRF gain lands, while the 7-member base expansion carries a
2026-ranking-set A_7 of 5,417 MSE — 8x its hold-out estimate — and its live
outcome sat inside a lottery too wide to sign. v33 harvests the small, priced
R_norm gain alone.
"""
import sys
from predict_v30 import main

R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in [42, 43, 44, 45, 46]]

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v33.parquet"
    main(name, r_norm_files=R_NORM_FILES)
