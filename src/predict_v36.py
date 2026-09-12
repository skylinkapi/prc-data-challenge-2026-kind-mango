"""v36 = v33 pipeline + F7 zero-clip repair (tenth-audit measure 1).

v33 shipped 23 rows predicted as exactly 0 s (18 LSZH, 3 LTFM, 2 LEBL).
Cause: per-member `np.clip(x, 0, None)` inside the base ensemble destroys
the calibrated ensemble floor whenever every member returns a non-positive
raw prediction.

v36 combines the two lightest audit-recommended fixes:
  1. Remove the per-member clip. The base mean uses raw predictions.
  2. As a safety net after the final clip at 0, any residual exact-zero row
     receives the median of positive predictions at its own airport.

Everything else stays on v33 defaults.
"""
import sys
from predict_v30 import main

R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in (42, 43, 44, 45, 46)]

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v36.parquet"
    main(name,
         r_norm_files=R_NORM_FILES,
         per_member_base_clip=False,
         fill_zero_rows_per_airport=True)
