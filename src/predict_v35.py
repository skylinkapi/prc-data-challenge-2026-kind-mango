"""v35 = v33 pipeline + v34's honestly trained p_fb classifier.

Ninth-audit section 7.3/7.5 repair. v34 refit two different pools together:
the classifier's OOF training pool (correctly excluded {1, 7}) and the
inference-time lookups (wrongly excluded {1, 7}). Dropping months from a
lookup that never sees the label only throws away seasonal information.

v35 keeps the honest classifier and reverts the lookups:
  - classifier + isotonic:   lgbm_p_fb_lirf_v34.txt, lirf_regime_v34.isotonic.pkl
  - fbrate scoring maps:     lirf_regime_v23.rate_maps.pkl  (all months, v33 default)
  - band table:              lirf_band_table_v30.json      (all months, v33 default)
  - NORMAL_MEAN_LIRF, R_NORM_CLIP: 1150, 4431 (predict_v30 defaults)

Expected: reverses the +0.202 s v34 regression by construction; back to
about 301.87 s plus classifier noise.
"""
import sys
from predict_v30 import main

R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in (42, 43, 44, 45, 46)]
P_FB_MEMBERS = [("lgbm_p_fb_lirf_v34.txt", "lirf_regime_v34.isotonic.pkl")]
P_FB_FEATURES = "lirf_regime_v34.features.txt"

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v35.parquet"
    main(name,
         r_norm_files=R_NORM_FILES,
         p_fb_members=P_FB_MEMBERS,
         p_fb_features=P_FB_FEATURES)
