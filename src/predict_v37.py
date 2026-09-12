"""v37 = v33 pipeline + 5-member p_fb_LIRF mean (seeds 42-46), everything else on v33.

The p_fb gate was the last single-draw model in the LIRF head. src/price_p_fb_seeds.py
prices the member mean label-free on the 2026 ranking set (docs/MODEL_ANALYSIS.md section 4).
"""
import sys
from predict_v30 import main

R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in (42, 43, 44, 45, 46)]
P_FB_MEMBERS = [(f"lgbm_p_fb_lirf_s{s}.txt", f"lirf_p_fb_s{s}.isotonic.pkl") for s in (42, 43, 44, 45, 46)]

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v37.parquet"
    main(name, r_norm_files=R_NORM_FILES, p_fb_members=P_FB_MEMBERS)
