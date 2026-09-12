"""Label-free price of the 5-member p_fb_LIRF mean on the 2026 ranking set.

Computes the ambiguity A of the final LIRF predictions (mixture, Step A, ITY340, clip)
across the p_fb members, with R_norm_LIRF fixed at the v33 5-member mean. The expected
gain of the mean over one member drawn the same way is A (k = 5, m = 1).
"""
import json, os
import numpy as np
import lightgbm as lgb

from price_ensemble import build_dep_frames
from predict_v23 import apply_stepA_v22
from predict_v30 import (MODELS, R_NORM_CLIP, P24_ITY, ITY_SD_THRESHOLD, NORMAL_MEAN_LIRF,
                         DEFAULT_P_FB_MEMBERS, DEFAULT_P_FB_FEATURES, predict_p_fb)

ROOT = os.path.dirname(MODELS)
N_RANK = 344_841
V33_RMSE = 301.87
LOTTERY_MSE = 268.0
DRAW_RATIO_MAX = 3.0
R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in (42, 43, 44, 45, 46)]
P_FB_MEMBERS = [(f"lgbm_p_fb_lirf_s{s}.txt", f"lirf_p_fb_s{s}.isotonic.pkl") for s in (42, 43, 44, 45, 46)]


def final_lirf(p_fb, r_norm, sd, fltid, band):
    """v33 LIRF scoring for one probability vector."""
    valid = ~np.isnan(sd)
    mix = np.where(valid, p_fb * sd + (1 - p_fb) * r_norm, r_norm)
    out, cell = apply_stepA_v22(mix, sd, np.full(len(sd), "LIRF"), fltid, band)
    ity = valid & (sd > ITY_SD_THRESHOLD) & ~cell
    out[ity] = P24_ITY * (86400 + NORMAL_MEAN_LIRF) + (1 - P24_ITY) * sd[ity]
    return np.clip(out, 0, None)


def main():
    dep, dep_lirf_full = build_dep_frames()
    lirf = (dep["ADEP_mvt"].astype(str) == "LIRF").values
    frame = dep_lirf_full[lirf].reset_index(drop=True)
    sd = dep["sched_delay"].values.astype(float)[lirf]
    fltid = dep["FLIGHT_ID_mvt"].values[lirf]
    with open(os.path.join(MODELS, "lirf_regime.features.txt")) as f:
        feat_norm = f.read().splitlines()
    r_norm = np.minimum(np.mean([np.clip(lgb.Booster(model_file=os.path.join(MODELS, fn))
                                         .predict(frame[feat_norm]), 0, None)
                                 for fn in R_NORM_FILES], axis=0), R_NORM_CLIP)
    with open(os.path.join(MODELS, "lirf_band_table_v30.json")) as f:
        band = json.load(f)

    members = np.stack([final_lirf(predict_p_fb(frame, [m], DEFAULT_P_FB_FEATURES), r_norm, sd, fltid, band)
                        for m in P_FB_MEMBERS])
    shipped = final_lirf(predict_p_fb(frame, DEFAULT_P_FB_MEMBERS, DEFAULT_P_FB_FEATURES), r_norm, sd, fltid, band)
    mean = members.mean(axis=0)
    a_lirf = float(((members - mean) ** 2).mean())
    d_shipped = float(((shipped - mean) ** 2).mean())
    gain = a_lirf * lirf.sum() / N_RANK
    draw_ratio = d_shipped / (1.5 * a_lirf)
    diff = np.abs(mean - shipped)
    report = {"n_lirf": int(lirf.sum()), "A_lirf": a_lirf, "gain_mse": gain,
              "predicted_rmse": float(np.sqrt(V33_RMSE ** 2 - gain)),
              "d_shipped_lirf": d_shipped, "draw_ratio": draw_ratio,
              "mean_abs_diff_vs_v33": float(diff.mean()), "max_abs_diff_vs_v33": float(diff.max()),
              "rows_moved_over_100s": int((diff > 100).sum()),
              "passes_price": bool(gain >= LOTTERY_MSE), "passes_draw": bool(draw_ratio <= DRAW_RATIO_MAX)}
    print(json.dumps(report, indent=1))
    with open(os.path.join(ROOT, "submission", "v37_price.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
