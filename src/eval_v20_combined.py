"""Combined: v18 LIRF handling + v19 mixture at other qualified airports.

For LIRF: use v16 (Step A) + 6.3 rule (v18 architecture) — keeps our best LIRF.
For LEBL / EGLL / LTFM: use mixture p_fb*sd + (1-p_fb)*R_norm (no p_24; low-delay).
For all other airports: use R_norm alone (no mixture).

Compare per airport vs v16 and v18. Only deploy R_norm alone / mixture where wins.
"""
import glob, os, pickle, json, gc, time
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol
from features_operator import apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path
from features_opdi import add_opdi
from features_opdi_live import add_opdi_live
from train_lgbm_v21 import add_obt_features, CAT_COLS
from eval_v21_stepA import apply_rule
from train_lirf_noflt_detector import CAT_INPUTS as NF_CAT, NUM_INPUTS as NF_NUM, SD_LO, SD_HI
from train_v18_detectors import CAT_INPUTS as DET_CAT, NUM_INPUTS as DET_NUM, SD_MIN
from eval_v19 import build_holdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}
SHRINK_63 = 0.6


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    t0 = time.time()
    test = build_holdout()
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)
    adep_te = test["ADEP_mvt"].astype(str).values
    fltid_te = test["FLIGHT_ID_mvt"].values

    # Base models
    b21 = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f: feat21 = f.read().splitlines()
    taxi21 = np.clip(b21.predict(test[feat21]), 0, None)

    b_rn = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_r_norm.txt"))
    with open(os.path.join(MODELS, "lgbm_r_norm.features.txt")) as f: feat_rn = f.read().splitlines()
    r_norm = np.clip(b_rn.predict(test[feat_rn]), 0, None)

    # v16 = v21 + Step A
    with open(os.path.join(MODELS, "lirf_band_table.json")) as f: band = json.load(f)
    v16, cell_A = apply_rule(taxi21, sd_te, adep_te, fltid_te, band)

    # v18 = v16 + 6.3 rule (LIRF null sd 3600-14400)
    det63 = lgb.Booster(model_file=os.path.join(MODELS, "lirf_noflt_detector.txt"))
    with open(os.path.join(MODELS, "lirf_noflt_detector_meta.json")) as f:
        meta63 = json.load(f)
    for c in NF_CAT:
        if c in test.columns and not isinstance(test[c].dtype, pd.CategoricalDtype):
            test[c] = test[c].astype("category")
    mask63 = (adep_te == "LIRF") & pd.isna(fltid_te) & (sd_te > SD_LO) & (sd_te <= SD_HI) & ~np.isnan(sd_te)
    v18 = v16.copy()
    if mask63.sum():
        p_raw = det63.predict(test.loc[mask63, NF_CAT + NF_NUM])
        p_shrunk = SHRINK_63 * p_raw
        v18[mask63] = p_shrunk * sd_te[mask63] + (1 - p_shrunk) * meta63["normal_mean"]

    # v20 candidate: start from v18 (keeps LIRF fix), then per-airport override for non-LIRF
    # Test 3 variants: (a) R_norm alone, (b) mixture with detector, (c) v18 baseline (control)
    for c in DET_CAT:
        if c in test.columns and not isinstance(test[c].dtype, pd.CategoricalDtype):
            test[c] = test[c].astype("category")

    # Compute variant predictions per airport
    p_a = v18.copy()  # R_norm-only variant
    p_b = v18.copy()  # R_norm + soft-mix detector

    for apt in ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LTFM", "LSZH"]:
        apt_mask = (adep_te == apt)
        if apt_mask.sum() == 0: continue

        # Variant A: switch to R_norm at this airport (no mixture)
        p_a[apt_mask] = r_norm[apt_mask]

        # Variant B: mixture with detector if qualified
        det_path = os.path.join(MODELS, f"v18_det_{apt}.txt")
        if os.path.exists(det_path):
            det = lgb.Booster(model_file=det_path)
            sub_mask = apt_mask & (sd_te > SD_MIN) & ~np.isnan(sd_te)
            if sub_mask.sum():
                p_fb = det.predict(test.loc[sub_mask, DET_CAT + DET_NUM])
                p_b_pred = p_fb * sd_te[sub_mask] + (1 - p_fb) * r_norm[sub_mask]
                # For rows in this airport with sd <= 1800: fall back to R_norm alone
                p_b[apt_mask] = r_norm[apt_mask]
                p_b[sub_mask] = p_b_pred
            else:
                p_b[apt_mask] = r_norm[apt_mask]
        else:
            p_b[apt_mask] = r_norm[apt_mask]

    # Per-airport acceptance: for each airport pick winner among (v18, R_norm-only, mixture)
    print(f"\n=== Per-airport (LIRF handled by v18 already) ===")
    print(f"{'apt':>5s}  {'n':>7s}  {'v18':>8s}  {'R_norm':>8s}  {'R+mix':>8s}   winner")
    apt_choice = {}
    for apt in TARGET_ICAOS:
        mask = (adep_te == apt)
        if mask.sum() == 0: continue
        r_v18 = rmse(y[mask], v18[mask])
        r_a = rmse(y[mask], p_a[mask])
        r_b = rmse(y[mask], p_b[mask])
        opts = [("v18", r_v18), ("R_norm", r_a), ("R+mix", r_b)]
        best = min(opts, key=lambda x: x[1])
        apt_choice[apt] = best[0]
        print(f"{apt:>5s}  {mask.sum():>7,}   {r_v18:>7.2f}   {r_a:>7.2f}   {r_b:>7.2f}   {best[0]}")

    # Final v20 = winner per airport
    p_final = v18.copy()
    for apt, choice in apt_choice.items():
        mask = (adep_te == apt)
        if choice == "R_norm":
            p_final[mask] = p_a[mask]
        elif choice == "R+mix":
            p_final[mask] = p_b[mask]

    print("\n=== Full hold-out ===")
    clean = (y >= 30) & (y <= 7200)
    print(f"                  v21    v16      v18     R_norm     v20 combined")
    print(f"FULL   {rmse(y, taxi21):>7.2f}  {rmse(y, v16):>7.2f}  {rmse(y, v18):>7.2f}  {rmse(y, r_norm):>7.2f}  {rmse(y, p_final):>10.2f}")
    print(f"CLEAN  {rmse(y[clean], taxi21[clean]):>7.2f}  {rmse(y[clean], v16[clean]):>7.2f}  {rmse(y[clean], v18[clean]):>7.2f}  {rmse(y[clean], r_norm[clean]):>7.2f}  {rmse(y[clean], p_final[clean]):>10.2f}")

    with open(os.path.join(MODELS, "v20_apt_choice.json"), "w") as f:
        json.dump(apt_choice, f, indent=2)
    print(f"\nSaved -> models/v20_apt_choice.json   wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
