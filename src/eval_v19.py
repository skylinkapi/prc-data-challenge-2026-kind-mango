"""Eval v19: v18 architecture (R_norm + per-airport detectors + Step A + 6.3 rule).

Compare with v16 and v18 on Jan+Jul hold-out. Per-airport acceptance.

Mixture formula:
  p_final = p_fb * sd + p_24 * (86400 + R_norm) + (1 - p_fb - p_24) * R_norm

- For LIRF null-flight sd > 14400: p_fb, p_24 from Step A band table
- For LIRF null-flight 3600 < sd <= 14400: p_fb from Section 6.3 detector (shrink 0.6)
- For other rows in qualified airports with sd > 1800: p_fb from v18 detector
- For rows with no p_fb source: p_final = R_norm (Rows outside qualified airports also fall back to R_norm)

Per-airport acceptance: only deploy the mixture where it wins vs v16 on Jan+Jul.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}
SHRINK_63 = 0.6

QUALIFIED = ["LIRF", "LEBL", "EGLL", "LTFM"]


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def build_holdout():
    t0 = time.time()
    print("Loading + features...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    sd = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds()
    dep["sched_delay"] = sd
    dep["sd"] = sd
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep["flt_id_null"] = dep["FLIGHT_ID_mvt"].isna().astype(np.int8)
    dep = add_obt_features(dep)
    dep["flt_prefix"] = dep["FLIGHT_mvt"].astype(str).str[:3].fillna("UNK")
    dep["stand_prefix"] = dep["STAND_mvt"].astype(str).str.extract(r"^([A-Za-z]+)")[0].fillna("UNK")
    dep["ades_region"] = dep["ADES_mvt"].astype(str).str[:2].fillna("UNK")
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]

    m_ctx = m[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    with open(os.path.join(MODELS, "lgbm_v21.encoders.pkl"), "rb") as f:
        enc = pickle.load(f)
    dep = apply_encoders(dep, enc)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m_ctx, daily_ec=ec_daily)
    del m_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train_cats = {c: dep.loc[~hold, c].astype("category").cat.categories for c in CAT_COLS}
    test = dep.loc[hold].copy()
    for c in CAT_COLS:
        test[c] = pd.Categorical(test[c], categories=train_cats[c])
    del dep; gc.collect()
    print(f"  Ready {len(test):,}  ({time.time()-t0:.1f}s)")
    return test


def main():
    t0 = time.time()
    test = build_holdout()
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)
    adep_te = test["ADEP_mvt"].astype(str).values
    fltid_te = test["FLIGHT_ID_mvt"].values

    # v21 predictions (for v16 baseline)
    b21 = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        feat21 = f.read().splitlines()
    taxi21 = np.clip(b21.predict(test[feat21]), 0, None)

    # R_norm predictions
    b_rn = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_r_norm.txt"))
    with open(os.path.join(MODELS, "lgbm_r_norm.features.txt")) as f:
        feat_rn = f.read().splitlines()
    r_norm = np.clip(b_rn.predict(test[feat_rn]), 0, None)

    # Step A band rule (band table on v21)
    with open(os.path.join(MODELS, "lirf_band_table.json")) as f:
        band = json.load(f)
    p_A_on_v21, cell_A = apply_rule(taxi21, sd_te, adep_te, fltid_te, band)

    # v16 baseline
    v16 = p_A_on_v21

    # v19 mixture pieces
    p_fb = np.zeros(len(y))
    p_24 = np.zeros(len(y))

    # Step A cell: LIRF null sd>14400 -> use band table probabilities directly
    for i in np.where(cell_A)[0]:
        s = sd_te[i]
        for (lo, hi), band_id in [((lo_,hi_), f"{lo_}_{hi_}") for lo_,hi_ in band["bands"]]:
            if lo < s <= hi and band_id in band["table"]:
                p_fb[i] = band["table"][band_id]["p_fb"]
                p_24[i] = band["table"][band_id]["p_24h"]
                break

    # 6.3 rule: LIRF null 3600 < sd <= 14400
    det63 = lgb.Booster(model_file=os.path.join(MODELS, "lirf_noflt_detector.txt"))
    with open(os.path.join(MODELS, "lirf_noflt_detector_meta.json")) as f:
        meta63 = json.load(f)
    for c in NF_CAT:
        test[c] = test[c].astype("category") if c in test.columns else test[c]
    mask63 = (adep_te == "LIRF") & pd.isna(fltid_te) & (sd_te > SD_LO) & (sd_te <= SD_HI) & ~np.isnan(sd_te)
    if mask63.sum():
        p_raw = det63.predict(test.loc[mask63, NF_CAT + NF_NUM])
        p_fb[mask63] = SHRINK_63 * p_raw

    # Per-airport v18 detectors
    for apt in QUALIFIED:
        det = lgb.Booster(model_file=os.path.join(MODELS, f"v18_det_{apt}.txt"))
        for c in DET_CAT:
            if c in test.columns and not isinstance(test[c].dtype, pd.CategoricalDtype):
                test[c] = test[c].astype("category")
        # apply to airport rows with sd > SD_MIN, excluding rows already covered above
        mask_apt = (adep_te == apt) & (sd_te > SD_MIN) & ~np.isnan(sd_te) & (p_fb == 0)
        if mask_apt.sum() == 0: continue
        p_raw = det.predict(test.loc[mask_apt, DET_CAT + DET_NUM])
        p_fb[mask_apt] = p_raw

    # Clip
    p_any = np.clip(p_fb + p_24, 0, 1)
    p_fb_use = p_fb * (1.0 - p_24 / (p_any + 1e-9))
    # Simpler: p_fb+p_24 already <= 1 by construction. Just clip to [0,1] jointly
    p_norm = np.clip(1 - p_fb - p_24, 0, 1)

    # Mixture
    p_mix = p_fb * np.where(np.isnan(sd_te), 0, sd_te) + p_24 * (86400 + r_norm) + p_norm * r_norm
    p_mix = np.where(np.isnan(sd_te) & (p_fb > 0), r_norm, p_mix)  # safety
    p_mix = np.clip(p_mix, 0, None)

    # Per-airport acceptance vs v16
    print("\n=== Per-airport acceptance ===")
    print(f"{'apt':>5s}  {'n':>7s}  {'v16_MSE':>13s}  {'v19_MSE':>13s}  {'delta_MSE':>12s}  deploy")
    deploy = {}
    for apt in TARGET_ICAOS:
        mask = (adep_te == apt)
        if mask.sum() == 0: continue
        m_v16 = float(np.sum((y[mask] - v16[mask])**2))
        m_v19 = float(np.sum((y[mask] - p_mix[mask])**2))
        wins = m_v19 < m_v16
        deploy[apt] = wins
        star = "*" if wins else " "
        print(f"{apt:>5s}  {mask.sum():>7,}  {m_v16:>13,.0f}  {m_v19:>13,.0f}  {m_v19-m_v16:>+12,.0f}  {star}")

    # Final v19 prediction: use mixture only at accepted airports
    p_final = v16.copy()
    for apt, ok in deploy.items():
        if ok:
            mask = (adep_te == apt)
            p_final[mask] = p_mix[mask]

    print("\n=== Full hold-out ===")
    print(f"                            v21    v16 (A)    v18 (A+6.3)   R_norm   v19 mixture   v19 gated")
    clean = (y >= 30) & (y <= 7200)
    # v18 for comparison
    v18 = v16.copy()
    if mask63.sum():
        p_raw = det63.predict(test.loc[mask63, NF_CAT + NF_NUM])
        p_shrunk = SHRINK_63 * p_raw
        pred = p_shrunk * sd_te[mask63] + (1 - p_shrunk) * meta63["normal_mean"]
        v18[mask63] = pred
    print(f"FULL   {rmse(y, taxi21):>7.2f}   {rmse(y, v16):>7.2f}   {rmse(y, v18):>9.2f}   {rmse(y, r_norm):>7.2f}   {rmse(y, p_mix):>9.2f}   {rmse(y, p_final):>9.2f}")
    print(f"CLEAN  {rmse(y[clean], taxi21[clean]):>7.2f}   {rmse(y[clean], v16[clean]):>7.2f}   {rmse(y[clean], v18[clean]):>9.2f}   {rmse(y[clean], r_norm[clean]):>7.2f}   {rmse(y[clean], p_mix[clean]):>9.2f}   {rmse(y[clean], p_final[clean]):>9.2f}")

    with open(os.path.join(MODELS, "v19_deploy.json"), "w") as f:
        json.dump({"deploy": {k: bool(v) for k,v in deploy.items()},
                   "shrink_63": SHRINK_63, "qualified": QUALIFIED}, f, indent=2)
    print(f"\nSaved deploy config -> models/v19_deploy.json")
    print(f"Wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
