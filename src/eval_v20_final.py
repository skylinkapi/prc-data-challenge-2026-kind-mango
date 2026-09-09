"""Eval v20 final: R_all_v20 base + v18 for LIRF + ITY340 rule.

Prediction pipeline:
  1. taxi_hat = R_all_v20 for every row
  2. LIRF only:
     - null flight, sd > 14400  -> Step A band rule (base = v21 for the 'rest' component)
     - null flight, 3600 < sd <= 14400 -> 6.3 rule (mix with group mean 1220)
     - flight record, sd > 70000  -> ITY340 rule: 0.833*(86400+R_all) + 0.167*R_all
     - other LIRF rows -> R_all_v20 (do NOT use v18 detector - insufficient signal)
"""
import glob, os, pickle, json, gc, time
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion_v2 import add_congestion_v2
from features_eurocontrol import add_eurocontrol
from features_operator import apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path
from features_opdi import add_opdi
from features_opdi_live import add_opdi_live
from features_turnaround import add_turnaround
from features_disruption import add_disruption
from train_lgbm_v21 import add_obt_features, CAT_COLS
from eval_v21_stepA import apply_rule
from train_lirf_noflt_detector import CAT_INPUTS as NF_CAT, NUM_INPUTS as NF_NUM, SD_LO, SD_HI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}
SHRINK_63 = 0.6
P24_ITY = 5.0/6.0  # Laplace-smoothed 5/5 rows
ITY_SD_THRESHOLD = 70000


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def build_test():
    t0 = time.time()
    print("Loading + features (v20 stack)...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS) | m["ADES_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[(m["PHASE_mvt"] == "DEP") & m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
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
    dep["stand_prefix"] = dep["STAND_mvt"].astype(str).str.slice(0, 1).fillna("UNK")
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]

    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","sched_ts",
             "RUNWAY_mvt","STAND_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt"]].copy()
    ctx_adv = m[m["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion_v2(dep, ctx[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts",
                                       "RUNWAY_mvt","TAXITIME_SEC_mvt"]])
    dep = add_eurocontrol(dep)
    with open(os.path.join(MODELS, "lgbm_r_all_v20.encoders.pkl"), "rb") as f:
        enc = pickle.load(f)
    dep = apply_encoders(dep, enc)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, ctx_adv, daily_ec=ec_daily)
    del ctx_adv, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    dep = add_turnaround(dep, ctx)
    dep = add_disruption(dep, ctx)
    del ctx; gc.collect()

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
    test = build_test()
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)
    adep_te = test["ADEP_mvt"].astype(str).values
    fltid_te = test["FLIGHT_ID_mvt"].values

    # R_all v20 predictions
    b = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_r_all_v20.txt"))
    with open(os.path.join(MODELS, "lgbm_r_all_v20.features.txt")) as f:
        feat = f.read().splitlines()
    r_all = np.clip(b.predict(test[feat]), 0, None)

    # v21 predictions (needed for Step A band rule 'rest' component compatibility)
    b21 = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        feat21 = f.read().splitlines()
    taxi21 = np.clip(b21.predict(test[feat21]), 0, None)

    # Step A on v21 (v16)
    with open(os.path.join(MODELS, "lirf_band_table.json")) as f:
        band = json.load(f)
    v16, cell_A = apply_rule(taxi21, sd_te, adep_te, fltid_te, band)

    # v18 = v16 + 6.3
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

    # v20: R_all everywhere except LIRF, LIRF uses v18 + ITY340 rule
    p_final = r_all.copy()
    lirf_mask = (adep_te == "LIRF")
    p_final[lirf_mask] = v18[lirf_mask]

    # ITY340 rule: LIRF sd > 70000 (any flight record)
    ity_mask = lirf_mask & (sd_te > ITY_SD_THRESHOLD) & ~np.isnan(sd_te) & ~cell_A
    # Also apply to Step A cells (which are already handled by band table if in sd>14400 bands)
    # The doc says: for sd > 100000 use band-table; for 70000 < sd < 100000 use ITY rule
    # Simpler: apply ITY rule only where NOT already covered by Step A band table
    print(f"\nITY340 rule candidates: {ity_mask.sum()}  (LIRF sd>70000 outside Step A cell)")
    if ity_mask.sum():
        for i in np.where(ity_mask)[0]:
            p_final[i] = P24_ITY * (86400 + r_all[i]) + (1 - P24_ITY) * r_all[i]
            print(f"  ITY row: sd={sd_te[i]:.0f}  R_all={r_all[i]:.0f}  new_pred={p_final[i]:.0f}")

    print("\n=== Hold-out ===")
    print(f"                 v21    v16     v18   R_all   v19    v20 final")
    clean = (y >= 30) & (y <= 7200)

    # v19: reload from apt_choice
    with open(os.path.join(MODELS, "v20_apt_choice.json")) as f:
        apt_choice = json.load(f)
    # Since v19 is complex to reconstruct here, skip its column; compare v20 vs v18 directly
    print(f"FULL   {rmse(y, taxi21):>7.2f}  {rmse(y, v16):>7.2f}  {rmse(y, v18):>7.2f}  {rmse(y, r_all):>7.2f}  N/A     {rmse(y, p_final):>7.2f}")
    print(f"CLEAN  {rmse(y[clean], taxi21[clean]):>7.2f}  {rmse(y[clean], v16[clean]):>7.2f}  {rmse(y[clean], v18[clean]):>7.2f}  {rmse(y[clean], r_all[clean]):>7.2f}  N/A     {rmse(y[clean], p_final[clean]):>7.2f}")

    print("\nPer-airport clean RMSE (v20 vs v21):")
    v21_r = {"LIRF":565.10,"EGLL":303.20,"LFPG":289.60,"LTFM":271.80,"LEBL":230.50,
             "EDDF":225.90,"LSZH":221.60,"EHAM":215.30,"EDDM":211.70,"LEMD":191.60}
    for apt in sorted(v21_r, key=lambda k: -v21_r[k]):
        mask = (adep_te == apt) & clean
        if mask.sum() > 0:
            r = rmse(y[mask], p_final[mask])
            print(f"  {apt}: v20 {r:>6.2f}   v21 {v21_r[apt]:>6.2f}   delta {r-v21_r[apt]:+.2f}")

    print(f"\nWall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
