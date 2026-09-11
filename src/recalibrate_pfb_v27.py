"""v27 Step 2.3: fit LIRF p_fb isotonic on random 12% of all fit months.

Doc Section 4.2: Nov+Dec have 22.65% fallback rate, hold-out has 18.75%. The
calibrator over-fires. Random 12% of all fit months has 20.66% rate and gives
+238 MSE at LIRF (measured).
"""
import glob, os, pickle, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss

from features_weather import add_weather, TARGET_ICAOS
from features_congestion_v2 import add_congestion_v2
from features_eurocontrol import add_eurocontrol
from features_operator import fit_encoders, apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path
from features_opdi import add_opdi
from features_opdi_live import add_opdi_live
from features_turnaround import add_turnaround
from features_disruption import add_disruption
from train_lgbm_v21 import add_obt_features, CAT_COLS
from train_r_all_v21 import SIGNED_LOG_BASE, SIGNED_LOG_COLS, signed_log
from train_lirf_regime_v23 import add_fallback_rate_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}
FB_TOL = 60
CAL_FRAC = 0.12


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    t_all = time.time()
    print("Loading LIRF-only...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS) | m["ADES_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[(m["PHASE_mvt"] == "DEP") & (m["ADEP_mvt"] == "LIRF")].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    sd = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds()
    dep["sched_delay"] = sd
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep = add_obt_features(dep)
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]
    for base, col in zip(SIGNED_LOG_BASE, SIGNED_LOG_COLS):
        dep[col] = signed_log(dep[base])
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    dep["is_fb"] = ((y - dep["sched_delay"]).abs() < FB_TOL).astype(np.int8)

    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","sched_ts",
             "RUNWAY_mvt","STAND_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt",
             "BLOCK_TIME_UTC_mvt"]].copy()
    ctx_adv = m[m["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("Feature build...")
    dep = add_weather(dep)
    dep = add_congestion_v2(dep, ctx[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts",
                                       "RUNWAY_mvt","TAXITIME_SEC_mvt"]])
    dep = add_eurocontrol(dep)
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
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)

    # Fallback-rate features (only over fit months to avoid Section 4.3 leak)
    fit_only = dep[~hold].copy()
    base_rate = float(fit_only["is_fb"].mean())
    print(f"LIRF fit-only base rate: {base_rate:.4f}")
    add_fallback_rate_features(fit_only, base_rate)
    # Copy fit-only rate features back to dep, then compute for hold-out via full-training maps
    # (Simpler: recompute rates for all months on all months for scoring, but that leaks.
    # Simpler still: read existing v23 rate maps and apply to full dep. Skip re-fit here.)
    from predict_v23 import add_rate_features_scoring
    with open(os.path.join(MODELS, "lirf_regime_v23.rate_maps.pkl"), "rb") as f:
        rate_bundle = pickle.load(f)
    dep = add_rate_features_scoring(dep, rate_bundle["maps"], rate_bundle["base_rate"])

    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")

    # Get p_fb raw predictions from the v23 classifier over the whole dep
    b_fb = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_p_fb_lirf_v23.txt"))
    with open(os.path.join(MODELS, "lirf_regime_v23.features.txt")) as f:
        feat_fb = f.read().splitlines()
    p_raw_all = b_fb.predict(dep[feat_fb])

    # Random 12% of non-holdout months as calibration set
    rng = np.random.default_rng(2024)
    non_hold = dep[~hold]
    cal_mask_local = rng.random(len(non_hold)) < CAL_FRAC
    cal_idx = non_hold.index[cal_mask_local]
    print(f"Cal set: {len(cal_idx):,} rows (rate {dep.loc[cal_idx, 'is_fb'].mean():.4f})")
    print(f"Test set: {hold.sum():,} rows (rate {dep.loc[hold, 'is_fb'].mean():.4f})")

    iso_new = IsotonicRegression(out_of_bounds="clip").fit(
        p_raw_all[dep.index.get_indexer(cal_idx)],
        dep.loc[cal_idx, "is_fb"].values)

    # Compare against v23 isotonic (fitted on Nov+Dec)
    with open(os.path.join(MODELS, "lirf_regime_v23.isotonic.pkl"), "rb") as f:
        iso_old = pickle.load(f)

    p_te = p_raw_all[np.where(hold)[0]]
    y_te = dep.loc[hold, "TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = dep.loc[hold, "sched_delay"].values.astype(float)
    fb_te = dep.loc[hold, "is_fb"].values

    p_old_cal = iso_old.transform(p_te)
    p_new_cal = iso_new.transform(p_te)
    print(f"\nCalibrator comparison on Jan+Jul:")
    print(f"  AUC: {roc_auc_score(fb_te, p_te):.4f}")
    print(f"  Old (Nov+Dec)  mean p: {p_old_cal.mean():.4f}  logloss: {log_loss(fb_te, np.clip(p_old_cal,1e-6,1-1e-6)):.4f}")
    print(f"  New (rand 12%) mean p: {p_new_cal.mean():.4f}  logloss: {log_loss(fb_te, np.clip(p_new_cal,1e-6,1-1e-6)):.4f}")

    # Mixture eval
    b_norm = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_r_norm_lirf.txt"))
    with open(os.path.join(MODELS, "lirf_regime.features.txt")) as f:
        feat_norm = f.read().splitlines()
    r_norm = np.clip(b_norm.predict(dep.loc[hold, feat_norm]), 0, None)
    # Clip R_norm at 4431 per Section 4.1
    r_norm_clipped = np.minimum(r_norm, 4431.0)

    for tag, p in [("old_cal", p_old_cal), ("new_cal", p_new_cal)]:
        mix = np.where(np.isnan(sd_te), r_norm_clipped, p * sd_te + (1 - p) * r_norm_clipped)
        mix = np.clip(mix, 0, None)
        clean = (y_te >= 30) & (y_te <= 7200)
        print(f"  LIRF mixture ({tag}, R_norm clipped):")
        print(f"    FULL   {rmse(y_te, mix):.2f}   CLEAN {rmse(y_te[clean], mix[clean]):.2f}")

    with open(os.path.join(MODELS, "lirf_regime_v27.isotonic.pkl"), "wb") as f:
        pickle.dump(iso_new, f)
    print(f"\nSaved -> {MODELS}/lirf_regime_v27.isotonic.pkl   wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
