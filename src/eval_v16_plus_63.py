"""Eval: v16 (v21+Step A) alone vs v16 + Section 6.3 (LIRF no-flight-record group).
Reports clean/full/cell RMSE. Also tests calibrated variants of the detector.
"""
import glob, os, pickle, json, gc, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

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
from train_lirf_noflt_detector import CAT_INPUTS as DET_CAT, NUM_INPUTS as DET_NUM, SD_LO, SD_HI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
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
    stop_test = dep.loc[dep["month"].isin({11, 12})].copy()
    for c in CAT_COLS:
        test[c] = pd.Categorical(test[c], categories=train_cats[c])
    del dep; gc.collect()

    b = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        feat = f.read().splitlines()
    taxi_hat = np.clip(b.predict(test[feat]), 0, None)
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)
    adep_te = test["ADEP_mvt"].astype(str).values
    fltid_te = test["FLIGHT_ID_mvt"].values

    with open(os.path.join(MODELS, "lirf_band_table.json")) as f:
        table_data = json.load(f)
    p_A, cell_A = apply_rule(taxi_hat, sd_te, adep_te, fltid_te, table_data)

    # 6.3 rule: LIRF, null FLIGHT_ID, 3600 < sd <= 14400  (disjoint from Step A cell)
    det = lgb.Booster(model_file=os.path.join(MODELS, "lirf_noflt_detector.txt"))
    with open(os.path.join(MODELS, "lirf_noflt_detector_meta.json")) as f:
        meta = json.load(f)
    normal_mean = meta["normal_mean"]

    mask63 = ((adep_te == "LIRF") & pd.isna(fltid_te) &
              (sd_te > SD_LO) & (sd_te <= SD_HI) & ~np.isnan(sd_te))
    print(f"\n6.3 cell rows in test: {mask63.sum()}")

    # Cast test-frame to the detector's expected cats
    for c in DET_CAT:
        # Rebuild category set from the training set (Feb-Jun+Aug-Oct)
        pass  # we simply use raw column values; LightGBM stores its own cat map

    for c in DET_CAT:
        test[c] = test[c].astype("category")

    if mask63.sum() > 0:
        p_fb_raw = det.predict(test.loc[mask63, DET_CAT + DET_NUM])
        # Calibrator: fit isotonic on stop set (Nov+Dec)
        stop_test["flt_prefix"] = stop_test["flt_prefix"].astype("category")
        stop_test["stand_prefix"] = stop_test["stand_prefix"].astype("category")
        stop_test["AIRCRAFT_TYPE_mvt"] = stop_test["AIRCRAFT_TYPE_mvt"].astype("category")
        stop_test["RUNWAY_mvt"] = stop_test["RUNWAY_mvt"].astype("category")
        mask63_stop = ((stop_test["ADEP_mvt"].astype(str) == "LIRF") & stop_test["FLIGHT_ID_mvt"].isna() &
                       (stop_test["sched_delay"] > SD_LO) & (stop_test["sched_delay"] <= SD_HI))
        stop_sub = stop_test.loc[mask63_stop].copy()
        p_stop = det.predict(stop_sub[DET_CAT + DET_NUM])
        y_stop_fb = ((stop_sub["TAXITIME_SEC_mvt"].astype(float) - stop_sub["sched_delay"]).abs() < 60).astype(int).values
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_stop, y_stop_fb)
        p_fb_cal = iso.transform(p_fb_raw)
        print(f"  stop-set rows for calibration: {len(stop_sub)}")
        print(f"  raw p_fb  mean {p_fb_raw.mean():.3f}  test obs share {((y[mask63]-sd_te[mask63]).__abs__() < 60).mean():.3f}")
        print(f"  cal p_fb  mean {p_fb_cal.mean():.3f}")

    # Variants: raw, calibrated, threshold, hard
    variants = {"none": p_A.copy()}
    if mask63.sum() > 0:
        for tag, p_fb in [("raw", p_fb_raw), ("cal_iso", p_fb_cal),
                          ("shrink0.6", 0.6 * p_fb_raw),
                          ("shrink0.5", 0.5 * p_fb_raw)]:
            pv = p_A.copy()
            pv[mask63] = p_fb * sd_te[mask63] + (1 - p_fb) * normal_mean
            variants[tag] = pv

    print("\n=== Full hold-out ===")
    print(f"{'variant':>12s}    CLEAN     CELL63_MSE     FULL")
    clean = (y >= 30) & (y <= 7200)
    for tag, pv in variants.items():
        cell_mse = float(np.sum((y[mask63]-pv[mask63])**2)) if mask63.sum() else 0.0
        print(f"{tag:>12s}   {rmse(y[clean], pv[clean]):>6.2f}   {cell_mse:>12.0f}   {rmse(y, pv):>7.2f}")
    print(f"\nWall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
