"""Step 3 part 3: widen p_fb training to all 10 airports.

The LIRF-only gate had 110,780 rows. Widening to all 10 airports gives 1.9 M
rows. Handling-agent structure is shared across airports, so more data helps
the low-rate keys. Airport enters as a categorical so the model can learn per-
airport calibration internally.

Read predictions only at LIRF at inference. Retire the LIRF-only p_fb.
"""
import glob, os, pickle, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss

from features_weather import add_weather, TARGET_ICAOS
from features_congestion_v2 import add_congestion_v2, CONG_V2_NUM_COLS
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS
from features_advanced import add_advanced, ADV_NUM_COLS
from features_osm_path import add_osm_path, OSM_PATH_NUM_COLS
from features_opdi import add_opdi, OPDI_NUM_COLS
from features_opdi_live import add_opdi_live, OPDI_LIVE_NUM_COLS
from features_turnaround import add_turnaround, TURN_NUM_COLS
from features_disruption import add_disruption, DISR_NUM_COLS
from train_lgbm_v21 import add_obt_features, CAT_COLS, BEST_PARAMS
from train_lgbm_v23 import BASE_NUM_COLS
from train_r_all_v21 import SIGNED_LOG_BASE, SIGNED_LOG_COLS, TEMPERATURE_COLS, signed_log
from train_lirf_regime_v23 import add_fallback_rate_features, RATE_KEYS, RATE_FEATURES, K_SMOOTH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}
EARLYSTOP_MONTHS = {11, 12}
TRAIN_MONTHS = {2, 3, 4, 5, 6, 8, 9, 10}
FB_TOL = 60


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    t_all = time.time()
    print("Loading all 10 airports...")
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
    t = time.time()
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
    print(f"  Done in {time.time()-t:.1f}s")

    # Fallback-rate features: pool across all airports (per-airport smoothing to global base)
    print("Computing OOF fallback-rate encodings (global base rate)...")
    base_rate = float(dep["is_fb"].mean())
    print(f"Global base rate: {base_rate:.4f}")
    dep["AIRCRAFT_OPERATOR_flt"] = dep["AIRCRAFT_OPERATOR_flt"]  # already col
    scoring_maps = add_fallback_rate_features(dep, base_rate)

    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")

    train = dep[dep["month"].isin(TRAIN_MONTHS)].copy()
    stop = dep[dep["month"].isin(EARLYSTOP_MONTHS)].copy()
    test = dep[dep["month"].isin(HOLDOUT_MONTHS)].copy()
    print(f"All-apt: train {len(train):,}  stop {len(stop):,}  test {len(test):,}")
    print(f"  fb pos: train {train['is_fb'].sum():,}  stop {stop['is_fb'].sum():,}  test {test['is_fb'].sum():,}")
    for c in CAT_COLS:
        stop[c] = pd.Categorical(stop[c], categories=train[c].cat.categories)
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = (CAT_COLS + BASE_NUM_COLS + SIGNED_LOG_COLS + TEMPERATURE_COLS +
            CONG_V2_NUM_COLS + ec_cols + op_cols +
            TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS +
            OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS + TURN_NUM_COLS + DISR_NUM_COLS +
            RATE_FEATURES)
    feat = [c for c in feat if c in train.columns]
    print(f"Feature count: {len(feat)}")

    dt_c = lgb.Dataset(train[feat], label=train["is_fb"].values,
                       categorical_feature=CAT_COLS, free_raw_data=True)
    dv_c = lgb.Dataset(stop[feat], label=stop["is_fb"].values,
                       categorical_feature=CAT_COLS, reference=dt_c, free_raw_data=True)
    params = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 5,
              "verbosity": -1, "num_threads": -1}
    t0 = time.time()
    b = lgb.train(params, dt_c, num_boost_round=3000,
                  valid_sets=[dv_c], valid_names=["stop"],
                  callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])
    print(f"  best iter {b.best_iteration}  time {time.time()-t0:.1f}s")

    # Calibrate per airport on Nov+Dec
    p_stop = b.predict(stop[feat], num_iteration=b.best_iteration)
    stop_apt = stop["ADEP_mvt"].astype(str).values
    isos_per_apt = {}
    for apt in TARGET_ICAOS:
        mask = stop_apt == apt
        if mask.sum() < 50 or stop.loc[mask, "is_fb"].nunique() < 2:
            isos_per_apt[apt] = None
            continue
        isos_per_apt[apt] = IsotonicRegression(out_of_bounds="clip").fit(
            p_stop[mask], stop.loc[mask, "is_fb"].values)

    # Read test only at LIRF
    lirf_test = test[test["ADEP_mvt"].astype(str) == "LIRF"].copy()
    print(f"\n=== LIRF-only test evaluation ===")
    print(f"LIRF test rows: {len(lirf_test):,}   fb pos: {lirf_test['is_fb'].sum():,}")
    p_raw = b.predict(lirf_test[feat], num_iteration=b.best_iteration)
    iso_lirf = isos_per_apt.get("LIRF")
    p_cal = iso_lirf.transform(p_raw) if iso_lirf is not None else p_raw
    print(f"  AUC {roc_auc_score(lirf_test['is_fb'], p_raw):.4f}  "
          f"logloss {log_loss(lirf_test['is_fb'], np.clip(p_raw,1e-6,1-1e-6)):.4f}")

    # Test-set mixture using v22 R_norm_LIRF + wide p_fb
    print("Loading R_norm_LIRF and computing mixture...")
    b_norm = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_r_norm_lirf.txt"))
    with open(os.path.join(MODELS, "lirf_regime.features.txt")) as f:
        feat_norm = f.read().splitlines()
    r_norm = np.clip(b_norm.predict(lirf_test[feat_norm]), 0, None)
    y_te = lirf_test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = lirf_test["sched_delay"].values.astype(float)
    mix = np.where(np.isnan(sd_te), r_norm, p_cal * sd_te + (1 - p_cal) * r_norm)
    mix = np.clip(mix, 0, None)
    clean = (y_te >= 30) & (y_te <= 7200)
    genuine = clean & (np.abs(y_te - sd_te) >= FB_TOL)
    print(f"  LIRF FULL     {rmse(y_te, mix):.2f}")
    print(f"  LIRF CLEAN    {rmse(y_te[clean], mix[clean]):.2f}")
    print(f"  LIRF GENUINE  {rmse(y_te[genuine], mix[genuine]):.2f}")

    b.save_model(os.path.join(MODELS, "lgbm_p_fb_wide.txt"))
    with open(os.path.join(MODELS, "p_fb_wide.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(MODELS, "p_fb_wide.isotonics.pkl"), "wb") as f:
        pickle.dump(isos_per_apt, f)
    with open(os.path.join(MODELS, "p_fb_wide.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(MODELS, "p_fb_wide.rate_maps.pkl"), "wb") as f:
        pickle.dump({"maps": scoring_maps, "base_rate": base_rate, "keys": RATE_KEYS}, f)
    print(f"\nSaved -> {MODELS}/p_fb_wide.*  wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
