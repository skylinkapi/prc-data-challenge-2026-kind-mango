"""v22 Step 1: LIRF regime head.

Train two models on LIRF-only rows using the v21 (R_all_v21) feature stack:
  1. R_norm_LIRF: LightGBM regressor on LIRF genuine rows (|y-sd|>=60 AND y<80000)
  2. p_fb_LIRF: LightGBM binary classifier for |y-sd|<60 on ALL LIRF rows

Both split: train months 2-6 + 8-10, early-stop 11-12, evaluate Jan+Jul once.
Isotonic calibration for p_fb fitted on Nov+Dec.
"""
import glob, os, pickle, time, gc, json
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}
EARLYSTOP_MONTHS = {11, 12}
TRAIN_MONTHS = {2, 3, 4, 5, 6, 8, 9, 10}
FB_TOL = 60


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def build_features():
    print("Loading + engineering...")
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

    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","sched_ts",
             "RUNWAY_mvt","STAND_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt",
             "BLOCK_TIME_UTC_mvt"]].copy()
    ctx_adv = m[m["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

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
    return dep, encoders


def main():
    t_all = time.time()
    dep, encoders = build_features()

    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")

    train = dep[dep["month"].isin(TRAIN_MONTHS)].copy()
    stop = dep[dep["month"].isin(EARLYSTOP_MONTHS)].copy()
    test = dep[dep["month"].isin(HOLDOUT_MONTHS)].copy()
    print(f"LIRF: train {len(train):,}  stop {len(stop):,}  test {len(test):,}")

    for c in CAT_COLS:
        stop[c] = pd.Categorical(stop[c], categories=train[c].cat.categories)
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = (CAT_COLS + BASE_NUM_COLS + SIGNED_LOG_COLS + TEMPERATURE_COLS +
            CONG_V2_NUM_COLS + ec_cols + op_cols +
            TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS +
            OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS + TURN_NUM_COLS + DISR_NUM_COLS)
    feat = [c for c in feat if c in train.columns]
    print(f"Feature count: {len(feat)}")

    # ---- R_norm_LIRF: LIRF genuine rows ----
    print("\n=== R_norm_LIRF ===")
    y_all_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    sd_tr = train["sched_delay"].values.astype(float)
    genuine_tr = (np.abs(y_all_tr - sd_tr) >= FB_TOL) & (y_all_tr < 80000)
    y_all_st = stop["TAXITIME_SEC_mvt"].values.astype(float)
    sd_st = stop["sched_delay"].values.astype(float)
    genuine_st = (np.abs(y_all_st - sd_st) >= FB_TOL) & (y_all_st < 80000)

    tr_g = train[genuine_tr]
    st_g = stop[genuine_st]
    print(f"Genuine train: {len(tr_g):,}  stop: {len(st_g):,}")

    dt = lgb.Dataset(tr_g[feat], label=tr_g["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(st_g[feat], label=st_g["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS, reference=dt, free_raw_data=True)
    params_reg = {**BEST_PARAMS,
                  "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0,
                  "num_leaves": 127,  # smaller per doc
                  "bagging_freq": 5, "verbosity": -1, "num_threads": -1}
    t0 = time.time()
    b_norm = lgb.train(params_reg, dt, num_boost_round=3000,
                       valid_sets=[dv], valid_names=["stop"],
                       callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  R_norm_LIRF best iter {b_norm.best_iteration}  time {time.time()-t0:.1f}s")

    # ---- p_fb classifier: ALL LIRF rows ----
    print("\n=== p_fb classifier ===")
    is_fb_tr = (np.abs(y_all_tr - sd_tr) < FB_TOL).astype(np.int8)
    is_fb_st = (np.abs(y_all_st - sd_st) < FB_TOL).astype(np.int8)
    print(f"is_fb train pos {is_fb_tr.sum():,}/{len(train):,}   stop pos {is_fb_st.sum():,}/{len(stop):,}")

    dt_c = lgb.Dataset(train[feat], label=is_fb_tr,
                       categorical_feature=CAT_COLS, free_raw_data=True)
    dv_c = lgb.Dataset(stop[feat], label=is_fb_st,
                       categorical_feature=CAT_COLS, reference=dt_c, free_raw_data=True)
    params_cls = {"objective": "binary", "metric": "binary_logloss",
                  "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
                  "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 5,
                  "verbosity": -1, "num_threads": -1}
    t0 = time.time()
    b_fb = lgb.train(params_cls, dt_c, num_boost_round=2000,
                     valid_sets=[dv_c], valid_names=["stop"],
                     callbacks=[lgb.early_stopping(50, min_delta=0),
                                lgb.log_evaluation(100)])
    print(f"  p_fb best iter {b_fb.best_iteration}  time {time.time()-t0:.1f}s")

    # Isotonic calibration on Nov+Dec
    p_st_raw = b_fb.predict(stop[feat], num_iteration=b_fb.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_st_raw, is_fb_st)
    print(f"  Stop AUC {roc_auc_score(is_fb_st, p_st_raw):.4f}  "
          f"logloss {log_loss(is_fb_st, np.clip(p_st_raw,1e-6,1-1e-6)):.4f}")

    # ---- Test-set evaluation ----
    print("\n=== Test-set mixture (one shot) ===")
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)

    r_norm = np.clip(b_norm.predict(test[feat], num_iteration=b_norm.best_iteration), 0, None)
    p_fb_raw = b_fb.predict(test[feat], num_iteration=b_fb.best_iteration)
    p_fb_cal = iso.transform(p_fb_raw)

    clean = (y_te >= 30) & (y_te <= 7200)
    genuine_te = clean & (np.abs(y_te - sd_te) >= FB_TOL)

    for tag, p in [("raw", p_fb_raw), ("cal", p_fb_cal)]:
        mix = p * sd_te + (1 - p) * r_norm
        mix = np.clip(np.where(np.isnan(sd_te), r_norm, mix), 0, None)
        print(f"  LIRF mixture ({tag} probability):")
        print(f"    FULL         RMSE {rmse(y_te, mix):.2f}   n={len(y_te):,}")
        print(f"    CLEAN        RMSE {rmse(y_te[clean], mix[clean]):.2f}   n={clean.sum():,}")
        print(f"    GENUINE      RMSE {rmse(y_te[genuine_te], mix[genuine_te]):.2f}   n={genuine_te.sum():,}")

    # Save
    b_norm.save_model(os.path.join(MODELS, "lgbm_r_norm_lirf.txt"))
    b_fb.save_model(os.path.join(MODELS, "lgbm_p_fb_lirf.txt"))
    with open(os.path.join(MODELS, "lirf_regime.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(MODELS, "lirf_regime.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(MODELS, "lirf_regime.isotonic.pkl"), "wb") as f:
        pickle.dump(iso, f)
    print(f"\nSaved -> {MODELS}/lirf_regime.*  wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
