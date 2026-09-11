"""Step 3 part 2: joint LIRF regressor.

Train one LIRF regressor with linear_tree on ALL LIRF rows, including fallback,
using p_fb (OOF by month) and sd as inputs among others. A linear leaf can then
output p*sd + (1-p)*f(x) natively.

Steps:
  1. Load LIRF-only features (same stack as train_lirf_regime).
  2. Compute p_fb OOF by month (12 folds: train on 11 months, predict on 1).
  3. Train R_joint_LIRF with linear_tree, p_fb as feature, y as target.
  4. Compare against current v23 mixture on Jan+Jul hold-out.
"""
import glob, os, pickle, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

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
from train_lirf_regime_v23 import add_fallback_rate_features, RATE_KEYS, RATE_FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}
EARLYSTOP_MONTHS = {11, 12}
TRAIN_MONTHS = {2, 3, 4, 5, 6, 8, 9, 10}
FB_TOL = 60


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def build_lirf_features():
    print("Loading LIRF + context...")
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

    # Fallback-rate features
    base_rate = float(dep["is_fb"].mean())
    scoring_maps = add_fallback_rate_features(dep, base_rate)
    return dep, encoders, scoring_maps, base_rate


def compute_oof_pfb(dep, feat, params):
    """12-fold by month OOF p_fb over training months. Test months use full-training fit."""
    oof = np.full(len(dep), np.nan)
    all_months = sorted(dep["month"].unique())
    non_hold = [m for m in all_months if m not in HOLDOUT_MONTHS]
    print(f"OOF p_fb over {len(non_hold)} non-hold months...")

    for m in non_hold:
        train_mask = dep["month"].isin([mm for mm in non_hold if mm != m])
        pred_mask = dep["month"] == m
        if pred_mask.sum() == 0: continue
        dt = lgb.Dataset(dep.loc[train_mask, feat], label=dep.loc[train_mask, "is_fb"].values,
                         categorical_feature=CAT_COLS, free_raw_data=True)
        b = lgb.train(params, dt, num_boost_round=250, callbacks=[lgb.log_evaluation(0)])
        oof[pred_mask] = b.predict(dep.loc[pred_mask, feat])
        del b, dt; gc.collect()
    # Test months: train on all non-hold, predict on hold
    hold_mask = dep["month"].isin(HOLDOUT_MONTHS)
    train_mask = ~hold_mask
    dt = lgb.Dataset(dep.loc[train_mask, feat], label=dep.loc[train_mask, "is_fb"].values,
                     categorical_feature=CAT_COLS, free_raw_data=True)
    b = lgb.train(params, dt, num_boost_round=250, callbacks=[lgb.log_evaluation(0)])
    oof[hold_mask] = b.predict(dep.loc[hold_mask, feat])
    del b, dt; gc.collect()
    return oof


def main():
    t_all = time.time()
    dep, encoders, scoring_maps, base_rate = build_lirf_features()
    print(f"LIRF rows: {len(dep):,}   base rate {base_rate:.4f}")

    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat_pfb = (CAT_COLS + BASE_NUM_COLS + SIGNED_LOG_COLS + TEMPERATURE_COLS +
                CONG_V2_NUM_COLS + ec_cols + op_cols +
                TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS +
                OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS + TURN_NUM_COLS + DISR_NUM_COLS +
                RATE_FEATURES)
    feat_pfb = [c for c in feat_pfb if c in dep.columns]

    pfb_params = {"objective": "binary", "metric": "binary_logloss",
                  "learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 50,
                  "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 5,
                  "verbosity": -1, "num_threads": -1, "seed": 42}
    oof_pfb = compute_oof_pfb(dep, feat_pfb, pfb_params)
    dep["oof_pfb"] = oof_pfb

    # Now train R_joint_LIRF with linear_tree and oof_pfb as feature
    feat_joint = feat_pfb + ["oof_pfb"]
    print(f"\nR_joint feature count: {len(feat_joint)}")

    train = dep[dep["month"].isin(TRAIN_MONTHS)].copy()
    stop = dep[dep["month"].isin(EARLYSTOP_MONTHS)].copy()
    test = dep[dep["month"].isin(HOLDOUT_MONTHS)].copy()
    for c in CAT_COLS:
        stop[c] = pd.Categorical(stop[c], categories=train[c].cat.categories)
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    dt = lgb.Dataset(train[feat_joint], label=train["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS, free_raw_data=True)
    dv = lgb.Dataset(stop[feat_joint], label=stop["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS, reference=dt, free_raw_data=True)
    params_reg = {**BEST_PARAMS,
                  "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0,
                  "num_leaves": 127,
                  "bagging_freq": 5, "verbosity": -1, "num_threads": -1}
    t0 = time.time()
    b = lgb.train(params_reg, dt, num_boost_round=3000,
                  valid_sets=[dv], valid_names=["stop"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  R_joint best iter {b.best_iteration}  time {time.time()-t0:.1f}s")

    # Evaluate on test set
    p_te = np.clip(b.predict(test[feat_joint], num_iteration=b.best_iteration), 0, None)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)
    clean = (y_te >= 30) & (y_te <= 7200)
    genuine = clean & (np.abs(y_te - sd_te) >= FB_TOL)
    print(f"\n=== R_joint LIRF test evaluation ===")
    print(f"  FULL     {rmse(y_te, p_te):.2f}")
    print(f"  CLEAN    {rmse(y_te[clean], p_te[clean]):.2f}")
    print(f"  GENUINE  {rmse(y_te[genuine], p_te[genuine]):.2f}")

    # Feature importance for oof_pfb
    imp = pd.Series(b.feature_importance(importance_type="gain"),
                    index=feat_joint).sort_values(ascending=False)
    r = list(imp.index).index("oof_pfb") + 1
    print(f"  oof_pfb rank {r}/{len(feat_joint)}  gain {imp['oof_pfb']:.2e}")

    b.save_model(os.path.join(MODELS, "lgbm_r_joint_lirf.txt"))
    with open(os.path.join(MODELS, "r_joint_lirf.features.txt"), "w") as f:
        f.write("\n".join(feat_joint))
    # Train a final "scoring" p_fb on all non-hold months for the ranking set
    print("\nTraining final scoring p_fb (all non-hold months)...")
    non_hold_mask = ~dep["month"].isin(HOLDOUT_MONTHS)
    dt_final = lgb.Dataset(dep.loc[non_hold_mask, feat_pfb],
                           label=dep.loc[non_hold_mask, "is_fb"].values,
                           categorical_feature=CAT_COLS, free_raw_data=True)
    b_pfb = lgb.train(pfb_params, dt_final, num_boost_round=250,
                      callbacks=[lgb.log_evaluation(0)])
    b_pfb.save_model(os.path.join(MODELS, "lgbm_p_fb_lirf_scoring.txt"))
    with open(os.path.join(MODELS, "r_joint_lirf.pfb_features.txt"), "w") as f:
        f.write("\n".join(feat_pfb))
    with open(os.path.join(MODELS, "r_joint_lirf.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(MODELS, "r_joint_lirf.rate_maps.pkl"), "wb") as f:
        pickle.dump({"maps": scoring_maps, "base_rate": base_rate, "keys": RATE_KEYS}, f)
    print(f"\nSaved -> {MODELS}/r_joint_lirf.*  wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
