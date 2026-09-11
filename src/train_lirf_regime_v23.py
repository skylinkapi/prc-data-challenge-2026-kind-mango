"""v23 Step 2: LIRF regime head with fallback-rate encodings.

Adds 7 out-of-fold fallback-rate encoders (rate + count each = 14 features) to
the p_fb classifier. Retrains R_norm_LIRF unchanged.

OOF: leave-one-month-out, K=30 smoothing toward LIRF base rate.
Keys: operator, flt_prefix, stand, aircraft_type, operator_x_stand,
      stand_first_char, destination.
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
K_SMOOTH = 30

RATE_KEYS = ["op", "flt_prefix", "stand", "aircraft_type", "op_stand",
             "stand_first_char", "destination"]
RATE_FEATURES = []
for k in RATE_KEYS:
    RATE_FEATURES.append(f"fbrate_{k}")
    RATE_FEATURES.append(f"fbcount_{k}")


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def add_fallback_rate_features(dep_all, base_rate):
    """OOF leave-one-month-out fallback rate for each key.
    dep_all must contain: month, is_fb (0/1), and the raw key columns.
    Adds RATE_FEATURES columns in place.
    Returns full-training-fit maps for scoring-time use (ranking).
    """
    key_defs = {
        "op": dep_all["AIRCRAFT_OPERATOR_flt"].astype(str),
        "flt_prefix": dep_all["FLIGHT_mvt"].astype(str).str[:3].fillna("UNK"),
        "stand": dep_all["STAND_mvt"].astype(str),
        "aircraft_type": dep_all["AIRCRAFT_TYPE_mvt"].astype(str),
        "op_stand": dep_all["AIRCRAFT_OPERATOR_flt"].astype(str) + "|" +
                    dep_all["STAND_mvt"].astype(str),
        "stand_first_char": dep_all["STAND_mvt"].astype(str).str.slice(0, 1).fillna("UNK"),
        "destination": dep_all["ADES_mvt"].astype(str),
    }
    all_months = sorted(dep_all["month"].unique())
    scoring_maps = {}
    for k, keys in key_defs.items():
        rate = np.full(len(dep_all), base_rate, dtype=np.float32)
        cnt = np.zeros(len(dep_all), dtype=np.float32)
        # leave-one-month-out
        for m in all_months:
            train_mask = dep_all["month"] != m
            eval_mask = dep_all["month"] == m
            if eval_mask.sum() == 0: continue
            grp = pd.DataFrame({"key": keys[train_mask].values,
                                "fb": dep_all.loc[train_mask, "is_fb"].astype(float).values}) \
                .groupby("key").agg(s=("fb", "sum"), n=("fb", "size"))
            grp["smooth"] = (grp["s"] + K_SMOOTH * base_rate) / (grp["n"] + K_SMOOTH)
            rmap = grp["smooth"].to_dict()
            cmap = grp["n"].to_dict()
            eval_keys = keys[eval_mask].values
            rate[eval_mask] = np.array([rmap.get(kk, base_rate) for kk in eval_keys])
            cnt[eval_mask] = np.array([cmap.get(kk, 0) for kk in eval_keys])
        dep_all[f"fbrate_{k}"] = rate
        dep_all[f"fbcount_{k}"] = cnt
        # full-training map for scoring
        grp_all = pd.DataFrame({"key": keys.values,
                                "fb": dep_all["is_fb"].astype(float).values}) \
            .groupby("key").agg(s=("fb", "sum"), n=("fb", "size"))
        grp_all["smooth"] = (grp_all["s"] + K_SMOOTH * base_rate) / (grp_all["n"] + K_SMOOTH)
        scoring_maps[k] = {"rate": grp_all["smooth"].to_dict(),
                           "count": grp_all["n"].to_dict()}
    return scoring_maps


def build_features_lirf():
    print("Loading + engineering LIRF only...")
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
    dep, encoders = build_features_lirf()

    # Compute OOF fallback-rate features
    print("Computing OOF fallback-rate encodings...")
    base_rate = float(dep["is_fb"].mean())
    print(f"LIRF base rate: {base_rate:.4f}")
    scoring_maps = add_fallback_rate_features(dep, base_rate)

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
            OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS + TURN_NUM_COLS + DISR_NUM_COLS +
            RATE_FEATURES)
    feat = [c for c in feat if c in train.columns]
    print(f"Feature count: {len(feat)}  (v22 had 153, added {len(RATE_FEATURES)})")

    # p_fb classifier with new features
    print("\n=== p_fb_v23 classifier ===")
    dt_c = lgb.Dataset(train[feat], label=train["is_fb"].values,
                       categorical_feature=CAT_COLS, free_raw_data=True)
    dv_c = lgb.Dataset(stop[feat], label=stop["is_fb"].values,
                       categorical_feature=CAT_COLS, reference=dt_c, free_raw_data=True)
    params_cls = {"objective": "binary", "metric": "binary_logloss",
                  "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
                  "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 5,
                  "verbosity": -1, "num_threads": -1}
    t0 = time.time()
    b_fb = lgb.train(params_cls, dt_c, num_boost_round=2000,
                     valid_sets=[dv_c], valid_names=["stop"],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"  p_fb_v23 best iter {b_fb.best_iteration}  time {time.time()-t0:.1f}s")

    # Isotonic calibration on Nov+Dec
    p_st_raw = b_fb.predict(stop[feat], num_iteration=b_fb.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_st_raw, stop["is_fb"].values)
    print(f"  Stop AUC {roc_auc_score(stop['is_fb'], p_st_raw):.4f}  "
          f"logloss {log_loss(stop['is_fb'], np.clip(p_st_raw,1e-6,1-1e-6)):.4f}")

    # Feature importance
    imp = pd.Series(b_fb.feature_importance(importance_type="gain"),
                    index=feat).sort_values(ascending=False)
    print("\nNew rate-feature ranks:")
    for c in RATE_FEATURES:
        if c in imp.index:
            r = list(imp.index).index(c) + 1
            print(f"  {c:26s} rank {r:>3d}/{len(feat)}  gain {imp[c]:.2e}")

    # Test-set eval: reuse R_norm_LIRF from v22
    print("\n=== Test-set mixture (v22 R_norm_LIRF + v23 p_fb) ===")
    b_norm = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_r_norm_lirf.txt"))
    with open(os.path.join(MODELS, "lirf_regime.features.txt")) as f:
        feat_norm = f.read().splitlines()
    r_norm = np.clip(b_norm.predict(test[feat_norm]), 0, None)

    p_fb_raw = b_fb.predict(test[feat], num_iteration=b_fb.best_iteration)
    p_fb_cal = iso.transform(p_fb_raw)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)
    mix = np.where(np.isnan(sd_te), r_norm,
                   p_fb_cal * sd_te + (1 - p_fb_cal) * r_norm)
    mix = np.clip(mix, 0, None)
    clean = (y_te >= 30) & (y_te <= 7200)
    genuine = clean & (np.abs(y_te - sd_te) >= FB_TOL)
    print(f"    LIRF FULL     {rmse(y_te, mix):.2f}   n={len(y_te)}")
    print(f"    LIRF CLEAN    {rmse(y_te[clean], mix[clean]):.2f}   n={clean.sum()}")
    print(f"    LIRF GENUINE  {rmse(y_te[genuine], mix[genuine]):.2f}   n={genuine.sum()}")

    b_fb.save_model(os.path.join(MODELS, "lgbm_p_fb_lirf_v23.txt"))
    with open(os.path.join(MODELS, "lirf_regime_v23.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(MODELS, "lirf_regime_v23.isotonic.pkl"), "wb") as f:
        pickle.dump(iso, f)
    with open(os.path.join(MODELS, "lirf_regime_v23.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(MODELS, "lirf_regime_v23.rate_maps.pkl"), "wb") as f:
        pickle.dump({"maps": scoring_maps, "base_rate": base_rate,
                     "keys": RATE_KEYS}, f)
    print(f"\nSaved -> {MODELS}/lirf_regime_v23.*  wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
