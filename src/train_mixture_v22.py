"""Step 3 of MODEL_ANALYSIS.md: two-stage mixture.

Trains two binary LGBM classifiers over the v21 feature stack:
  1. `is_fb`  = |y - sd| < 1 for rows with sd > 1800  (BLOCK==SCHED fallback)
  2. `is_24h` = y > 80000                             (24h-offset rows)

Combines with v21 regressor at hold-out time to get one predicted taxi time:

    p_final = P_fb  * sd
            + P_24h * sd
            + (1 - P_fb - P_24h) * taxi_hat_v21

Both fallback and 24h cases have `y = sd` exactly (see analysis 3.1). Uses OOF
probabilities on the training set for downstream tuning; the hold-out eval uses
classifiers trained on non-holdout only.
"""
import glob, os, pickle, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS
from features_advanced import add_advanced, ADV_NUM_COLS
from features_osm_path import add_osm_path, OSM_PATH_NUM_COLS
from features_opdi import add_opdi, OPDI_NUM_COLS
from features_opdi_live import add_opdi_live, OPDI_LIVE_NUM_COLS
from train_lgbm_v21 import add_obt_features, CAT_COLS, NUM_COLS_V21

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}

CLF_PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 40,
    "feature_fraction": 0.7, "bagging_fraction": 0.9, "bagging_freq": 5,
    "verbosity": -1, "num_threads": -1,
}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def build_stack():
    t0 = time.time()
    print("Loading...")
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
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep = add_obt_features(dep)
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]

    m_ctx = m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("Features...")
    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m_ctx, daily_ec=ec_daily)
    del m_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  Done in {time.time()-t0:.1f}s")
    return dep, encoders


def main():
    t_all = time.time()
    dep, encoders = build_stack()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train = dep.loc[~hold].copy()
    test = dep.loc[hold].copy()
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V21 + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS + OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS
    print(f"Feature count: {len(feat)}")

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    sd_tr = train["sched_delay"].values.astype(float)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)

    # Two target vectors on training
    is_fb_tr  = ((np.abs(y_tr - sd_tr) < 1) & (sd_tr > 1800)).astype(np.int8)
    is_24h_tr = (y_tr > 80000).astype(np.int8)
    is_fb_te  = ((np.abs(y_te - sd_te) < 1) & (sd_te > 1800)).astype(np.int8)
    is_24h_te = (y_te > 80000).astype(np.int8)
    print(f"is_fb   train pos {is_fb_tr.sum():>6}  ({is_fb_tr.mean()*100:.4f}%)   test pos {is_fb_te.sum():>4}")
    print(f"is_24h  train pos {is_24h_tr.sum():>6}  ({is_24h_tr.mean()*100:.4f}%)   test pos {is_24h_te.sum():>4}")

    # Regressor: load v21 booster and predict on test
    print("\nLoading v21 regressor...")
    reg = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        reg_feat = f.read().splitlines()
    taxi_hat_te = reg.predict(test[reg_feat])

    # Two classifiers on full train, evaluate on hold-out
    print("\nTraining is_fb classifier...")
    d_tr = lgb.Dataset(train[feat], label=is_fb_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    d_va = lgb.Dataset(test[feat],  label=is_fb_te, categorical_feature=CAT_COLS,
                       reference=d_tr, free_raw_data=True)
    bfb = lgb.train(CLF_PARAMS, d_tr, num_boost_round=1500,
                    valid_sets=[d_va], valid_names=["valid"],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    p_fb_te = bfb.predict(test[feat], num_iteration=bfb.best_iteration)

    print("\nTraining is_24h classifier...")
    d_tr2 = lgb.Dataset(train[feat], label=is_24h_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    d_va2 = lgb.Dataset(test[feat],  label=is_24h_te, categorical_feature=CAT_COLS,
                        reference=d_tr2, free_raw_data=True)
    b24 = lgb.train(CLF_PARAMS, d_tr2, num_boost_round=1500,
                    valid_sets=[d_va2], valid_names=["valid"],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    p_24_te = b24.predict(test[feat], num_iteration=b24.best_iteration)

    # Combine (both classes predict y = sd)
    p_any = np.clip(p_fb_te + p_24_te, 0, 1)
    p_mix = p_any * sd_te + (1 - p_any) * taxi_hat_te
    # Safe fallback: if sd is NaN, revert to taxi_hat
    p_mix = np.where(np.isnan(p_mix), taxi_hat_te, p_mix)
    p_mix = np.clip(p_mix, 0, None)

    # Baselines
    p_v21 = np.clip(taxi_hat_te, 0, None)

    print("\n=== Hold-out RMSE ===")
    print(f"v21 alone         : {rmse(y_te, p_v21):.2f} s")
    print(f"v22 mixture       : {rmse(y_te, p_mix):.2f} s")

    clean = (y_te >= 30) & (y_te <= 7200)
    print(f"\nCLEAN (30<=y<=7200):")
    print(f"  v21             : {rmse(y_te[clean], p_v21[clean]):.2f}")
    print(f"  v22 mixture     : {rmse(y_te[clean], p_mix[clean]):.2f}")
    tail = y_te > 7200
    print(f"\nTAIL  (y > 7200)   n={tail.sum()}:")
    print(f"  v21             : {rmse(y_te[tail], p_v21[tail]):.2f}   mean_p {p_v21[tail].mean():.0f}   mean_y {y_te[tail].mean():.0f}")
    print(f"  v22 mixture     : {rmse(y_te[tail], p_mix[tail]):.2f}   mean_p {p_mix[tail].mean():.0f}")

    print(f"\np_fb  on hold-out: max {p_fb_te.max():.3f}   mean {p_fb_te.mean():.4f}   >0.5 {(p_fb_te>0.5).sum()}")
    print(f"p_24h on hold-out: max {p_24_te.max():.3f}   mean {p_24_te.mean():.4f}   >0.5 {(p_24_te>0.5).sum()}")

    # Save classifiers
    bfb.save_model(os.path.join(MODELS, "lgbm_v22_isfb.txt"))
    b24.save_model(os.path.join(MODELS, "lgbm_v22_is24h.txt"))
    with open(os.path.join(MODELS, "lgbm_v22.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(MODELS, "lgbm_v22.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {MODELS}/lgbm_v22_*   total wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
