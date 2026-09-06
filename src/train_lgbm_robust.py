"""Compare three tail-robust variants of the LightGBM model.

A. Log-target: train on log1p(y), predict, expm1 back
B. Huber loss: objective=huber, alpha=0.9
C. Tight cap: same MSE objective, but training target clipped at 3600s

All three keep the exact same features. The comparison isolates the loss-shape
effect on hold-out RMSE, especially at LIRF where the tail dominates.
"""
import glob
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]
NUM_COLS = ["hour", "dow", "month", "sched_delay",
            "sknt", "gust_kt", "wind_cross_kt", "wind_head_kt",
            "vis_km", "low_vis", "very_low_vis",
            "ceiling_ft", "low_ceiling",
            "wx_precip", "wx_snow", "wx_thunder", "wx_freezing",
            "dep_load_prev_15m", "dep_load_prev_30m", "dep_load_prev_60m",
            "arr_load_prev_15m", "arr_load_prev_30m", "arr_load_prev_60m",
            "dep_same_rwy_prev_15m", "dep_same_rwy_prev_30m",
            "dep_same_rwy_prev_60m", "dep_queue_next_10m"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load_and_prep():
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep = dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)]
    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    return dep


def split_encode(dep):
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train, test = dep.loc[~hold].copy(), dep.loc[hold].copy()
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    return train, test


BASE_PARAMS = dict(
    learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    lambda_l2=1.0, verbosity=-1, num_threads=-1, metric="rmse",
)


def train_variant(name, params, y_transform, y_inverse, train, test, feat, y_train_override=None):
    y_tr = y_train_override if y_train_override is not None else train["TAXITIME_SEC_mvt"].values
    y_te = test["TAXITIME_SEC_mvt"].values
    y_tr_t = y_transform(y_tr)
    dtrain = lgb.Dataset(train[feat], label=y_tr_t, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_transform(y_te),
                         categorical_feature=CAT_COLS, reference=dtrain, free_raw_data=False)
    t0 = time.time()
    booster = lgb.train(
        params, dtrain, num_boost_round=3000,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    p_te = y_inverse(booster.predict(test[feat], num_iteration=booster.best_iteration))
    dt = time.time() - t0
    return booster, p_te, dt


def per_airport_rmse(test, p):
    y = test["TAXITIME_SEC_mvt"].values
    df = pd.DataFrame({"apt": test["ADEP_mvt"].values, "y": y, "p": p})
    return df.groupby("apt", observed=True).apply(
        lambda g: rmse(g["y"], g["p"]), include_groups=False).round(1)


def main():
    print("Loading + features...")
    dep = load_and_prep()
    train, test = split_encode(dep)
    print(f"Train {len(train):,}   Hold-out {len(test):,}")
    feat = CAT_COLS + NUM_COLS

    results = {}

    # Baseline reference (same MSE, no cap) from lgbm_v1
    print("\n--- A. Log-target (MSE on log1p) ---")
    _, p_A, dtA = train_variant("log", {**BASE_PARAMS, "objective": "regression"},
                                np.log1p, np.expm1, train, test, feat)
    r_A = rmse(test["TAXITIME_SEC_mvt"], p_A)
    print(f"Overall RMSE: {r_A:.1f}s   ({dtA:.1f}s)")

    print("\n--- B. Huber loss (alpha=0.9) ---")
    _, p_B, dtB = train_variant("huber", {**BASE_PARAMS, "objective": "huber", "alpha": 0.9},
                                lambda x: x, lambda x: x, train, test, feat)
    r_B = rmse(test["TAXITIME_SEC_mvt"], p_B)
    print(f"Overall RMSE: {r_B:.1f}s   ({dtB:.1f}s)")

    print("\n--- C. MSE with training target clipped at 3600s ---")
    y_train_clip = np.minimum(train["TAXITIME_SEC_mvt"].values, 3600)
    _, p_C, dtC = train_variant("clip", {**BASE_PARAMS, "objective": "regression"},
                                lambda x: x, lambda x: x, train, test, feat,
                                y_train_override=y_train_clip)
    r_C = rmse(test["TAXITIME_SEC_mvt"], p_C)
    print(f"Overall RMSE: {r_C:.1f}s   ({dtC:.1f}s)")

    print("\n=== SUMMARY (vs lgbm_v1 = 302.2s) ===")
    print(f"  A. log-target      : {r_A:.1f}s")
    print(f"  B. Huber alpha=0.9 : {r_B:.1f}s")
    print(f"  C. MSE + cap 3600s : {r_C:.1f}s")

    print("\nPer-airport RMSE:")
    tab = pd.DataFrame({
        "log":   per_airport_rmse(test, p_A),
        "huber": per_airport_rmse(test, p_B),
        "clip":  per_airport_rmse(test, p_C),
    })
    print(tab.sort_index().to_string())


if __name__ == "__main__":
    main()
