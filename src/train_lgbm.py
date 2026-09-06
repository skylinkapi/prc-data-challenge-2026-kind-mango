"""LightGBM on the full safe-feature set (categoricals + weather + congestion).

LightGBM handles high-cardinality categoricals natively (no top-N cap),
supports L1/L2, early stopping on RMSE, and typically beats sklearn's HistGBR
by 5-15 s on this dataset size.

Hold-out: Jan + Jul 2025. Reports overall + per-airport RMSE and feature gain.
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


def load_movements() -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    return m


def prep_dep(m: pd.DataFrame) -> pd.DataFrame:
    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    return dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)]


def main():
    t0 = time.time()
    print("Loading movements...")
    m = load_movements()
    print(f"  {len(m):,} rows in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Building DEP + weather + congestion...")
    dep = prep_dep(m)
    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    print(f"  {len(dep):,} DEP rows in {time.time()-t0:.1f}s")

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train, test = dep.loc[~hold].copy(), dep.loc[hold].copy()
    print(f"Train {len(train):,}   Hold-out {len(test):,}")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    feat = CAT_COLS + NUM_COLS
    X_tr, X_te = train[feat], test[feat]
    y_tr, y_te = train["TAXITIME_SEC_mvt"].values, test["TAXITIME_SEC_mvt"].values

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS,
                         free_raw_data=False)
    dvalid = lgb.Dataset(X_te, label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": -1,
    }

    print("\nTraining LightGBM...")
    t0 = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    print(f"  best iter: {booster.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = booster.predict(X_te, num_iteration=booster.best_iteration)
    print(f"\nLightGBM overall RMSE: {rmse(y_te, p_te):.1f}s")
    print(f"Reference: HistGBR 313.8s, baseline (4) 365.1s")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].values, "y": y_te, "p": p_te}) \
        .groupby("apt", observed=True).apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(1)
    print("\nPer-airport RMSE:")
    print(tab.sort_values("rmse").to_string())

    print("\nTop-20 features by gain:")
    imp = pd.Series(booster.feature_importance(importance_type="gain"),
                    index=feat).sort_values(ascending=False)
    print(imp.head(20).round(0).to_string())

    # save model + feature list for reuse
    out_dir = os.path.join(ROOT, "models")
    os.makedirs(out_dir, exist_ok=True)
    booster.save_model(os.path.join(out_dir, "lgbm_v1.txt"))
    with open(os.path.join(out_dir, "lgbm_v1.features.txt"), "w") as f:
        f.write("\n".join(feat))
    print(f"\nSaved model -> {out_dir}/lgbm_v1.txt")


if __name__ == "__main__":
    main()
