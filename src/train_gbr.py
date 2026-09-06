"""Full model: baseline categoricals + sched_delay + weather + congestion,
trained as a single HistGradientBoostingRegressor.

Hold-out: Jan + Jul 2025 (mirrors the ranking window Jan + Jul 2026).
Reports overall and per-airport RMSE vs the group-median baseline (365 s).
"""
import glob
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]

NUM_COLS_TIME = ["hour", "dow", "month", "sched_delay"]

NUM_COLS_WX = ["sknt", "gust_kt", "wind_cross_kt", "wind_head_kt",
               "vis_km", "low_vis", "very_low_vis",
               "ceiling_ft", "low_ceiling",
               "wx_precip", "wx_snow", "wx_thunder", "wx_freezing"]

NUM_COLS_CONG = ["dep_load_prev_15m", "dep_load_prev_30m", "dep_load_prev_60m",
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


CAT_MAX = 200   # HistGBR limit is 255; keep headroom for "OTHER"


def encode_cats(train: pd.DataFrame, test: pd.DataFrame, cols: list) -> None:
    """Cap each categorical to top-CAT_MAX values by train frequency; rest -> 'OTHER'."""
    for c in cols:
        top = train[c].value_counts(dropna=True).head(CAT_MAX).index
        cats = pd.Index(list(top) + ["OTHER"])
        for df in (train, test):
            df[c] = df[c].where(df[c].isin(top), other="OTHER")
            df[c] = pd.Categorical(df[c], categories=cats)


def main():
    t0 = time.time()
    print("Loading movements...")
    m = load_movements()
    print(f"  {len(m):,} rows in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Building DEP subset + weather + congestion...")
    dep = prep_dep(m)
    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    print(f"  {len(dep):,} DEP rows in {time.time()-t0:.1f}s")

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train, test = dep.loc[~hold].copy(), dep.loc[hold].copy()
    print(f"Train {len(train):,}   Hold-out {len(test):,}")

    feat = CAT_COLS + NUM_COLS_TIME + NUM_COLS_WX + NUM_COLS_CONG
    encode_cats(train, test, CAT_COLS)

    X_tr = train[feat]
    X_te = test[feat]
    y_tr = train["TAXITIME_SEC_mvt"].values
    y_te = test["TAXITIME_SEC_mvt"].values

    cat_mask = [c in CAT_COLS for c in feat]

    print("\nFitting HistGBR...")
    t0 = time.time()
    gbr = HistGradientBoostingRegressor(
        max_iter=600, max_depth=8, learning_rate=0.05,
        min_samples_leaf=100, l2_regularization=1.0,
        categorical_features=cat_mask, random_state=0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
    )
    gbr.fit(X_tr, y_tr)
    print(f"  iters used: {gbr.n_iter_}   in {time.time()-t0:.1f}s")

    p_te = gbr.predict(X_te)
    print(f"\nHistGBR (all features)  RMSE: {rmse(y_te, p_te):.1f}s")
    print(f"Baseline (4) reference:       365.1s")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].values, "y": y_te, "p": p_te}) \
        .groupby("apt").apply(lambda g: pd.Series({
            "n": len(g), "rmse": rmse(g["y"], g["p"])}), include_groups=False).round(1)
    print("\nPer-airport RMSE:")
    print(tab.sort_values("rmse").to_string())


if __name__ == "__main__":
    main()
