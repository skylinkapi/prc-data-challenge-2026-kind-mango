"""Baseline model 4 (per-group median + sched_delay slope) plus a
HistGradientBoostingRegressor on residuals using weather features.

Isolates the marginal RMSE gain from METAR + runway crosswind alone.
"""
import glob
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features_weather import add_weather, TARGET_ICAOS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load_dep() -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    df["sched_ts"] = pd.to_datetime(df["SCHED_TIME_UTC_mvt"], errors="coerce")
    df["hour"] = df["mvt_ts"].dt.hour
    df["month"] = df["mvt_ts"].dt.month
    df["dow"] = df["mvt_ts"].dt.dayofweek
    df["sched_delay"] = (df["mvt_ts"] - df["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    df = df[df["TAXITIME_SEC_mvt"].between(30, 7200)]
    return df


def group_median_pred(train, test, keys, fallback):
    stats = train.groupby(keys)["TAXITIME_SEC_mvt"].median()
    idx = test.set_index(keys).index.map(stats)
    return pd.Series(idx, index=test.index).astype(float).fillna(fallback).values


def baseline4(train, test):
    g = train["TAXITIME_SEC_mvt"].median()
    keys_full = ["ADEP_mvt", "RUNWAY_mvt", "hour", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt"]
    keys_mid = ["ADEP_mvt", "RUNWAY_mvt", "hour"]
    def stack(frame):
        pf = group_median_pred(train, frame, keys_full, g)
        pm = group_median_pred(train, frame, keys_mid, g)
        pa = group_median_pred(train, frame, ["ADEP_mvt"], g)
        return np.where(pf != g, pf, np.where(pm != g, pm, pa))
    pred_tr = stack(train); pred_te = stack(test)
    # sched_delay slope on training residual
    resid_tr = train["TAXITIME_SEC_mvt"].values - pred_tr
    x = train["sched_delay"].fillna(0).values
    slope = np.dot(x - x.mean(), resid_tr - resid_tr.mean()) / max(np.dot(x - x.mean(), x - x.mean()), 1e-9)
    intercept = resid_tr.mean() - slope * x.mean()
    pred_tr = pred_tr + intercept + slope * x
    xt = test["sched_delay"].fillna(0).values
    pred_te = pred_te + intercept + slope * xt
    return pred_tr, pred_te


WEATHER_COLS = ["sknt", "gust_kt", "wind_cross_kt", "wind_head_kt",
                "vis_km", "low_vis", "very_low_vis",
                "ceiling_ft", "low_ceiling",
                "wx_precip", "wx_snow", "wx_thunder", "wx_freezing"]


def main():
    print("Loading + weather join...")
    df = load_dep()
    df = add_weather(df)
    hold = df["month"].isin(HOLDOUT_MONTHS)
    train, test = df.loc[~hold].copy(), df.loc[hold].copy()
    print(f"Train {len(train):,}   Hold-out {len(test):,}")

    # Baseline (4)
    pred_tr_b, pred_te_b = baseline4(train, test)
    print(f"\nBaseline (4) overall RMSE: {rmse(test['TAXITIME_SEC_mvt'], pred_te_b):.1f}s")

    # Residuals for weather model
    resid_tr = train["TAXITIME_SEC_mvt"].values - pred_tr_b
    X_tr = train[WEATHER_COLS].fillna(0).values
    X_te = test[WEATHER_COLS].fillna(0).values

    print("\nFitting HistGBR on residuals (weather features only)...")
    gbr = HistGradientBoostingRegressor(
        max_iter=400, max_depth=6, learning_rate=0.05,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0,
    )
    gbr.fit(X_tr, resid_tr)
    resid_te_pred = gbr.predict(X_te)
    pred_te_wx = pred_te_b + resid_te_pred

    r_base = rmse(test["TAXITIME_SEC_mvt"], pred_te_b)
    r_wx = rmse(test["TAXITIME_SEC_mvt"], pred_te_wx)
    print(f"\nBaseline (4)          RMSE: {r_base:.1f}s")
    print(f"+ weather residual    RMSE: {r_wx:.1f}s   (delta = {r_base - r_wx:+.1f}s)")

    print("\nPer-airport RMSE (baseline vs +weather):")
    perf = pd.DataFrame({
        "apt": test["ADEP_mvt"].values,
        "y": test["TAXITIME_SEC_mvt"].values,
        "base": pred_te_b,
        "wx": pred_te_wx,
    })
    tab = perf.groupby("apt").apply(
        lambda g: pd.Series({
            "n": len(g),
            "base": rmse(g["y"], g["base"]),
            "wx":   rmse(g["y"], g["wx"]),
        }), include_groups=False
    ).round(1)
    tab["delta"] = (tab["base"] - tab["wx"]).round(1)
    print(tab.sort_values("delta", ascending=False).to_string())

    print("\nFeature importance (permutation-free proxy: split gains)")
    # HistGBR doesn't expose gain directly; use feature_importances_ from prediction contribution.
    # Use built-in feature importances if available.
    if hasattr(gbr, "feature_importances_"):
        imp = pd.Series(gbr.feature_importances_, index=WEATHER_COLS).sort_values(ascending=False)
        print(imp.round(4).to_string())


if __name__ == "__main__":
    main()
