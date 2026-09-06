"""Floor baselines for PRC DC 2026 taxi-out prediction.

Reports RMSE on a hold-out that mirrors the ranking window: Jan and Jul 2025
DEP rows. Train on the remaining 10 months of 2025.
"""
import glob
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
TARGET_ICAOS = {"EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
                "LFPG", "LIRF", "LTAI", "LTFM", "LSZH"}
HOLDOUT_MONTHS = {1, 7}


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def load_dep() -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    df["sched_ts"] = pd.to_datetime(df["SCHED_TIME_UTC_mvt"], errors="coerce")
    df["hour"] = df["mvt_ts"].dt.hour
    df["month"] = df["mvt_ts"].dt.month
    df["sched_delay"] = (df["mvt_ts"] - df["sched_ts"]).dt.total_seconds()
    df = df[df["TAXITIME_SEC_mvt"].between(30, 7200)]
    return df


def split(df: pd.DataFrame):
    hold = df["month"].isin(HOLDOUT_MONTHS)
    return df.loc[~hold].copy(), df.loc[hold].copy()


def fit_group_median(train: pd.DataFrame, keys: list[str], fallback: float) -> callable:
    stats = train.groupby(keys)["TAXITIME_SEC_mvt"].median()
    def predict(df):
        out = df.set_index(keys).index.map(stats)
        return pd.Series(out, index=df.index).astype(float).fillna(fallback).values
    return predict


def report(name: str, y: np.ndarray, yhat: np.ndarray, apt: pd.Series) -> None:
    overall = rmse(y, yhat)
    per_apt = pd.DataFrame({"apt": apt.values, "y": y, "yhat": yhat}) \
        .groupby("apt").apply(lambda g: rmse(g["y"], g["yhat"])).round(1)
    print(f"\n[{name}]  overall RMSE = {overall:.1f}s")
    print(per_apt.sort_values().to_string())


def sched_delay_correction(train: pd.DataFrame, test: pd.DataFrame, base_pred: np.ndarray) -> np.ndarray:
    resid = train["TAXITIME_SEC_mvt"].values - train["_base"].values
    x = train["sched_delay"].clip(-1800, 3600).fillna(0).values
    slope = np.dot(x - x.mean(), resid - resid.mean()) / max(np.dot(x - x.mean(), x - x.mean()), 1e-9)
    intercept = resid.mean() - slope * x.mean()
    xt = test["sched_delay"].clip(-1800, 3600).fillna(0).values
    return base_pred + intercept + slope * xt


def main():
    print("Loading training corpus...")
    df = load_dep()
    train, test = split(df)
    print(f"Train rows: {len(train):,}   Hold-out rows: {len(test):,}")

    y_test = test["TAXITIME_SEC_mvt"].values
    apt_test = test["ADEP_mvt"]

    # 1. Global median
    g = train["TAXITIME_SEC_mvt"].median()
    report("1. global median", y_test, np.full(len(test), g), apt_test)

    # 2. Per-airport median
    pred2 = fit_group_median(train, ["ADEP_mvt"], g)
    report("2. per-airport median", y_test, pred2(test), apt_test)

    # 3. Per (airport, runway, hour, wake, segment) median with fallback chain
    keys_full = ["ADEP_mvt", "RUNWAY_mvt", "hour", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt"]
    keys_mid = ["ADEP_mvt", "RUNWAY_mvt", "hour"]
    p_full = fit_group_median(train, keys_full, g)
    p_mid = fit_group_median(train, keys_mid, g)
    p_apt = fit_group_median(train, ["ADEP_mvt"], g)
    pf = p_full(test); pm = p_mid(test); pa = p_apt(test)
    # fallback where deeper key was missing (stats returned nan → filled with g)
    pred3 = np.where(pf != g, pf, np.where(pm != g, pm, pa))
    report("3. per (ADEP,RWY,hour,wake,segment) median", y_test, pred3, apt_test)

    # 4. Add sched_delay linear correction on top of (3)
    train["_base"] = np.where(
        fit_group_median(train, keys_full, g)(train) != g,
        fit_group_median(train, keys_full, g)(train),
        np.where(
            fit_group_median(train, keys_mid, g)(train) != g,
            fit_group_median(train, keys_mid, g)(train),
            fit_group_median(train, ["ADEP_mvt"], g)(train),
        ),
    )
    pred4 = sched_delay_correction(train, test, pred3)
    report("4. (3) + sched_delay slope", y_test, pred4, apt_test)


if __name__ == "__main__":
    main()
