"""Step 1 of MODEL_ANALYSIS.md: evaluate saved v15 booster on the FULL hold-out
(no `between(30, 7200)` filter, no `_qual_bad` filter). Prints clean vs full RMSE.
"""
import os, glob, gc, pickle, time
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol
from features_operator import apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path
from features_opdi import add_opdi
from features_opdi_live import add_opdi_live

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}

CAT_COLS = ["ADEP_mvt","ADES_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt",
            "WK_TBL_CAT_flt","MARKET_SEGMENT_flt","AIRCRAFT_OPERATOR_flt",
            "FLIGHT_RULE_mvt","FLIGHT_TYPE_flt"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    t0 = time.time()
    print("Loading training corpus (unfiltered labels)...")
    files = sorted(glob.glob(os.path.join(ROOT, "training", "*.parquet")))
    m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)

    # NO label filter here (that is the point). Only drop y <= 0.
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]
    print(f"DEP rows kept: {len(dep):,}   with y > 7200: {(dep['TAXITIME_SEC_mvt']>7200).sum():,}")

    m_ctx = m[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    with open(os.path.join(MODELS, "lgbm_v15.encoders.pkl"), "rb") as f:
        enc = pickle.load(f)
    dep = apply_encoders(dep, enc)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m_ctx, daily_ec=ec_daily)
    del m_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    gc.collect()

    # Categorical align via v15 train categories: use its features file only
    # (LGBM stores its own category maps in the model).
    b = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v15.txt"))
    with open(os.path.join(MODELS, "lgbm_v15.features.txt")) as f:
        feat = f.read().splitlines()
    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    test = dep.loc[hold]
    y_full = test["TAXITIME_SEC_mvt"].values.astype(float)
    print(f"\nHold-out rows: {len(test):,}   y>7200: {(y_full>7200).sum():,}   y>20000: {(y_full>20000).sum():,}   y>80000: {(y_full>80000).sum():,}")

    p = b.predict(test[feat])
    # Two prediction variants
    p_clipped = np.clip(p, 60, 7200)
    p_raw = np.clip(p, 0, None)

    # Full RMSE
    print(f"\nFULL hold-out RMSE (clip 60,7200):   {rmse(y_full, p_clipped):.2f} s")
    print(f"FULL hold-out RMSE (no upper clip):  {rmse(y_full, p_raw):.2f} s")

    # Clean subset for reference
    clean = (y_full >= 30) & (y_full <= 7200)
    print(f"CLEAN hold-out RMSE:                  {rmse(y_full[clean], p_clipped[clean]):.2f} s  (n={clean.sum():,})")

    # By label band
    print("\nBy label band on full hold-out (predicted vs true):")
    bands = [(0, 30), (30, 7200), (7200, 20000), (20000, 80000), (80000, 1e9)]
    for lo, hi in bands:
        m2 = (y_full > lo) & (y_full <= hi)
        n = int(m2.sum())
        if n == 0: continue
        contrib = float(np.sum((y_full[m2] - p_clipped[m2]) ** 2)) / len(y_full)
        print(f"  ({lo:>6}, {hi:>7}]:  n={n:>6,}   mean_y={y_full[m2].mean():>8.0f}   MSE contrib {contrib:>10.1f}")

    print(f"\nWall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
