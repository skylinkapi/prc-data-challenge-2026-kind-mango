"""Test tail-classifier mixture with base regressor.

Base regressor: pick the best current model (v11).
Tail classifier: from train_tail_classifier.py (P_tail per row).
Tail median: per-airport median of training tail rows.

Mixture variants tested:
  1. linear:      pred = (1 - P) * base + P * tail_median
  2. squared:     pred = (1 - P^2) * base + P^2 * tail_median   (dampens low-P noise)
  3. cubed:       pred = (1 - P^3) * base + P^3 * tail_median   (only high-P shifts)
  4. threshold:   pred = base if P < t else tail_median         (hard switch)
  5. blend@thr:   pred = base if P < t else 0.5*(base + tail_median)  (soft high-P)
"""
import json
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol
from features_operator import fit_encoders, apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path
from features_opdi import add_opdi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def build_hold():
    import glob, time
    t0 = time.time()
    print("Loading + features...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ROOT, "training", "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep = dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)]

    m_ctx = m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m_ctx, daily_ec=ec_daily)
    del m_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train = dep.loc[~hold]
    test = dep.loc[hold].copy()

    for c in CAT_COLS:
        cats = train[c].astype("category").cat.categories
        test[c] = pd.Categorical(test[c], categories=cats)
    del train, dep; gc.collect()
    print(f"  Hold-out ready in {time.time()-t0:.1f}s")
    return test


def load_pred(tag, X):
    b = lgb.Booster(model_file=os.path.join(MODELS_DIR, f"lgbm_{tag}.txt"))
    with open(os.path.join(MODELS_DIR, f"lgbm_{tag}.features.txt")) as f:
        feat = f.read().splitlines()
    return b.predict(X[feat])


def main():
    test = build_hold()
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    apt = test["ADEP_mvt"].astype(str).values

    print("Predicting v11 (base) + tail classifier...")
    base = load_pred("v11", test)

    tc = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lgbm_tail_classifier.txt"))
    with open(os.path.join(MODELS_DIR, "lgbm_tail_classifier.features.txt")) as f:
        tc_feat = f.read().splitlines()
    P = tc.predict(test[tc_feat])

    with open(os.path.join(MODELS_DIR, "tail_medians.json")) as f:
        tail_medians = json.load(f)
    # per-row tail target
    tail_target = np.array([tail_medians.get(a, 4000.0) for a in apt])

    print(f"\nBase v11 RMSE: {rmse(y, base):.3f}s")
    print(f"P_tail stats: mean {P.mean():.4f}   p95 {np.percentile(P, 95):.4f}   p99 {np.percentile(P, 99):.4f}   max {P.max():.4f}")
    print(f"Rows with P_tail > 0.5: {(P > 0.5).sum():,}   > 0.8: {(P > 0.8).sum():,}   > 0.9: {(P > 0.9).sum():,}")
    print(f"True tail rate: {(y > 3600).mean()*100:.3f}%")

    variants = {}
    variants["1. linear P"]           = (1 - P) * base + P * tail_target
    variants["2. squared P^2"]        = (1 - P**2) * base + P**2 * tail_target
    variants["3. cubed P^3"]          = (1 - P**3) * base + P**3 * tail_target
    variants["4. P^0.5 (aggressive)"] = (1 - np.sqrt(P)) * base + np.sqrt(P) * tail_target

    for name, pred in variants.items():
        r = rmse(y, pred)
        print(f"  {name:25s} RMSE {r:.3f}s   delta {r - rmse(y, base):+.3f}")

    print("\nHard-threshold variants:")
    for thr in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        pred_hard = np.where(P > thr, tail_target, base)
        pred_soft = np.where(P > thr, 0.5 * (base + tail_target), base)
        r_h = rmse(y, pred_hard)
        r_s = rmse(y, pred_soft)
        n = (P > thr).sum()
        print(f"  thr={thr:.2f}  n_switch={n:5d}   hard RMSE {r_h:.3f} (delta{r_h-rmse(y,base):+.2f})   soft-blend RMSE {r_s:.3f} (delta{r_s-rmse(y,base):+.2f})")

    print("\nDelta by airport (best variant tbd):")
    # Try P^3 (moderate) and hard@0.9 (aggressive)
    p_cube = (1 - P**3) * base + P**3 * tail_target
    p_hard = np.where(P > 0.9, tail_target, base)
    tab = pd.DataFrame({"apt": apt, "y": y, "base": base, "cube": p_cube, "hard": p_hard}) \
        .groupby("apt").apply(
            lambda g: pd.Series({
                "n": len(g),
                "rmse_base": rmse(g["y"], g["base"]),
                "rmse_cube": rmse(g["y"], g["cube"]),
                "rmse_hard": rmse(g["y"], g["hard"]),
            }), include_groups=False).round(2)
    tab["dcube"] = (tab["rmse_cube"] - tab["rmse_base"]).round(2)
    tab["dhard"] = (tab["rmse_hard"] - tab["rmse_base"]).round(2)
    print(tab.sort_values("dhard").to_string())


if __name__ == "__main__":
    main()
