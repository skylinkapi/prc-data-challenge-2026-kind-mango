"""Grid-search blend weights over v5+v6+v7+v9+v10 on hold-out.

Different models have different feature sets:
  v5, v7: 97 features (base)
  v6    : 98 features (+ flt_null)
  v9    : 103 features (+ 5 advanced physical)
  v10   : 106 features (+ 3 OSM path)

Rebuild the full feature set from training parquets to score every model.
"""
import glob
import os
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import product

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol
from features_operator import fit_encoders, apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load_full_holdout():
    """Recompute all features on training corpus and return the hold-out subset."""
    t0 = time.time()
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

    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]],
                       daily_ec=ec_daily)
    dep = add_osm_path(dep)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train = dep.loc[~hold]
    test = dep.loc[hold].copy()

    for c in CAT_COLS:
        cats = train[c].astype("category").cat.categories
        test[c] = pd.Categorical(test[c], categories=cats)
    print(f"  Hold-out rebuild took {time.time()-t0:.1f}s")
    return test


def predict(tag, X):
    b = lgb.Booster(model_file=os.path.join(MODELS_DIR, f"lgbm_{tag}.txt"))
    with open(os.path.join(MODELS_DIR, f"lgbm_{tag}.features.txt")) as f:
        feat = f.read().splitlines()
    return b.predict(X[feat])


def grid_search_weights(y, preds, step=0.05):
    """Search all w on the 5-simplex with granularity step."""
    keys = list(preds.keys())
    K = len(keys)
    S = int(1 / step)
    best = (float("inf"), None)
    # generate integer partitions of S into K nonneg parts
    def _rec(i, rem, cur):
        nonlocal best
        if i == K - 1:
            cur[i] = rem
            w = np.array(cur, dtype=float) / S
            p = sum(w[k] * preds[keys[k]] for k in range(K))
            r = rmse(y, p)
            if r < best[0]:
                best = (r, tuple(w))
            return
        for v in range(rem + 1):
            cur[i] = v
            _rec(i+1, rem-v, cur)
    _rec(0, S, [0]*K)
    return best


def main():
    print("Loading hold-out with all features...")
    test = load_full_holdout()
    y = test["TAXITIME_SEC_mvt"].values.astype(float)

    print("Predicting each model...")
    tags = ["v5", "v6", "v7", "v9", "v10"]
    preds = {}
    for t in tags:
        preds[t] = predict(t, test)
        print(f"  {t}: RMSE {rmse(y, preds[t]):.3f}s")

    # simple average
    simple = sum(preds.values()) / len(preds)
    print(f"\nSimple average: {rmse(y, simple):.3f}s")

    # 5-simplex grid at 0.05 (choose(5+20-1,20)=10626 combos)
    print("Grid search 5-way weights (step 0.05, ~10.6k combos)...")
    t0 = time.time()
    r, w = grid_search_weights(y, preds, step=0.05)
    dt = time.time() - t0
    print(f"  best: {r:.3f}s   ({dt:.1f}s)")
    for t, wt in zip(tags, w):
        print(f"    {t}: {wt:.2f}")

    ens = sum(wt * preds[t] for t, wt in zip(tags, w))

    # Per-airport
    apt = test["ADEP_mvt"].astype(str).values
    tab = pd.DataFrame({"apt": apt, "y": y, "p": ens, "p_v10": preds["v10"]}) \
        .groupby("apt").apply(
            lambda g: pd.Series({"n": len(g),
                                 "rmse_ens": rmse(g["y"], g["p"]),
                                 "rmse_v10": rmse(g["y"], g["p_v10"])}),
            include_groups=False).round(2)
    tab["delta"] = (tab["rmse_ens"] - tab["rmse_v10"]).round(2)
    print("\nPer-airport (ensemble vs v10):")
    print(tab.sort_values("rmse_ens").to_string())

    with open(os.path.join(MODELS_DIR, "ensemble_weights_final.txt"), "w") as f:
        for t, wt in zip(tags, w):
            f.write(f"{t}: {wt}\n")
    print(f"\nSaved -> {MODELS_DIR}/ensemble_weights_final.txt")


if __name__ == "__main__":
    main()
