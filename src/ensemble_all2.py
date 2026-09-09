"""Ensemble NNLS over all 11 trained models: v5-v11, v15-v19."""
import glob
import os
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls, minimize

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol
from features_operator import fit_encoders, apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path
from features_opdi import add_opdi
from features_opdi_live import add_opdi_live
from features_opdi_extended import add_opdi_ext
from features_ssl import add_ssl
from features_openap import add_openap
from features_opdi_climate import add_opdi_climate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}
TAGS = ["v5", "v6", "v7", "v9", "v10", "v11", "v15", "v16", "v17", "v18", "v19"]

CAT_COLS = ["ADEP_mvt","ADES_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt",
            "WK_TBL_CAT_flt","MARKET_SEGMENT_flt","AIRCRAFT_OPERATOR_flt",
            "FLIGHT_RULE_mvt","FLIGHT_TYPE_flt"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def build_hold():
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

    training_ssl_ref = dep[["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "sched_delay"]].copy()
    m_ctx = m[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m_ctx, daily_ec=ec_daily)
    del m_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    dep = add_opdi_ext(dep)
    dep = add_ssl(dep, training_ssl_ref)
    del training_ssl_ref; gc.collect()
    dep = add_openap(dep)
    dep = add_opdi_climate(dep)
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


def predict(tag, X):
    b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_{tag}.txt"))
    with open(os.path.join(MODELS, f"lgbm_{tag}.features.txt")) as f:
        feat = f.read().splitlines()
    return b.predict(X[feat])


def main():
    print("Loading hold-out (v19 stack)...")
    test = build_hold()
    y = test["TAXITIME_SEC_mvt"].values.astype(float)

    P = {}
    for t in TAGS:
        p = predict(t, test)
        P[t] = p
        print(f"  {t:5s} RMSE {rmse(y, p):.3f}s")

    stack = np.column_stack([P[t] for t in TAGS])

    print("\nNNLS (no sum constraint)...")
    w_nnls, _ = nnls(stack, y)
    p_nnls = stack @ w_nnls
    print(f"  RMSE {rmse(y, p_nnls):.3f}s   weights sum {w_nnls.sum():.3f}")
    for t, w in zip(TAGS, w_nnls):
        if w > 1e-6: print(f"    {t}: {w:.4f}")

    # Renormalize for stability
    w_norm = w_nnls / w_nnls.sum()
    p_norm = stack @ w_norm
    print(f"\nNormalised weights RMSE: {rmse(y, p_norm):.3f}s")

    print(f"\nBest previous ensemble (v10 sub, 5-model): RMSE 292.601s")

    with open(os.path.join(MODELS, "ensemble_weights_all2.txt"), "w") as f:
        f.write(f"# nnls_norm  hold_rmse={rmse(y, p_norm):.4f}\n")
        for t, w in zip(TAGS, w_norm):
            f.write(f"{t}: {w}\n")
    print(f"Saved -> {MODELS}/ensemble_weights_all2.txt")


if __name__ == "__main__":
    main()
