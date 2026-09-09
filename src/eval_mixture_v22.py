"""Tune v22 mixture: threshold on p_fb / p_24h and sd-gate.
Loads v21 regressor + v22 classifiers, rebuilds features, sweeps thresholds.
"""
import glob, os, pickle, time, gc
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
from features_opdi_live import add_opdi_live
from train_lgbm_v21 import add_obt_features, CAT_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")
HOLDOUT_MONTHS = {1, 7}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def build_test():
    t0 = time.time()
    print("Loading + features...")
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

    m_ctx = m[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    with open(os.path.join(MODELS, "lgbm_v21.encoders.pkl"), "rb") as f:
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

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train_cats = {c: dep.loc[~hold, c].astype("category").cat.categories for c in CAT_COLS}
    test = dep.loc[hold].copy()
    for c in CAT_COLS:
        test[c] = pd.Categorical(test[c], categories=train_cats[c])
    print(f"  Done in {time.time()-t0:.1f}s   test n={len(test):,}")
    return test


def main():
    test = build_test()
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)

    reg = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        reg_feat = f.read().splitlines()
    taxi_hat = reg.predict(test[reg_feat])

    bfb = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v22_isfb.txt"))
    b24 = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v22_is24h.txt"))
    with open(os.path.join(MODELS, "lgbm_v22.features.txt")) as f:
        clf_feat = f.read().splitlines()
    p_fb = bfb.predict(test[clf_feat])
    p_24 = b24.predict(test[clf_feat])

    p_v21 = np.clip(taxi_hat, 0, None)
    print(f"\nv21 baseline full RMSE: {rmse(y_te, p_v21):.2f}")
    clean = (y_te >= 30) & (y_te <= 7200)
    tail = y_te > 7200
    print(f"  clean {rmse(y_te[clean], p_v21[clean]):.2f}   tail {rmse(y_te[tail], p_v21[tail]):.2f}")

    print("\nSoft mixture (uncond):")
    p_any = np.clip(p_fb + p_24, 0, 1)
    p_soft = np.where(np.isnan(sd_te), taxi_hat, p_any * sd_te + (1 - p_any) * taxi_hat)
    p_soft = np.clip(p_soft, 0, None)
    print(f"  full {rmse(y_te, p_soft):.2f}   clean {rmse(y_te[clean], p_soft[clean]):.2f}   tail {rmse(y_te[tail], p_soft[tail]):.2f}")

    print("\nHard-threshold + sd-gate variants:")
    print(f"{'thr':>5} {'sd_gate':>8} {'full':>8} {'clean':>8} {'tail':>10} {'n_swap':>8}")
    for thr in [0.5, 0.7, 0.9, 0.95, 0.99]:
        for sd_gate in [0, 1800, 3600, 7200]:
            swap = ((p_fb > thr) | (p_24 > thr)) & (sd_te > sd_gate) & ~np.isnan(sd_te)
            p_hard = np.where(swap, sd_te, taxi_hat)
            p_hard = np.clip(p_hard, 0, None)
            print(f"{thr:>5.2f} {sd_gate:>8} {rmse(y_te, p_hard):>8.2f} {rmse(y_te[clean], p_hard[clean]):>8.2f} {rmse(y_te[tail], p_hard[tail]):>10.2f} {swap.sum():>8}")

    print("\nOracle (perfect classifier):")
    is_fb = (np.abs(y_te - sd_te) < 1) & (sd_te > 1800)
    is_24h = y_te > 80000
    swap = is_fb | is_24h
    p_or = np.where(swap, sd_te, taxi_hat)
    p_or = np.clip(p_or, 0, None)
    print(f"  full {rmse(y_te, p_or):.2f}   clean {rmse(y_te[clean], p_or[clean]):.2f}   tail {rmse(y_te[tail], p_or[tail]):.2f}   n_swap {swap.sum()}")


if __name__ == "__main__":
    main()
