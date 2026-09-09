"""Eval v21 alone vs v21 + Step A (gate 14400) vs v21 + Step A v2 (gate 3600)."""
import glob, os, pickle, json, gc, time
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
from train_lgbm_v21 import add_obt_features, CAT_COLS
from eval_v21_stepA import apply_rule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
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
    del dep; gc.collect()

    b = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        feat = f.read().splitlines()
    taxi_hat = np.clip(b.predict(test[feat]), 0, None)
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)
    adep_te = test["ADEP_mvt"].astype(str).values
    fltid_te = test["FLIGHT_ID_mvt"].values

    with open(os.path.join(MODELS, "lirf_band_table.json")) as f:
        t1 = json.load(f)
    with open(os.path.join(MODELS, "lirf_band_table_v2.json")) as f:
        t2 = json.load(f)

    p_v1, c1 = apply_rule(taxi_hat, sd_te, adep_te, fltid_te, t1)
    p_v2, c2 = apply_rule(taxi_hat, sd_te, adep_te, fltid_te, t2)

    print(f"\n=== Hold-out RMSE comparison ===")
    print(f"                          v21 alone    +ruleV1(gate 14400)    +ruleV2(gate 3600)")
    print(f"CELL rows affected                              {c1.sum():>4}                   {c2.sum():>4}")
    clean = (y >= 30) & (y <= 7200)
    print(f"CLEAN         {rmse(y[clean], taxi_hat[clean]):>8.2f}      {rmse(y[clean], p_v1[clean]):>8.2f}              {rmse(y[clean], p_v2[clean]):>8.2f}")
    print(f"FULL          {rmse(y, taxi_hat):>8.2f}      {rmse(y, p_v1):>8.2f}              {rmse(y, p_v2):>8.2f}")

    # Cell-only RMSE (v2 cell)
    if c2.sum() > 0:
        print(f"\nCell (gate 3600) rows only, n={c2.sum()}:")
        print(f"  v21 alone   RMSE {rmse(y[c2], taxi_hat[c2]):.2f}")
        print(f"  ruleV2      RMSE {rmse(y[c2], p_v2[c2]):.2f}")
        # Isolate the newly-added rows (in v2 but not v1)
        new_rows = c2 & ~c1
        if new_rows.sum() > 0:
            print(f"\nNEW rows (v2 - v1), n={new_rows.sum()}:")
            print(f"  v21 alone   RMSE {rmse(y[new_rows], taxi_hat[new_rows]):.2f}")
            print(f"  ruleV2      RMSE {rmse(y[new_rows], p_v2[new_rows]):.2f}")
    print(f"\nWall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
