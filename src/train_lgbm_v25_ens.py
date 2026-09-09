"""Step 6: refit v24 recipe on 12 months + train 4-member seed/ff ensemble.

Each member: v24 stack, LGBM with different (seed, feature_fraction).
Two stages per member:
  1) fit on 10 months (Jan+Jul as validation, early stopping) to get best_iter
  2) refit on all 12 months with round_count = 1.2 * best_iter, no early stop

Then NNLS on hold-out predictions (from stage 1) for ensemble weights.
"""
import glob, os, pickle, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls

from features_weather import add_weather, TARGET_ICAOS
from features_congestion_v2 import add_congestion_v2, CONG_V2_NUM_COLS
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS
from features_advanced import add_advanced, ADV_NUM_COLS
from features_osm_path import add_osm_path, OSM_PATH_NUM_COLS
from features_opdi_v2 import add_opdi_v2, OPDI_V2_NUM_COLS
from features_opdi_live_v2 import add_opdi_live_v2, OPDI_LIVE_V2_NUM_COLS
from train_lgbm_v21 import add_obt_features, CAT_COLS, BEST_PARAMS
from train_lgbm_v23 import BASE_NUM_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}

VARIANTS = [
    ("v25a", {"seed": 42,  "feature_fraction": 0.6737}),
    ("v25b", {"seed": 43,  "feature_fraction": 0.6737}),
    ("v25c", {"seed": 42,  "feature_fraction": 0.55}),
    ("v25d", {"seed": 44,  "feature_fraction": 0.75}),
]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def build_features():
    t0 = time.time()
    print("Loading + features...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS) | m["ADES_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[(m["PHASE_mvt"] == "DEP") & m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    sd = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds()
    dep["sched_delay"] = sd
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep = add_obt_features(dep)
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]

    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","sched_ts",
             "RUNWAY_mvt","TAXITIME_SEC_mvt"]].copy()
    ctx_adv = m[m["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion_v2(dep, ctx)
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, ctx_adv, daily_ec=ec_daily)
    del ctx, ctx_adv, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi_v2(dep)
    dep = add_opdi_live_v2(dep)
    gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  Feature build done in {time.time()-t0:.1f}s")
    return dep, encoders


def main():
    t_all = time.time()
    dep, encoders = build_features()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train10 = dep.loc[~hold].copy()
    test = dep.loc[hold].copy()
    for c in CAT_COLS:
        train10[c] = train10[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train10[c].cat.categories)
        dep[c] = pd.Categorical(dep[c], categories=train10[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + BASE_NUM_COLS + CONG_V2_NUM_COLS + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS + OPDI_V2_NUM_COLS + OPDI_LIVE_V2_NUM_COLS
    print(f"Feature count: {len(feat)}")

    y_tr10 = train10["TAXITIME_SEC_mvt"].values.astype(float)
    y_te   = test["TAXITIME_SEC_mvt"].values.astype(float)
    y_all  = dep["TAXITIME_SEC_mvt"].values.astype(float)

    holdout_preds = {}
    best_iters = {}

    # Stage 1: fit on 10m, early stop on Jan+Jul, get hold-out preds
    for tag, over in VARIANTS:
        print(f"\n[Stage 1] {tag}  overrides={over}")
        params = {**BEST_PARAMS, **over, "objective": "regression", "metric": "rmse",
                  "bagging_freq": 5, "verbosity": -1, "num_threads": -1,
                  "bagging_seed": over.get("seed", 42)}
        dt = lgb.Dataset(train10[feat], label=y_tr10, categorical_feature=CAT_COLS, free_raw_data=True)
        dv = lgb.Dataset(test[feat], label=y_te, categorical_feature=CAT_COLS,
                         reference=dt, free_raw_data=True)
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=3000,
                      valid_sets=[dv], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(400)])
        p = b.predict(test[feat], num_iteration=b.best_iteration)
        holdout_preds[tag] = p
        best_iters[tag] = b.best_iteration
        print(f"  {tag}: best iter {b.best_iteration}   hold RMSE {rmse(y_te, np.clip(p,0,None)):.2f}   ({time.time()-t0:.1f}s)")
        del b, dt, dv; gc.collect()

    # NNLS on hold-out
    stack = np.column_stack([holdout_preds[t] for t, _ in VARIANTS])
    w, _ = nnls(stack, y_te)
    ens_hold = stack @ w
    print(f"\nNNLS hold-out RMSE: {rmse(y_te, ens_hold):.2f}   weights sum {w.sum():.3f}")
    for (tag, _), wi in zip(VARIANTS, w):
        print(f"  {tag}: {wi:.4f}")
    w_norm = w / w.sum()
    print(f"Normalised weights: " + "  ".join(f"{t}={wi:.3f}" for (t,_), wi in zip(VARIANTS, w_norm)))
    with open(os.path.join(ROOT, "models", "v25_ensemble_weights.json"), "w") as fp:
        import json
        json.dump({tag: float(wi) for (tag, _), wi in zip(VARIANTS, w_norm)}, fp)

    # Stage 2: refit each on ALL 12 months with 1.2 * best_iter, no early stop
    print(f"\n[Stage 2] Refitting each on 12 months with 1.2x best_iter")
    for tag, over in VARIANTS:
        params = {**BEST_PARAMS, **over, "objective": "regression", "metric": "rmse",
                  "bagging_freq": 5, "verbosity": -1, "num_threads": -1,
                  "bagging_seed": over.get("seed", 42)}
        n_round = int(1.2 * best_iters[tag])
        print(f"  {tag}: {n_round} rounds on {len(dep):,} rows")
        dt = lgb.Dataset(dep[feat], label=y_all, categorical_feature=CAT_COLS, free_raw_data=True)
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=n_round,
                      callbacks=[lgb.log_evaluation(400)])
        b.save_model(os.path.join(ROOT, "models", f"lgbm_{tag}.txt"))
        with open(os.path.join(ROOT, "models", f"lgbm_{tag}.features.txt"), "w") as fp:
            fp.write("\n".join(feat))
        print(f"    saved  ({time.time()-t0:.1f}s)")
        del b, dt; gc.collect()

    with open(os.path.join(ROOT, "models", f"lgbm_v25.encoders.pkl"), "wb") as fp:
        pickle.dump(encoders, fp)
    print(f"\nDone. total wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
