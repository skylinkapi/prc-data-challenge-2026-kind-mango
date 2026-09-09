"""lgbm_v23 = v21 + Step 4 arrival-context fix.

Uses features_congestion_v2 which keys ARR rows on ADES_mvt and adds:
  arr_same_rwy_prev_15m, arr_same_rwy_next_10m, arr_taxi_in_mean_60m.

Passes the unfiltered movement corpus (ADEP or ADES in TARGET) so ARR rows
at target airports are visible.
"""
import glob, os, pickle, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion_v2 import add_congestion_v2, CONG_V2_NUM_COLS
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS
from features_advanced import add_advanced, ADV_NUM_COLS
from features_osm_path import add_osm_path, OSM_PATH_NUM_COLS
from features_opdi import add_opdi, OPDI_NUM_COLS
from features_opdi_live import add_opdi_live, OPDI_LIVE_NUM_COLS
from train_lgbm_v21 import add_obt_features, CAT_COLS, BEST_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}

# v21 NUM_COLS without the six old congestion cols (replaced by CONG_V2_NUM_COLS)
BASE_NUM_COLS = ["hour", "dow", "month", "sched_delay", "sd_mod_86400",
                 "mvt_eobt1", "mvt_iobt", "eobt1_sched", "eobt1_iobt",
                 "sknt", "gust_kt", "wind_cross_kt", "wind_head_kt",
                 "vis_km", "low_vis", "very_low_vis",
                 "ceiling_ft", "low_ceiling",
                 "wx_precip", "wx_snow", "wx_thunder", "wx_freezing",
                 "flt_null", "flt_id_null"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    t_all = time.time()
    print("Loading (both ADEP and ADES in target set)...")
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

    print(f"DEP rows: {len(dep):,}   Total context (DEP+ARR both endpoints): {len(m):,}")

    ctx = m[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","sched_ts",
             "RUNWAY_mvt","TAXITIME_SEC_mvt"]].copy()
    # advanced still needs original filter (only DEP at target)
    ctx_adv = m[m["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    print("Adding features...")
    t = time.time()
    dep = add_weather(dep)
    dep = add_congestion_v2(dep, ctx)
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, ctx_adv, daily_ec=ec_daily)
    del ctx, ctx_adv, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  Done in {time.time()-t:.1f}s")

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train = dep.loc[~hold].copy()
    test = dep.loc[hold].copy()
    print(f"Train {len(train):,}   Hold-out (full) {len(test):,}")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + BASE_NUM_COLS + CONG_V2_NUM_COLS + ec_cols + op_cols + TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS + OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS
    print(f"Feature count: {len(feat)}  (v21 had 130)")

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(float)
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=True)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining v23...")
    t0 = time.time()
    b = lgb.train(params, dtrain, num_boost_round=5000,
                  valid_sets=[dvalid], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {b.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = b.predict(test[feat], num_iteration=b.best_iteration)
    p_pos = np.clip(p_te, 0, None)

    print(f"\n=== v23 hold-out RMSE (v21 baseline in parens) ===")
    print(f"FULL   :  {rmse(y_te, p_pos):>7.2f} s   (v21 455.24)")
    clean = (y_te >= 30) & (y_te <= 7200)
    print(f"CLEAN  :  {rmse(y_te[clean], p_pos[clean]):>7.2f} s   (v21 283.85)   n={clean.sum():,}")
    tail = y_te > 7200
    print(f"TAIL   :  {rmse(y_te[tail], p_pos[tail]):>7.2f} s   (v21 17105.27)   n={tail.sum():,}")

    imp = pd.Series(b.feature_importance(importance_type="gain"),
                    index=feat).sort_values(ascending=False)
    print("\nNew arrival-feature ranks:")
    for c in ["arr_load_prev_15m", "arr_load_prev_30m", "arr_load_prev_60m",
              "arr_same_rwy_prev_15m", "arr_same_rwy_next_10m",
              "arr_taxi_in_mean_60m"]:
        r = list(imp.index).index(c) + 1
        print(f"  {c:26s} rank {r:>3d} / {len(feat)}   gain {imp[c]:.2e}")

    out_dir = os.path.join(ROOT, "models")
    b.save_model(os.path.join(out_dir, "lgbm_v23.txt"))
    with open(os.path.join(out_dir, "lgbm_v23.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_v23.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {out_dir}/lgbm_v23.*   total wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
