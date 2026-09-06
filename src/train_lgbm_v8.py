"""lgbm_v8 = v6 recipe with expanded target range [30, 21600].

Why: leaderboard v1 scored 562.4 vs our 295 hold-out. The gap is consistent
with ~1-2 % of scored rows having true taxi times > 7200 s that our model
was trained never to predict. This trains the model to see the same tail
values our clipped filter previously dropped.

Same clean-up as v6 (drop BLOCK==SCHED rows) so we do not learn the buggy
pattern. Same Optuna params as v5. Same flt_null feature as v6. Only the
[30, 7200] -> [30, 21600] training filter changes.
"""
import glob
import os
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}

# Broader range: 30 s to 6 hours (was 30 to 2 hours)
TARGET_MIN = 30
TARGET_MAX = 21600

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]
NUM_COLS_V1 = ["hour", "dow", "month", "sched_delay",
               "sknt", "gust_kt", "wind_cross_kt", "wind_head_kt",
               "vis_km", "low_vis", "very_low_vis",
               "ceiling_ft", "low_ceiling",
               "wx_precip", "wx_snow", "wx_thunder", "wx_freezing",
               "dep_load_prev_15m", "dep_load_prev_30m", "dep_load_prev_60m",
               "arr_load_prev_15m", "arr_load_prev_30m", "arr_load_prev_60m",
               "dep_same_rwy_prev_15m", "dep_same_rwy_prev_30m",
               "dep_same_rwy_prev_60m", "dep_queue_next_10m",
               "flt_null"]

BEST_PARAMS = {
    "learning_rate": 0.022810159868114487,
    "num_leaves": 440,
    "min_data_in_leaf": 76,
    "feature_fraction": 0.6737480156722359,
    "bagging_fraction": 0.9366340145580665,
    "lambda_l1": 0.0887022084743886,
    "lambda_l2": 0.3423614383968606,
    "min_gain_to_split": 1.8309877581737415,
}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    t0 = time.time()
    print("Loading movements...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    m["blk_ts"] = pd.to_datetime(m["BLOCK_TIME_UTC_mvt"], errors="coerce")
    print(f"  {len(m):,} in {time.time()-t0:.1f}s")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep["_bs_diff"] = (dep["blk_ts"] - dep["sched_ts"]).dt.total_seconds().abs()
    dep["_qual_bad"] = dep["_bs_diff"] < 1

    # Report the change in retained rows
    n_narrow = dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)].shape[0]
    n_broad = dep[dep["TAXITIME_SEC_mvt"].between(TARGET_MIN, TARGET_MAX)].shape[0]
    n_extra = n_broad - n_narrow
    print(f"\nTarget filter [30, 7200]: {n_narrow:,}   [30, {TARGET_MAX}]: {n_broad:,}   extra: {n_extra:,}")

    dep = dep[dep["TAXITIME_SEC_mvt"].between(TARGET_MIN, TARGET_MAX)]

    t = time.time(); print("Adding features...");
    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)
    print(f"  {time.time()-t:.1f}s")

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train_all = dep.loc[~hold]
    n_bad = train_all["_qual_bad"].sum()
    train = train_all[~train_all["_qual_bad"]].copy()
    test = dep.loc[hold].copy()
    print(f"Filtered {n_bad:,} BLOCK==SCHED from training")
    print(f"Train {len(train):,}   Hold-out {len(test):,}  (unchanged)")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS

    y_tr, y_te = train["TAXITIME_SEC_mvt"].values, test["TAXITIME_SEC_mvt"].values
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining LightGBM v8 (broader target range)...")
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=5000,
                        valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {booster.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = booster.predict(test[feat], num_iteration=booster.best_iteration)

    # Report on the SAME hold-out set as v5/v6 (rows with y in [30, 7200])
    narrow_mask = (y_te >= 30) & (y_te <= 7200)
    print(f"\nOverall v8 RMSE on broader hold-out ({len(y_te):,} rows, y up to {TARGET_MAX}s): "
          f"{rmse(y_te, p_te):.2f}s")
    print(f"Overall v8 RMSE on narrow hold-out (y<=7200 only, {narrow_mask.sum():,} rows): "
          f"{rmse(y_te[narrow_mask], p_te[narrow_mask]):.2f}s   (v5 = 295.49)")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].astype(str).values,
                        "y": y_te, "p": p_te}) \
        .groupby("apt").apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(2)
    v5 = pd.Series({"LEMD":206.6,"LSZH":228.1,"LEBL":241.3,"EDDF":245.4,"EDDM":246.6,
                    "EHAM":250.5,"LTFM":293.6,"LFPG":331.0,"EGLL":362.0,"LIRF":462.2})
    tab["v5"] = v5
    tab["delta"] = (tab["rmse"] - tab["v5"]).round(2)
    print("\nPer-airport RMSE (v8 on broader hold-out vs v5 on narrow):")
    print(tab.sort_values("rmse").to_string())

    # Prediction distribution
    print(f"\nPrediction distribution:")
    print(pd.Series(p_te).describe(percentiles=[.5, .95, .99, .999]).round(1).to_string())
    print(f"Predictions > 7200: {(p_te > 7200).sum():,}   > 10800: {(p_te > 10800).sum():,}   > 14400: {(p_te > 14400).sum():,}")
    print(f"True values > 7200: {(y_te > 7200).sum():,}   > 10800: {(y_te > 10800).sum():,}   > 14400: {(y_te > 14400).sum():,}")

    out_dir = os.path.join(ROOT, "models")
    booster.save_model(os.path.join(out_dir, "lgbm_v8.txt"))
    with open(os.path.join(out_dir, "lgbm_v8.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_v8.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {out_dir}/lgbm_v8.*")


if __name__ == "__main__":
    main()
