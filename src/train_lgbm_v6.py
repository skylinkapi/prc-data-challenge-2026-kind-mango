"""lgbm_v6 = v5 (best Optuna params) + data-quality cleaning.

Two additions to v5:
1. Drop training rows where BLOCK_TIME == SCHED_TIME (8,074 rows, Henri's
   fallback pattern; targets inflated by pushback delay).
2. Add `flt_null` binary flag as a feature (helps EHAM Jan 2026 which has
   4.96% _flt missing vs 1.65% in training).

Evaluated on the SAME unchanged hold-out as v5 so RMSE deltas are comparable.
"""
import glob
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import fit_encoders, apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}

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
    print("Loading...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    m["blk_ts"] = pd.to_datetime(m["BLOCK_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)

    # === Data-quality flag (BLOCK==SCHED fallback) ===
    dep["_bs_diff"] = (dep["blk_ts"] - dep["sched_ts"]).dt.total_seconds().abs()
    dep["_qual_bad"] = dep["_bs_diff"] < 1

    dep = dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)]
    dep = add_weather(dep)
    dep = add_congestion(dep, m[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train_slice = dep[~hold]
    encoders = fit_encoders(train_slice)
    dep = apply_encoders(dep, encoders)

    print(f"  {len(dep):,} DEP rows in {time.time()-t0:.1f}s")

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    # KEY CHANGE: drop bad rows from TRAIN only. Hold-out unchanged so RMSE
    # is directly comparable to v5.
    train_all = dep.loc[~hold]
    n_bad = train_all["_qual_bad"].sum()
    train = train_all[~train_all["_qual_bad"]].copy()
    test = dep.loc[hold].copy()
    print(f"Filtered {n_bad:,} BLOCK==SCHED rows from training")
    print(f"Train {len(train):,}   Hold-out {len(test):,}  (hold-out unchanged)")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = CAT_COLS + NUM_COLS_V1 + ec_cols + op_cols + TAXI_NUM_COLS
    print(f"Feature count: {len(feat)}  (v5 had 97)")

    y_tr, y_te = train["TAXITIME_SEC_mvt"].values, test["TAXITIME_SEC_mvt"].values
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=False)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining LightGBM v6 (v5 params + cleaned train + flt_null flag)...")
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=5000,
                        valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {booster.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = booster.predict(test[feat], num_iteration=booster.best_iteration)
    print(f"\nLGBM v6 overall RMSE: {rmse(y_te, p_te):.2f}s   (v5 = 295.49)")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].values, "y": y_te, "p": p_te}) \
        .groupby("apt", observed=True).apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(2)
    v5 = pd.Series({"LEMD":206.6,"LSZH":228.1,"LEBL":241.3,"EDDF":245.4,"EDDM":246.6,
                    "EHAM":250.5,"LTFM":293.6,"LFPG":331.0,"EGLL":362.0,"LIRF":462.2})
    tab["v5"] = v5
    tab["delta"] = (tab["rmse"] - tab["v5"]).round(2)
    print("\nPer-airport RMSE (v6 vs v5):")
    print(tab.sort_values("rmse").to_string())

    out_dir = os.path.join(ROOT, "models")
    booster.save_model(os.path.join(out_dir, "lgbm_v6.txt"))
    with open(os.path.join(out_dir, "lgbm_v6.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(out_dir, "lgbm_v6.encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nSaved -> {out_dir}/lgbm_v6.*")


if __name__ == "__main__":
    main()
