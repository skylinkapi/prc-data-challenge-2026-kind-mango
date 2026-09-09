"""Section 6.4: taxi-in drift meter.

Train the v21 recipe on 2025 arrivals (target = taxi-in time), score it on
2026 arrivals from ranking.parquet. If 2026 RMSE matches 2025 Jan+Jul hold-out
RMSE, no 2025->2026 drift. If it is worse by >5 %, drift is a real problem.

Simplified: only key numeric + categorical features. No OPDI. No congestion.
This is a diagnostic, not a submission model.
"""
import glob, os, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
RANK = os.path.join(ROOT, "submission", "ranking.parquet")

TARGET_ICAOS = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"]
HOLDOUT_MONTHS = {1, 7}

CAT_COLS = ["ADES_mvt", "ADEP_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
            "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt"]
NUM_COLS = ["hour", "dow", "month"]

PARAMS = {"objective": "regression", "metric": "rmse",
          "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 50,
          "feature_fraction": 0.85, "bagging_fraction": 0.9, "bagging_freq": 5,
          "verbosity": -1, "num_threads": -1}


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def prep(df):
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    df["hour"] = df["mvt_ts"].dt.hour
    df["dow"] = df["mvt_ts"].dt.dayofweek
    df["month"] = df["mvt_ts"].dt.month
    return df


def main():
    t0 = time.time()
    print("Loading 2025 ARR at target airports...")
    frames = []
    for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet"))):
        t = pd.read_parquet(f, columns=CAT_COLS + ["PHASE_mvt", "MVT_TIME_UTC_mvt",
                                                    "TAXITIME_SEC_mvt"])
        t = t[(t["PHASE_mvt"] == "ARR") & t["ADES_mvt"].isin(TARGET_ICAOS)]
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    df = prep(df)
    df = df[df["TAXITIME_SEC_mvt"].between(30, 7200)]
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    hold = df["month"].isin(HOLDOUT_MONTHS)
    train = df[~hold]
    test25 = df[hold]
    print(f"2025 train: {len(train):,}   2025 Jan+Jul: {len(test25):,}")

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(float)
    y_te25 = test25["TAXITIME_SEC_mvt"].values.astype(float)
    dtr = lgb.Dataset(train[CAT_COLS + NUM_COLS], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    for c in CAT_COLS:
        test25.loc[:, c] = pd.Categorical(test25[c], categories=train[c].cat.categories)
    dva = lgb.Dataset(test25[CAT_COLS + NUM_COLS], label=y_te25, categorical_feature=CAT_COLS, reference=dtr, free_raw_data=True)
    b = lgb.train(PARAMS, dtr, num_boost_round=800,
                  valid_sets=[dva], valid_names=["25"],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    p25 = b.predict(test25[CAT_COLS + NUM_COLS], num_iteration=b.best_iteration)
    rmse25 = rmse(y_te25, p25)
    print(f"\n2025 Jan+Jul hold-out RMSE: {rmse25:.2f} s")

    # Now score on 2026 ranking ARR
    print("\nLoading 2026 ARR from ranking.parquet...")
    r = pd.read_parquet(RANK, columns=CAT_COLS + ["PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
    r = r[(r["PHASE_mvt"] == "ARR") & r["ADES_mvt"].isin(TARGET_ICAOS)]
    r = prep(r)
    r = r[r["TAXITIME_SEC_mvt"].between(30, 7200)]
    for c in CAT_COLS:
        r[c] = pd.Categorical(r[c], categories=train[c].cat.categories)
    y_26 = r["TAXITIME_SEC_mvt"].values.astype(float)
    p26 = b.predict(r[CAT_COLS + NUM_COLS])
    rmse26 = rmse(y_26, p26)
    print(f"2026 ARR RMSE: {rmse26:.2f} s  ({len(r):,} rows)")

    delta = rmse26 - rmse25
    pct = delta / rmse25 * 100
    print(f"\ndelta (2026 - 2025): {delta:+.2f} s  ({pct:+.2f} %)")
    if pct > 5:
        print("=> DRIFT DETECTED: 2026 has meaningful distribution shift.")
    else:
        print("=> NO DRIFT: 2026 matches 2025 within 5 %.")

    # Per airport
    print("\nPer airport (2025 vs 2026):")
    for apt in TARGET_ICAOS:
        s25 = (test25["ADES_mvt"].astype(str) == apt).values
        s26 = (r["ADES_mvt"].astype(str) == apt).values
        if s25.sum() and s26.sum():
            r25 = rmse(y_te25[s25], p25[s25])
            r26 = rmse(y_26[s26], p26[s26])
            print(f"  {apt}: 2025 {r25:6.2f} ({s25.sum():>5})   2026 {r26:6.2f} ({s26.sum():>5})   delta {r26-r25:+6.2f}")

    print(f"\nWall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
