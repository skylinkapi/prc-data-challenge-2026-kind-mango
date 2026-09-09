"""Section 6.3: LIRF no-flight-record detector for 3,600 < sd <= 14,400.

Target: |y - sd| < 60. Inputs: flight-number prefix (3 char), stand prefix,
aircraft type, runway, hour, sd, sd_mod_86400.
Honest split: train Feb-Jun + Aug-Oct, early stop Nov+Dec, evaluate Jan+Jul ONCE.
Reports per-decile reliability. Mix formula at deploy: p*sd + (1-p)*1150.
"""
import glob, os, pickle, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")

TRAIN_MONTHS = {2, 3, 4, 5, 6, 8, 9, 10}
STOP_MONTHS = {11, 12}
TEST_MONTHS = {1, 7}
SD_LO, SD_HI = 3600, 14400

CAT_INPUTS = ["flt_prefix", "stand_prefix", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt"]
NUM_INPUTS = ["sd", "sd_mod_86400", "hour", "dow", "mvt_eobt1", "eobt1_sched"]

PARAMS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
    "feature_fraction": 0.85, "bagging_fraction": 0.9, "bagging_freq": 5,
    "verbosity": -1, "num_threads": -1,
}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load_lirf():
    frames = []
    for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet"))):
        cols = ["ADEP_mvt", "PHASE_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt",
                "STAND_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
                "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt",
                "TAXITIME_SEC_mvt"]
        frames.append(pd.read_parquet(f, columns=cols))
    m = pd.concat(frames, ignore_index=True)
    m = m[(m["PHASE_mvt"] == "DEP") & (m["ADEP_mvt"] == "LIRF")].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    m["eobt_ts"] = pd.to_datetime(m["EOBT_1_flt"], errors="coerce")
    m["sd"] = (m["mvt_ts"] - m["sched_ts"]).dt.total_seconds()
    m["y"] = m["TAXITIME_SEC_mvt"].astype(float)
    m["month"] = m["mvt_ts"].dt.month
    m["hour"] = m["mvt_ts"].dt.hour
    m["dow"] = m["mvt_ts"].dt.dayofweek
    m["sd_mod_86400"] = m["sd"].mod(86400).where(m["sd"].notna())
    m["mvt_eobt1"] = (m["mvt_ts"] - m["eobt_ts"]).dt.total_seconds()
    m["eobt1_sched"] = (m["eobt_ts"] - m["sched_ts"]).dt.total_seconds()
    m["flt_prefix"] = m["FLIGHT_mvt"].astype(str).str[:3].fillna("UNK")
    m["stand_prefix"] = m["STAND_mvt"].astype(str).str.extract(r"^([A-Za-z]+)")[0].fillna("UNK")
    m["is_fb"] = ((m["y"] - m["sd"]).abs() < 60).astype(np.int8)
    m["is_24h"] = (m["y"] > 80000).astype(np.int8)

    # Cell filter: LIRF null-flight, 3600 < sd <= 14400
    cell = m[m["FLIGHT_ID_mvt"].isna() &
             (m["sd"] > SD_LO) & (m["sd"] <= SD_HI) &
             m["y"].notna()].copy()
    return cell


def main():
    t0 = time.time()
    print("Loading LIRF null-flight cell...")
    m = load_lirf()
    print(f"Cell rows: {len(m):,}")
    print(f"  fb: {m['is_fb'].sum():,}   24h: {m['is_24h'].sum():,}   normal: {(~m['is_fb'].astype(bool) & ~m['is_24h'].astype(bool)).sum():,}")

    # Group normal mean (for mix formula at deploy)
    norm_rows = m[(m["is_fb"] == 0) & (m["is_24h"] == 0)]
    normal_mean = float(norm_rows["y"].mean())
    print(f"Group normal mean: {normal_mean:.1f} s   (doc: ~1150)")

    for c in CAT_INPUTS:
        m[c] = m[c].astype("category")

    train = m[m["month"].isin(TRAIN_MONTHS)]
    stop  = m[m["month"].isin(STOP_MONTHS)]
    test  = m[m["month"].isin(TEST_MONTHS)]
    print(f"Split: train={len(train):,}  stop={len(stop):,}  test={len(test):,}   "
          f"train fb+={train['is_fb'].sum():,}")

    # Align categoricals across splits to training cats
    for c in CAT_INPUTS:
        cats = train[c].cat.categories
        stop.loc[:, c] = pd.Categorical(stop[c], categories=cats)
        test.loc[:, c] = pd.Categorical(test[c], categories=cats)

    feat = CAT_INPUTS + NUM_INPUTS
    dtr = lgb.Dataset(train[feat], label=train["is_fb"].values,
                      categorical_feature=CAT_INPUTS, free_raw_data=True)
    dstop = lgb.Dataset(stop[feat], label=stop["is_fb"].values,
                        categorical_feature=CAT_INPUTS, reference=dtr, free_raw_data=True)
    b = lgb.train(PARAMS, dtr, num_boost_round=1500,
                  valid_sets=[dstop], valid_names=["stop"],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"  best iter: {b.best_iteration}")

    p_te = b.predict(test[feat], num_iteration=b.best_iteration)
    auc_te = float(roc_auc_score(test["is_fb"], p_te))
    print(f"\nTest AUC (Jan+Jul, one shot): {auc_te:.4f}")

    # Reliability table by decile
    print("\nReliability by decile:")
    df = pd.DataFrame({"p": p_te, "y": test["is_fb"].values, "sd": test["sd"].values, "label": test["y"].values})
    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    tab = df.groupby("decile").agg(
        n=("y", "size"),
        mean_p=("p", "mean"),
        obs_share=("y", "mean"),
    )
    tab["diff"] = (tab["mean_p"] - tab["obs_share"]).round(3)
    print(tab.round(3).to_string())

    # Applied-metric: mix formula vs baseline (constant normal_mean)
    y_true = test["y"].values.astype(float)
    sd_te = test["sd"].values.astype(float)
    baseline = np.full(len(test), normal_mean)   # naive: always normal
    v21_like_group_mean = np.full(len(test), norm_rows["y"].mean())  # what v21 does ~= group mean
    mix = p_te * sd_te + (1 - p_te) * normal_mean

    # Compare against a fair v21 baseline: mean of ALL rows in the group (including fb)
    v21_bias = float(m[m["month"].isin(TRAIN_MONTHS)]["y"].mean())
    v21_ish = np.full(len(test), v21_bias)

    print(f"\nApplied on Jan+Jul test cell (n={len(test)}):")
    print(f"  baseline (const normal_mean 1150):  RMSE {rmse(y_true, baseline):>7.2f}   MSE_sum {float(np.sum((y_true-baseline)**2)):>10.0f}")
    print(f"  v21-ish (const group mean {v21_bias:.0f}): RMSE {rmse(y_true, v21_ish):>7.2f}   MSE_sum {float(np.sum((y_true-v21_ish)**2)):>10.0f}")
    print(f"  mix p*sd + (1-p)*1150:              RMSE {rmse(y_true, mix):>7.2f}   MSE_sum {float(np.sum((y_true-mix)**2)):>10.0f}")

    b.save_model(os.path.join(MODELS, "lirf_noflt_detector.txt"))
    with open(os.path.join(MODELS, "lirf_noflt_detector_meta.json"), "w") as f:
        json.dump({"normal_mean": normal_mean, "test_auc": auc_te,
                   "sd_lo": SD_LO, "sd_hi": SD_HI,
                   "cat_inputs": CAT_INPUTS, "num_inputs": NUM_INPUTS,
                   "best_iter": b.best_iteration,
                   "train_flt_prefix": sorted(train["flt_prefix"].cat.categories.tolist()),
                   "train_stand_prefix": sorted(train["stand_prefix"].cat.categories.tolist()),
                   "train_aircraft": sorted(train["AIRCRAFT_TYPE_mvt"].cat.categories.tolist()),
                   "train_runway": sorted(train["RUNWAY_mvt"].cat.categories.tolist()),
                   }, f)
    print(f"\nSaved -> {MODELS}/lirf_noflt_detector.*   wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
