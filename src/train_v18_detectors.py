"""Section 6.2 step 2-3: per-airport fallback detectors.

For each airport with at least 300 training positives (|y-sd|<60 AND sd>1800):
  - Honest split: train Feb-Jun+Aug-Oct, early stop Nov+Dec, evaluate Jan+Jul once.
  - Inputs: NO raw STAND_mvt or ADES_mvt. Use derived prefixes and region.
  - min_data_in_leaf = 50.
  - Report reliability by decile.
"""
import glob, os, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")
TARGET_ICAOS = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"]

TRAIN_MONTHS = {2, 3, 4, 5, 6, 8, 9, 10}
STOP_MONTHS = {11, 12}
TEST_MONTHS = {1, 7}
SD_MIN = 1800  # gate for both training and application

CAT_INPUTS = ["AIRCRAFT_OPERATOR_flt", "flt_prefix", "stand_prefix", "RUNWAY_mvt",
              "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_mvt",
              "WK_TBL_CAT_flt", "ades_region"]
NUM_INPUTS = ["sd", "mvt_eobt1", "eobt1_sched", "eobt1_iobt",
              "hour", "dow", "sd_mod_86400", "flt_id_null"]

PARAMS = {"objective": "binary", "metric": "auc",
          "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 5,
          "verbosity": -1, "num_threads": -1}


def load_all():
    frames = []
    for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet"))):
        cols = ["ADEP_mvt", "ADES_mvt", "PHASE_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt",
                "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt",
                "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_mvt",
                "WK_TBL_CAT_flt",
                "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt",
                "TAXITIME_SEC_mvt"]
        frames.append(pd.read_parquet(f, columns=cols))
    m = pd.concat(frames, ignore_index=True)
    m = m[(m["PHASE_mvt"] == "DEP") & m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    m["eobt_ts"] = pd.to_datetime(m["EOBT_1_flt"], errors="coerce")
    m["iobt_ts"] = pd.to_datetime(m["IOBT_flt"], errors="coerce")
    m["sd"] = (m["mvt_ts"] - m["sched_ts"]).dt.total_seconds()
    m["y"] = m["TAXITIME_SEC_mvt"].astype(float)
    m["mvt_eobt1"] = (m["mvt_ts"] - m["eobt_ts"]).dt.total_seconds()
    m["eobt1_sched"] = (m["eobt_ts"] - m["sched_ts"]).dt.total_seconds()
    m["eobt1_iobt"] = (m["eobt_ts"] - m["iobt_ts"]).dt.total_seconds()
    m["hour"] = m["mvt_ts"].dt.hour
    m["dow"] = m["mvt_ts"].dt.dayofweek
    m["month"] = m["mvt_ts"].dt.month
    m["sd_mod_86400"] = m["sd"].mod(86400).where(m["sd"].notna())
    m["flt_prefix"] = m["FLIGHT_mvt"].astype(str).str[:3].fillna("UNK")
    m["stand_prefix"] = m["STAND_mvt"].astype(str).str.extract(r"^([A-Za-z]+)")[0].fillna("UNK")
    m["ades_region"] = m["ADES_mvt"].astype(str).str[:2].fillna("UNK")
    m["flt_id_null"] = m["FLIGHT_ID_mvt"].isna().astype(np.int8)
    m["is_fb"] = ((m["y"] - m["sd"]).abs() < 60).astype(np.int8)
    return m


def main():
    t_all = time.time()
    print("Loading + engineering...")
    m = load_all()
    print(f"Total DEP rows at targets: {len(m):,}")

    results = {}
    for apt in TARGET_ICAOS:
        sub = m[(m["ADEP_mvt"] == apt) & (m["sd"] > SD_MIN) & m["y"].notna()].copy()
        # Pre-cast categoricals on the FULL subset first, so train/stop/test share categories.
        for c in CAT_INPUTS:
            sub[c] = sub[c].astype("category")
        train = sub[sub["month"].isin(TRAIN_MONTHS)].copy()
        stop = sub[sub["month"].isin(STOP_MONTHS)].copy()
        test = sub[sub["month"].isin(TEST_MONTHS)].copy()

        n_train_pos = int(train["is_fb"].sum())
        n_stop_pos = int(stop["is_fb"].sum())
        n_test_pos = int(test["is_fb"].sum())
        qualifies = n_train_pos >= 300

        print(f"\n{apt}: train {len(train):,} (+{n_train_pos})  "
              f"stop {len(stop):,} (+{n_stop_pos})  test {len(test):,} (+{n_test_pos})  "
              f"qualifies={qualifies}")
        if not qualifies:
            results[apt] = {"qualifies": False,
                            "n_train": len(train), "n_train_pos": n_train_pos}
            continue

        # Restrict test/stop categoricals to those seen in train (avoid new-category mismatch)
        for c in CAT_INPUTS:
            train_cats = train[c].cat.categories
            stop[c] = pd.Categorical(stop[c].astype(object), categories=train_cats)
            test[c] = pd.Categorical(test[c].astype(object), categories=train_cats)

        feat = CAT_INPUTS + NUM_INPUTS
        dtr = lgb.Dataset(train[feat], label=train["is_fb"].values,
                          categorical_feature=CAT_INPUTS, free_raw_data=True)
        dst = lgb.Dataset(stop[feat], label=stop["is_fb"].values,
                          categorical_feature=CAT_INPUTS, reference=dtr, free_raw_data=True)
        b = lgb.train(PARAMS, dtr, num_boost_round=2000,
                      valid_sets=[dst], valid_names=["stop"],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        p_te = b.predict(test[feat], num_iteration=b.best_iteration)
        auc_te = float(roc_auc_score(test["is_fb"], p_te)) if test["is_fb"].nunique() > 1 else np.nan
        auc_stop = float(roc_auc_score(stop["is_fb"], b.predict(stop[feat], num_iteration=b.best_iteration))) if stop["is_fb"].nunique() > 1 else np.nan

        # Reliability by decile
        df = pd.DataFrame({"p": p_te, "y": test["is_fb"].values})
        try:
            df["dec"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
        except ValueError:
            df["dec"] = pd.qcut(df["p"], min(10, df["p"].nunique()), labels=False, duplicates="drop")
        tab = df.groupby("dec").agg(n=("y","size"), mean_p=("p","mean"), obs=("y","mean"))
        tab["overshoot"] = (tab["mean_p"] - tab["obs"]).round(3)
        worst_over = float(tab["overshoot"].max())

        print(f"  best iter {b.best_iteration}  stop AUC {auc_stop:.3f}  test AUC {auc_te:.3f}  "
              f"worst decile overshoot {worst_over:.3f}")
        results[apt] = {
            "qualifies": True, "test_auc": auc_te, "stop_auc": auc_stop,
            "best_iter": b.best_iteration,
            "n_train_pos": n_train_pos, "n_test_pos": n_test_pos,
            "worst_decile_overshoot": worst_over,
        }
        b.save_model(os.path.join(MODELS, f"v18_det_{apt}.txt"))

    with open(os.path.join(MODELS, "v18_detectors_meta.json"), "w") as f:
        json.dump({"sd_min": SD_MIN, "cat_inputs": CAT_INPUTS,
                   "num_inputs": NUM_INPUTS, "params": PARAMS,
                   "results": results}, f, indent=2)
    print(f"\nDone. wall {time.time()-t_all:.1f}s")

    print("\n=== Summary ===")
    for apt in TARGET_ICAOS:
        r = results.get(apt, {})
        if not r.get("qualifies"):
            print(f"  {apt}: DISQUALIFIED (need 300+ positives, have {r.get('n_train_pos',0)})")
            continue
        flag = "PASS" if r["worst_decile_overshoot"] < 0.1 else "FAIL"
        print(f"  {apt}: AUC {r['test_auc']:.3f}  worst overshoot {r['worst_decile_overshoot']:+.3f}  "
              f"decile-{flag}")


if __name__ == "__main__":
    main()
