"""lgbm_v7 = v5 params + sample-weighted training toward winter/summer months.

Motivation: leaderboard v1 scored 562.4 vs our 295 hold-out. The 2026 ranking
window is Jan (~45 %) + Jul (~55 %). Our training months (Feb-Jun, Aug-Dec)
have a shoulder-season sked_delay distribution 200-360 s lower than Jan 2026
at 5 airports. Weighting training rows toward the seasons closest to the
scoring months tilts the fit toward the target distribution.

Weight scheme (motivated by season, not sched_delay drift directly, so it does
not overfit to any single measurement):
  Feb, Mar, Nov, Dec  -> 2.0    (winter, close to Jan)
  Jun, Aug            -> 2.0    (summer, close to Jul)
  Apr, May, Sep, Oct  -> 1.0    (shoulder)

Features frozen from v5 (uses cached features written by tune_lgbm.py).
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "models", "features_cache")
MODELS_DIR = os.path.join(ROOT, "models")

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]

MONTH_WEIGHTS = {2: 2.0, 3: 2.0, 11: 2.0, 12: 2.0,
                 6: 2.0, 8: 2.0,
                 4: 1.0, 5: 1.0, 9: 1.0, 10: 1.0}

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
    print("Loading cached features...")
    train = pd.read_parquet(CACHE + ".train.parquet")
    test = pd.read_parquet(CACHE + ".test.parquet")
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().splitlines()
    print(f"  train {len(train):,}   test {len(test):,}   feats {len(feat)}   ({time.time()-t0:.1f}s)")

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    train["weight"] = train["month"].map(MONTH_WEIGHTS).fillna(1.0).astype(float)
    print("\nWeight distribution:")
    print(train.groupby("month")["weight"].agg(["count", "first"]).to_string())
    print(f"Sum weights: {train['weight'].sum():,.0f}  (vs {len(train):,} unweighted)")

    y_tr, y_te = train["TAXITIME_SEC_mvt"].values, test["TAXITIME_SEC_mvt"].values
    w_tr = train["weight"].values

    dtrain = lgb.Dataset(train[feat], label=y_tr, weight=w_tr,
                         categorical_feature=CAT_COLS, free_raw_data=False)
    dvalid = lgb.Dataset(test[feat], label=y_te,
                         categorical_feature=CAT_COLS, reference=dtrain, free_raw_data=False)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining LightGBM v7 (winter-weighted)...")
    t0 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=5000,
                        valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {booster.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = booster.predict(test[feat], num_iteration=booster.best_iteration)
    print(f"\nLGBM v7 hold-out RMSE: {rmse(y_te, p_te):.2f}s   (v5 = 295.49)")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].astype(str).values, "y": y_te, "p": p_te}) \
        .groupby("apt").apply(
            lambda g: pd.Series({"n": len(g), "rmse": rmse(g["y"], g["p"])}),
            include_groups=False).round(2)
    v5 = pd.Series({"LEMD":206.6,"LSZH":228.1,"LEBL":241.3,"EDDF":245.4,"EDDM":246.6,
                    "EHAM":250.5,"LTFM":293.6,"LFPG":331.0,"EGLL":362.0,"LIRF":462.2})
    tab["v5"] = v5
    tab["delta"] = (tab["rmse"] - tab["v5"]).round(2)
    print("\nPer-airport RMSE (v7 vs v5):")
    print(tab.sort_values("rmse").to_string())

    # Per-hold-out-month breakdown (Jan vs Jul separately)
    print("\nHold-out per-month RMSE:")
    for month in [1, 7]:
        mask = test["month"] == month
        r = rmse(y_te[mask], p_te[mask])
        n = mask.sum()
        print(f"  Month {month}: n={n:,}  RMSE={r:.2f}s")

    booster.save_model(os.path.join(MODELS_DIR, "lgbm_v7.txt"))
    with open(os.path.join(MODELS_DIR, "lgbm_v7.features.txt"), "w") as f:
        f.write("\n".join(feat))
    # v5 encoders (same features)
    import shutil
    shutil.copy(os.path.join(MODELS_DIR, "lgbm_v5.encoders.pkl"),
                os.path.join(MODELS_DIR, "lgbm_v7.encoders.pkl"))
    print(f"\nSaved -> {MODELS_DIR}/lgbm_v7.*")


if __name__ == "__main__":
    main()
