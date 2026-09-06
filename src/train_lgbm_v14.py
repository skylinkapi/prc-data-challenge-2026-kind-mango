"""lgbm_v14 = v5 features (from cache) + OOF p_tail feature.

Uses cached features (v5-era: 97 cols) since building the full v11 feature set
runs out of memory. The purpose is to test whether OOF p_tail (leak-free)
adds signal to a regressor that already has 97 baseline features.
"""
import os
import time
import gc
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "models", "features_cache")
MODELS_DIR = os.path.join(ROOT, "models")

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]

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
    print("Loading cache + p_tail OOF...")
    train = pd.read_parquet(CACHE + ".train.parquet")
    test = pd.read_parquet(CACHE + ".test.parquet")
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().splitlines()

    ptrain = pd.read_parquet(os.path.join(MODELS_DIR, "p_tail_train_oof.parquet"))
    ptest = pd.read_parquet(os.path.join(MODELS_DIR, "p_tail_test.parquet"))
    train["p_tail"] = ptrain["p_tail"].values
    test["p_tail"] = ptest["p_tail"].values
    feat = feat + ["p_tail"]
    print(f"  train {len(train):,}   test {len(test):,}   feats {len(feat)}")

    for c in feat:
        if c in CAT_COLS: continue
        if c in train.columns and train[c].dtype == np.float64:
            train[c] = train[c].astype(np.float32)
        if c in test.columns and test[c].dtype == np.float64:
            test[c] = test[c].astype(np.float32)
    gc.collect()

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    y_tr = train["TAXITIME_SEC_mvt"].values.astype(np.float32)
    y_te = test["TAXITIME_SEC_mvt"].values.astype(np.float32)
    dtrain = lgb.Dataset(train[feat], label=y_tr, categorical_feature=CAT_COLS, free_raw_data=True)
    dvalid = lgb.Dataset(test[feat],  label=y_te, categorical_feature=CAT_COLS,
                         reference=dtrain, free_raw_data=True)
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "bagging_freq": 5, "verbosity": -1, "num_threads": -1}

    print("\nTraining v14 (v5-cache + p_tail OOF)...")
    t0 = time.time()
    b = lgb.train(params, dtrain, num_boost_round=3000,
                  valid_sets=[dvalid], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f"  best iter: {b.best_iteration}   time: {time.time()-t0:.1f}s")

    p_te = b.predict(test[feat], num_iteration=b.best_iteration)
    r = rmse(y_te, p_te)
    print(f"\nLGBM v14 hold-out RMSE: {r:.2f}s   (v5 baseline = 295.49, v11 = 294.18)")

    # Compare with v5 predictions (same features, no p_tail)
    v5_b = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lgbm_v5.txt"))
    p_v5 = v5_b.predict(test[feat[:-1]])
    print(f"v5 (same cache no p_tail): {rmse(y_te, p_v5):.2f}s")

    tab = pd.DataFrame({"apt": test["ADEP_mvt"].astype(str).values,
                        "y": y_te, "p14": p_te, "p5": p_v5}) \
        .groupby("apt").apply(
            lambda g: pd.Series({"n": len(g),
                                 "rmse_v14": rmse(g["y"], g["p14"]),
                                 "rmse_v5":  rmse(g["y"], g["p5"])}),
            include_groups=False).round(2)
    tab["delta_vs_v5"] = (tab["rmse_v14"] - tab["rmse_v5"]).round(2)
    print("\nPer-airport (v14 vs v5, both from same feature cache):")
    print(tab.sort_values("rmse_v14").to_string())

    imp = pd.Series(b.feature_importance(importance_type="gain"),
                    index=feat).sort_values(ascending=False)
    r = list(imp.index).index("p_tail") + 1
    print(f"\np_tail rank: {r} / {len(feat)}  gain {imp['p_tail']:.2e}")

    b.save_model(os.path.join(MODELS_DIR, "lgbm_v14.txt"))
    with open(os.path.join(MODELS_DIR, "lgbm_v14.features.txt"), "w") as f:
        f.write("\n".join(feat))
    print(f"\nSaved -> {MODELS_DIR}/lgbm_v14.*")


if __name__ == "__main__":
    main()
