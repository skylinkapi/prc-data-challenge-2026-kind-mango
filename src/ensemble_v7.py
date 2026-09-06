"""Grid-search weights over lgbm_v5, v6, v7 on hold-out.

v5 and v7 share the 97-feature layout from the tune_lgbm cache.
v6 needs the flt_null column added.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "models", "features_cache")
MODELS_DIR = os.path.join(ROOT, "models")

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def predict(model_tag: str, X: pd.DataFrame) -> np.ndarray:
    b = lgb.Booster(model_file=os.path.join(MODELS_DIR, f"lgbm_{model_tag}.txt"))
    with open(os.path.join(MODELS_DIR, f"lgbm_{model_tag}.features.txt")) as f:
        feat = f.read().splitlines()
    return b.predict(X[feat])


def search_weights(y: np.ndarray, preds: dict, step: float = 0.02) -> tuple:
    keys = list(preds.keys())
    best = (float("inf"), None)
    vals = np.arange(0, 1.0 + 1e-9, step)
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                continue
            w3 = max(0.0, w3)
            p = w1 * preds[keys[0]] + w2 * preds[keys[1]] + w3 * preds[keys[2]]
            r = rmse(y, p)
            if r < best[0]:
                best = (r, {keys[0]: w1, keys[1]: w2, keys[2]: w3})
    return best


def main():
    print("Loading cached hold-out...")
    train = pd.read_parquet(CACHE + ".train.parquet")
    test = pd.read_parquet(CACHE + ".test.parquet")
    for c in CAT_COLS:
        test[c] = pd.Categorical(test[c], categories=train[c].astype("category").cat.categories)
    del train

    test["flt_null"] = test["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    apt = test["ADEP_mvt"].astype(str).values

    print("Predicting v5, v6, v7...")
    preds = {v: predict(v, test) for v in ["v5", "v6", "v7"]}
    for v, p in preds.items():
        print(f"  {v}: RMSE {rmse(y, p):.3f}s")

    simple = (preds["v5"] + preds["v6"] + preds["v7"]) / 3
    print(f"\nSimple avg (1/3): {rmse(y, simple):.3f}s")

    print("\nGrid search weights (step 0.02)...")
    r, w = search_weights(y, preds, step=0.02)
    print(f"  best: {r:.3f}s   weights: {w}")

    ens = sum(w[v] * preds[v] for v in preds)

    print("\nPer-airport (ensemble vs v5):")
    tab = pd.DataFrame({"apt": apt, "y": y, "p_ens": ens, "p_v5": preds["v5"]}) \
        .groupby("apt").apply(
            lambda g: pd.Series({
                "n": len(g),
                "rmse_ens": rmse(g["y"], g["p_ens"]),
                "rmse_v5": rmse(g["y"], g["p_v5"]),
            }), include_groups=False).round(2)
    tab["delta"] = (tab["rmse_ens"] - tab["rmse_v5"]).round(2)
    print(tab.sort_values("rmse_ens").to_string())

    with open(os.path.join(MODELS_DIR, "ensemble_weights_v567.txt"), "w") as f:
        for v, wt in w.items():
            f.write(f"{v}: {wt}\n")
    print(f"\nSaved -> {MODELS_DIR}/ensemble_weights_v567.txt")


if __name__ == "__main__":
    main()
