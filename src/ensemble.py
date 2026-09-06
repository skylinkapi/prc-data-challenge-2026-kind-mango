"""Ensemble v4 + v5 + v6 predictions on the hold-out. Find the best weights
via simple constrained search (weights sum to 1, non-negative), then apply to
ranking.parquet to produce the ensemble submission.
"""
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "models", "features_cache")
MODELS_DIR = os.path.join(ROOT, "models")

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load_predict(booster_path: str, feat_path: str, X: pd.DataFrame,
                 y_present: bool = True) -> np.ndarray:
    b = lgb.Booster(model_file=booster_path)
    with open(feat_path) as f:
        feat = f.read().splitlines()
    missing = [c for c in feat if c not in X.columns]
    if missing:
        raise RuntimeError(f"Missing cols for {booster_path}: {missing}")
    return b.predict(X[feat])


def best_weights_grid(y: np.ndarray, preds: dict, step: float = 0.05) -> tuple:
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
                best = (r, (w1, w2, w3))
    return best


def main():
    print("Loading cached hold-out features...")
    test = pd.read_parquet(CACHE + ".test.parquet")
    train = pd.read_parquet(CACHE + ".train.parquet")  # only needed for categorical vocab
    for c in CAT_COLS:
        cats = train[c].astype("category").cat.categories
        test[c] = pd.Categorical(test[c], categories=cats)
    del train

    # v6 needs flt_null; add it
    test["flt_null"] = test["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    apt = test["ADEP_mvt"].astype(str).values

    print("Predicting with v4, v5, v6 on hold-out...")
    preds = {}
    for v in ["v4", "v5", "v6"]:
        preds[v] = load_predict(
            os.path.join(MODELS_DIR, f"lgbm_{v}.txt"),
            os.path.join(MODELS_DIR, f"lgbm_{v}.features.txt"),
            test,
        )

    print("\nSingle-model RMSE:")
    for v, p in preds.items():
        print(f"  {v}: {rmse(y, p):.3f}s")

    print(f"\nSimple average (1/3 each): {rmse(y, (preds['v4']+preds['v5']+preds['v6'])/3):.3f}s")

    print("\nGrid searching optimal weights (step=0.05)...")
    r, w = best_weights_grid(y, preds, step=0.05)
    print(f"  best RMSE = {r:.3f}s   weights (v4, v5, v6) = ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})")

    print("\nFine grid (step=0.02)...")
    r_fine, w_fine = best_weights_grid(y, preds, step=0.02)
    print(f"  best RMSE = {r_fine:.3f}s   weights = ({w_fine[0]:.3f}, {w_fine[1]:.3f}, {w_fine[2]:.3f})")

    w4, w5, w6 = w_fine
    p_ens = w4 * preds["v4"] + w5 * preds["v5"] + w6 * preds["v6"]

    print("\nPer-airport RMSE (ensemble vs v5):")
    v5_ref = pd.Series({"LEMD":206.6,"LSZH":228.1,"LEBL":241.3,"EDDF":245.4,"EDDM":246.6,
                        "EHAM":250.5,"LTFM":293.6,"LFPG":331.0,"EGLL":362.0,"LIRF":462.2})
    per = pd.DataFrame({"apt": apt, "y": y, "p_ens": p_ens, "p_v5": preds["v5"]})
    tab = per.groupby("apt").apply(
        lambda g: pd.Series({
            "n": len(g),
            "rmse_ens": rmse(g["y"], g["p_ens"]),
            "rmse_v5":  rmse(g["y"], g["p_v5"]),
        }), include_groups=False).round(2)
    tab["delta"] = (tab["rmse_ens"] - tab["rmse_v5"]).round(2)
    print(tab.sort_values("rmse_ens").to_string())

    # Save weights for the submission step
    with open(os.path.join(MODELS_DIR, "ensemble_weights.txt"), "w") as f:
        f.write(f"v4: {w4}\nv5: {w5}\nv6: {w6}\n")
    print(f"\nSaved weights -> {MODELS_DIR}/ensemble_weights.txt")

    return w_fine


if __name__ == "__main__":
    main()
