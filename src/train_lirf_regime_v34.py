"""v34 = v23 recipe with OOF and scoring maps built on fit months only.

Audit Item 2 (F2). The v23 fbrate OOF pool spanned all 12 months. Each
training row got features derived from rows in {1, 7} — the hold-out — so
the p_fb classifier's training features carried hold-out signal.

This script:
  - drops {1, 7} from the frame before OOF
  - computes base_rate on fit months only
  - trains p_fb on {2..6, 8..10}, early-stops on {11, 12}
  - fits isotonic on {11, 12}
  - saves lgbm_p_fb_lirf_v34.txt and lirf_regime_v34.* artefacts

The R_norm_LIRF members (v33 recipe, 5 seeds) are unchanged because
train_lirf_regime.py already excludes {1, 7} from training.
"""
import os, pickle, time, gc
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss

import pandas as pd

from features_congestion_v2 import CONG_V2_NUM_COLS
from features_eurocontrol import eurocontrol_numeric_cols
from features_operator import operator_numeric_cols
from features_taxi_distance import TAXI_NUM_COLS
from features_advanced import ADV_NUM_COLS
from features_osm_path import OSM_PATH_NUM_COLS
from features_opdi import OPDI_NUM_COLS
from features_opdi_live import OPDI_LIVE_NUM_COLS
from features_turnaround import TURN_NUM_COLS
from features_disruption import DISR_NUM_COLS
from train_lgbm_v21 import CAT_COLS
from train_lgbm_v23 import BASE_NUM_COLS
from train_r_all_v21 import SIGNED_LOG_COLS, TEMPERATURE_COLS
from train_lirf_regime_v23 import (build_features_lirf, RATE_KEYS,
                                    RATE_FEATURES, K_SMOOTH, FB_TOL,
                                    HOLDOUT_MONTHS, EARLYSTOP_MONTHS,
                                    TRAIN_MONTHS, MODELS, rmse)

FIT_MONTHS = TRAIN_MONTHS | EARLYSTOP_MONTHS  # {2..6, 8..12}


def add_fallback_rate_features_fit(dep_fit, base_rate):
    """OOF leave-one-month-out on fit months only. Same shape as v23.

    dep_fit must exclude HOLDOUT_MONTHS. Returns scoring maps built from
    fit months for use on the ranking frame.
    """
    key_defs = {
        "op": dep_fit["AIRCRAFT_OPERATOR_flt"].astype(str),
        "flt_prefix": dep_fit["FLIGHT_mvt"].astype(str).str[:3].fillna("UNK"),
        "stand": dep_fit["STAND_mvt"].astype(str),
        "aircraft_type": dep_fit["AIRCRAFT_TYPE_mvt"].astype(str),
        "op_stand": dep_fit["AIRCRAFT_OPERATOR_flt"].astype(str) + "|" +
                    dep_fit["STAND_mvt"].astype(str),
        "stand_first_char": dep_fit["STAND_mvt"].astype(str).str.slice(0, 1).fillna("UNK"),
        "destination": dep_fit["ADES_mvt"].astype(str),
    }
    fit_months = sorted(dep_fit["month"].unique())
    scoring_maps = {}
    for k, keys in key_defs.items():
        rate = np.full(len(dep_fit), base_rate, dtype=np.float32)
        cnt = np.zeros(len(dep_fit), dtype=np.float32)
        for m in fit_months:
            tr = dep_fit["month"] != m
            ev = dep_fit["month"] == m
            if ev.sum() == 0: continue
            grp = pd.DataFrame({"key": keys[tr].values,
                                "fb": dep_fit.loc[tr, "is_fb"].astype(float).values}) \
                .groupby("key").agg(s=("fb", "sum"), n=("fb", "size"))
            grp["smooth"] = (grp["s"] + K_SMOOTH * base_rate) / (grp["n"] + K_SMOOTH)
            rmap = grp["smooth"].to_dict()
            cmap = grp["n"].to_dict()
            ev_keys = keys[ev].values
            rate[ev.values] = np.array([rmap.get(kk, base_rate) for kk in ev_keys])
            cnt[ev.values] = np.array([cmap.get(kk, 0) for kk in ev_keys])
        dep_fit[f"fbrate_{k}"] = rate
        dep_fit[f"fbcount_{k}"] = cnt
        grp_all = pd.DataFrame({"key": keys.values,
                                "fb": dep_fit["is_fb"].astype(float).values}) \
            .groupby("key").agg(s=("fb", "sum"), n=("fb", "size"))
        grp_all["smooth"] = (grp_all["s"] + K_SMOOTH * base_rate) / (grp_all["n"] + K_SMOOTH)
        scoring_maps[k] = {"rate": grp_all["smooth"].to_dict(),
                           "count": grp_all["n"].to_dict()}
    return scoring_maps


def main():
    t_all = time.time()
    dep, encoders = build_features_lirf()

    # Drop hold-out before OOF (audit Item 2). Encoders were already fit on
    # ~hold in build_features_lirf, so drop rows now for a clean pool.
    dep_fit = dep[dep["month"].isin(FIT_MONTHS)].copy()
    print(f"Rows after dropping {sorted(HOLDOUT_MONTHS)}: {len(dep_fit):,}"
          f"  (was {len(dep):,})")

    base_rate = float(dep_fit["is_fb"].mean())
    print(f"LIRF base rate on fit months: {base_rate:.4f}")
    scoring_maps = add_fallback_rate_features_fit(dep_fit, base_rate)

    for c in CAT_COLS:
        dep_fit[c] = dep_fit[c].astype("category")

    train = dep_fit[dep_fit["month"].isin(TRAIN_MONTHS)].copy()
    stop = dep_fit[dep_fit["month"].isin(EARLYSTOP_MONTHS)].copy()
    print(f"LIRF: train {len(train):,}  stop {len(stop):,}")
    for c in CAT_COLS:
        stop[c] = pd.Categorical(stop[c], categories=train[c].cat.categories)

    ec_cols = eurocontrol_numeric_cols()
    op_cols = operator_numeric_cols()
    feat = (CAT_COLS + BASE_NUM_COLS + SIGNED_LOG_COLS + TEMPERATURE_COLS +
            CONG_V2_NUM_COLS + ec_cols + op_cols +
            TAXI_NUM_COLS + ADV_NUM_COLS + OSM_PATH_NUM_COLS +
            OPDI_NUM_COLS + OPDI_LIVE_NUM_COLS + TURN_NUM_COLS + DISR_NUM_COLS +
            RATE_FEATURES)
    feat = [c for c in feat if c in train.columns]
    print(f"Feature count: {len(feat)}")

    print("\n=== p_fb_v34 classifier ===")
    dt_c = lgb.Dataset(train[feat], label=train["is_fb"].values,
                       categorical_feature=CAT_COLS, free_raw_data=True)
    dv_c = lgb.Dataset(stop[feat], label=stop["is_fb"].values,
                       categorical_feature=CAT_COLS, reference=dt_c, free_raw_data=True)
    params_cls = {"objective": "binary", "metric": "binary_logloss",
                  "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
                  "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 5,
                  "verbosity": -1, "num_threads": -1}
    t0 = time.time()
    b_fb = lgb.train(params_cls, dt_c, num_boost_round=2000,
                     valid_sets=[dv_c], valid_names=["stop"],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"  p_fb_v34 best iter {b_fb.best_iteration}  time {time.time()-t0:.1f}s")

    p_st_raw = b_fb.predict(stop[feat], num_iteration=b_fb.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_st_raw, stop["is_fb"].values)
    print(f"  Stop AUC {roc_auc_score(stop['is_fb'], p_st_raw):.4f}  "
          f"logloss {log_loss(stop['is_fb'], np.clip(p_st_raw,1e-6,1-1e-6)):.4f}")

    b_fb.save_model(os.path.join(MODELS, "lgbm_p_fb_lirf_v34.txt"))
    with open(os.path.join(MODELS, "lirf_regime_v34.features.txt"), "w") as f:
        f.write("\n".join(feat))
    with open(os.path.join(MODELS, "lirf_regime_v34.isotonic.pkl"), "wb") as f:
        pickle.dump(iso, f)
    with open(os.path.join(MODELS, "lirf_regime_v34.rate_maps.pkl"), "wb") as f:
        pickle.dump({"maps": scoring_maps, "base_rate": base_rate,
                     "keys": RATE_KEYS, "fit_months": sorted(FIT_MONTHS)}, f)
    print(f"\nSaved -> {MODELS}/lirf_regime_v34.*  wall {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
