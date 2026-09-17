"""MD2 + MH1: rebuild the LIRF head without the drifted columns.

The 38 `ec_*`, 9 `opdi_*` and 9 `opdi_live_*` columns (56 total) leave both
the R_norm_LIRF regressor and the p_fb LIRF gate. Everything else stays on
the deployed v41 / v23 recipe:

- R_norm_LIRF: linear_tree, num_leaves 127, linear_lambda 1.0, BEST_PARAMS,
  five seeds 42..46, trained on genuine rows (|y-sd| >= 60, y < 80,000),
  early-stopped on months 11 and 12.
- p_fb LIRF: binary log-loss, 63 leaves, lr 0.05, min_data 50,
  feature_fraction 0.8, bagging 0.9. Isotonic fit on the same stop months.

Reads the LIRF frame cache and merges TEMPO_COLS from the src_v2 training
frame by MVT_ID_mvt (same as train_r_norm_lirf_v41.py). Saves:

- `lgbm_r_norm_lirf_v49_s{42..46}.txt`
- `lgbm_p_fb_lirf_v49.txt`
- `lirf_regime_v49.features.txt`         (R_norm feature list)
- `lirf_regime_v49.gate_features.txt`    (p_fb feature list)
- `lirf_regime_v49.isotonic.pkl`
- `lirf_regime_v49.holdout.json`         (paired vs v41 head on months 1 and 7)
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src_v3 import config as C
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

MODELS_OLD = C.ROOT / "models"
LIRF_CACHE = MODELS_OLD / "lirf_frame_cache.parquet"
V2_FRAME = MODELS_OLD / "v2" / "cache" / "frame_train.parquet"

FB_TOL = 60                          # LIRF head uses 60-second fallback
TRAIN_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
STOP_MONTHS = (11, 12)
HOLDOUT_MONTHS = (1, 7)
SEEDS = (42, 43, 44, 45, 46)

TEMPO_COLS = ["nb_eobt_med_apt30", "nb_eobt_mean_apt30", "nb_eobt_cnt_apt30",
              "nb_eobt_med_apt60", "nb_eobt_mean_apt60", "nb_eobt_cnt_apt60",
              "nb_eobt_med_rwy30", "nb_eobt_mean_rwy30", "nb_eobt_cnt_rwy30",
              "order_later_eobt_30", "order_total_30", "stand_gap",
              "queue_eobt_mvt"]

log = logging.getLogger(__name__)


def _drifted_columns(feat: list[str]) -> list[str]:
    return [c for c in feat if c.startswith("ec_") or c.startswith("opdi_")]


def _filter(feat: list[str]) -> list[str]:
    drifted = set(_drifted_columns(feat))
    return [c for c in feat if c not in drifted]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
    from train_lirf_regime_v23 import add_fallback_rate_features, FB_TOL as FB_TOL_V23

    log.info("loading LIRF frame cache")
    dep = pd.read_parquet(LIRF_CACHE)
    log.info("merging tempo columns from v2 frame")
    tempo = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    log.info("LIRF frame %d rows, tempo coverage %.3f",
             len(dep), dep["nb_eobt_med_apt30"].notna().mean())

    # v23 gate reads the fbrate/fbcount rate features; compute them OOF here
    # so the p_fb training data has them, matching the deployed recipe.
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    sd = dep["sched_delay"].astype(float)
    dep["is_fb"] = ((y - sd).abs() < FB_TOL_V23).astype(np.int8)
    base_rate = float(dep["is_fb"].mean())
    log.info("LIRF fallback base rate: %.4f", base_rate)
    add_fallback_rate_features(dep, base_rate)

    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")

    train = dep[dep["month"].isin(TRAIN_MONTHS)].copy()
    stop = dep[dep["month"].isin(STOP_MONTHS)].copy()
    test = dep[dep["month"].isin(HOLDOUT_MONTHS)].copy()
    log.info("train %d stop %d test %d", len(train), len(stop), len(test))
    for c in CAT_COLS:
        stop[c] = pd.Categorical(stop[c], categories=train[c].cat.categories)
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

    with open(MODELS_OLD / "lirf_regime_v41.features.txt") as f:
        feat_v41 = [x for x in f.read().splitlines() if x]
    with open(MODELS_OLD / "lirf_regime_v23.features.txt") as f:
        feat_v23 = [x for x in f.read().splitlines() if x]

    feat_r_norm = _filter(feat_v41)
    feat_p_fb = _filter(feat_v23)
    log.info("v41 R_norm feat %d -> v49 %d (dropped %d)",
             len(feat_v41), len(feat_r_norm), len(feat_v41) - len(feat_r_norm))
    log.info("v23 p_fb feat %d -> v49 %d (dropped %d)",
             len(feat_v23), len(feat_p_fb), len(feat_v23) - len(feat_p_fb))

    # --- R_norm_LIRF, 5 seeds, genuine rows only ---
    log.info("== training R_norm_LIRF v49 ==")
    y_tr = train["TAXITIME_SEC_mvt"].astype(float).values
    sd_tr = train["sched_delay"].astype(float).values
    genuine_tr = (np.abs(y_tr - sd_tr) >= FB_TOL) & (y_tr < 80000)
    y_st = stop["TAXITIME_SEC_mvt"].astype(float).values
    sd_st = stop["sched_delay"].astype(float).values
    genuine_st = (np.abs(y_st - sd_st) >= FB_TOL) & (y_st < 80000)
    tr_g = train[genuine_tr]
    st_g = stop[genuine_st]
    log.info("genuine train %d stop %d", len(tr_g), len(st_g))

    dt = lgb.Dataset(tr_g[feat_r_norm],
                     label=tr_g["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})
    dv = lgb.Dataset(st_g[feat_r_norm],
                     label=st_g["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS, reference=dt,
                     params={"linear_tree": True, "feature_pre_filter": False})
    iters: list[int] = []
    for s in SEEDS:
        params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0, "num_leaves": 127,
                  "bagging_freq": 5, "verbosity": -1,
                  "num_threads": C.NUM_THREADS,
                  "deterministic": C.DETERMINISTIC,
                  "seed": s, "bagging_seed": s, "feature_fraction_seed": s}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=3000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        b.save_model(str(MODELS_OLD / f"lgbm_r_norm_lirf_v49_s{s}.txt"))
        iters.append(b.best_iteration)
        log.info("v49 R_norm seed %d best iter %d in %.0fs",
                 s, b.best_iteration, time.time() - t0)

    with open(MODELS_OLD / "lirf_regime_v49.features.txt", "w") as f:
        f.write("\n".join(feat_r_norm))

    # --- p_fb classifier, all LIRF rows, single booster ---
    log.info("== training p_fb v49 ==")
    is_fb_tr = (np.abs(y_tr - sd_tr) < FB_TOL).astype(np.int8)
    is_fb_st = (np.abs(y_st - sd_st) < FB_TOL).astype(np.int8)
    # v23 uses rate features (`fbrate_*`) — keep them if present
    dt_c = lgb.Dataset(train[feat_p_fb], label=is_fb_tr,
                       categorical_feature=CAT_COLS,
                       params={"feature_pre_filter": False})
    dv_c = lgb.Dataset(stop[feat_p_fb], label=is_fb_st,
                       categorical_feature=CAT_COLS, reference=dt_c,
                       params={"feature_pre_filter": False})
    params_cls = {"objective": "binary", "metric": "binary_logloss",
                  "learning_rate": 0.05, "num_leaves": 63,
                  "min_data_in_leaf": 50, "feature_fraction": 0.8,
                  "bagging_fraction": 0.9, "bagging_freq": 5,
                  "verbosity": -1,
                  "num_threads": C.NUM_THREADS,
                  "deterministic": C.DETERMINISTIC,
                  "seed": C.GLOBAL_SEED, "bagging_seed": C.GLOBAL_SEED,
                  "feature_fraction_seed": C.GLOBAL_SEED}
    t0 = time.time()
    b_fb = lgb.train(params_cls, dt_c, num_boost_round=2000, valid_sets=[dv_c],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
    log.info("v49 p_fb best iter %d in %.0fs",
             b_fb.best_iteration, time.time() - t0)
    p_st_raw = b_fb.predict(stop[feat_p_fb], num_iteration=b_fb.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_st_raw, is_fb_st)
    stop_auc = roc_auc_score(is_fb_st, p_st_raw)
    stop_ll = log_loss(is_fb_st, np.clip(p_st_raw, 1e-6, 1 - 1e-6))
    log.info("stop AUC %.4f  log-loss %.4f", stop_auc, stop_ll)

    b_fb.save_model(str(MODELS_OLD / "lgbm_p_fb_lirf_v49.txt"))
    with open(MODELS_OLD / "lirf_regime_v49.gate_features.txt", "w") as f:
        f.write("\n".join(feat_p_fb))
    with open(MODELS_OLD / "lirf_regime_v49.isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)

    # --- Hold-out (months 1 and 7) paired scoring ---
    log.info("== hold-out paired ==")
    y_te = test["TAXITIME_SEC_mvt"].astype(float).values
    sd_te = test["sched_delay"].astype(float).values
    valid_sd = ~np.isnan(sd_te)
    clean = (y_te >= 30) & (y_te <= 7200)

    def _mixture(members_dir: list[str], feat_n: list[str], p_fb_file: str,
                 iso_file: str, feat_pfb: list[str]) -> np.ndarray:
        preds = [np.clip(lgb.Booster(model_file=str(MODELS_OLD / fn)).predict(test[feat_n]),
                         0, None) for fn in members_dir]
        r_norm = np.mean(preds, axis=0)
        b_fb2 = lgb.Booster(model_file=str(MODELS_OLD / p_fb_file))
        with open(MODELS_OLD / iso_file, "rb") as fh:
            iso2 = pickle.load(fh)
        p_fb_cal = iso2.transform(b_fb2.predict(test[feat_pfb]))
        return np.where(valid_sd, p_fb_cal * sd_te + (1 - p_fb_cal) * r_norm, r_norm)

    # v41 head control
    v41_files = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in SEEDS]
    pred_v41 = _mixture(v41_files, feat_v41, "lgbm_p_fb_lirf_v23.txt",
                        "lirf_regime_v23.isotonic.pkl", feat_v23)
    # v49 head
    v49_files = [f"lgbm_r_norm_lirf_v49_s{s}.txt" for s in SEEDS]
    pred_v49 = _mixture(v49_files, feat_r_norm, "lgbm_p_fb_lirf_v49.txt",
                        "lirf_regime_v49.isotonic.pkl", feat_p_fb)

    report = {
        "n_test": int(len(test)),
        "dropped_from_r_norm": len(feat_v41) - len(feat_r_norm),
        "dropped_from_p_fb": len(feat_v23) - len(feat_p_fb),
        "r_norm_v41_full_rmse": rmse(y_te, pred_v41),
        "r_norm_v49_full_rmse": rmse(y_te, pred_v49),
        "r_norm_v41_clean_rmse": rmse(y_te[clean], pred_v41[clean]),
        "r_norm_v49_clean_rmse": rmse(y_te[clean], pred_v49[clean]),
        "delta_full_mse": float((pred_v49 - y_te).var() - (pred_v41 - y_te).var()),
        "iters_r_norm": iters,
        "p_fb_stop_auc": float(stop_auc),
        "p_fb_stop_logloss": float(stop_ll),
    }
    (MODELS_OLD / "lirf_regime_v49.holdout.json").write_text(json.dumps(report, indent=1))
    log.info("v41 LIRF head FULL %.2f CLEAN %.2f",
             report["r_norm_v41_full_rmse"], report["r_norm_v41_clean_rmse"])
    log.info("v49 LIRF head FULL %.2f CLEAN %.2f",
             report["r_norm_v49_full_rmse"], report["r_norm_v49_clean_rmse"])


if __name__ == "__main__":
    main()
