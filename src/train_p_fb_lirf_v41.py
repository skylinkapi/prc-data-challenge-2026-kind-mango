"""Paired test of the tempo columns on the LIRF fallback gate p_fb.

Both arms read out-of-fold fallback-rate columns on the hold-out rows, so the
deployed gate loses the all-month leak of its scoring maps. Control: the v23
booster and calibrator. Candidate: the v23 recipe plus the 13 tempo columns,
calibrated on the stop months. Each arm scores the whole LIRF head with the
v41 R_norm members. Also scores the head without the 4,431 s clip.
"""
import json
import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import eval_v33_holdout as E
from predict_v23 import apply_stepA_v22
from predict_v30 import ITY_SD_THRESHOLD, NORMAL_MEAN_LIRF, P24_ITY, R_NORM_CLIP
from train_lgbm_v21 import CAT_COLS
from train_lirf_regime import EARLYSTOP_MONTHS, FB_TOL, HOLDOUT_MONTHS, MODELS, TRAIN_MONTHS, build_features
from train_lirf_regime_v23 import RATE_FEATURES, add_fallback_rate_features
from train_r_all_v40 import TEMPO_COLS, V2_FRAME, class_table, classify
from train_r_norm_lirf_seeds import SEEDS

N_HOLDOUT = 344336
GATE_PARAMS = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 63,
               "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 5,
               "verbosity": -1, "num_threads": -1, "seed": 42, "bagging_seed": 42, "feature_fraction_seed": 42}
log = logging.getLogger(__name__)


def train_gate(dep: pd.DataFrame, feat: list[str]):
    """v23 recipe on the train months, isotonic map on the stop months."""
    tr = dep[dep["month"].isin(TRAIN_MONTHS)]
    st = dep[dep["month"].isin(EARLYSTOP_MONTHS)]
    dt = lgb.Dataset(tr[feat], label=tr["is_fb"].values, categorical_feature=CAT_COLS)
    dv = lgb.Dataset(st[feat], label=st["is_fb"].values, categorical_feature=CAT_COLS, reference=dt)
    b = lgb.train(GATE_PARAMS, dt, num_boost_round=2000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    raw = b.predict(st[feat], num_iteration=b.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, st["is_fb"].values)
    log.info("gate best iter %d stop AUC %.4f", b.best_iteration, roc_auc_score(st["is_fb"], raw))
    return b, iso


def head(hold: pd.DataFrame, p_fb: np.ndarray, r_norm: np.ndarray) -> np.ndarray:
    sd = hold["sched_delay"].values.astype(float)
    valid = ~np.isnan(sd)
    pred = np.where(valid, p_fb * sd + (1 - p_fb) * r_norm, r_norm)
    with open(os.path.join(MODELS, "lirf_band_table_v30.json")) as f:
        band = json.load(f)
    pred, cell = apply_stepA_v22(pred, sd, hold["ADEP_mvt"].astype(str).values, hold["FLIGHT_ID_mvt"].values, band)
    ity = (sd > ITY_SD_THRESHOLD) & valid & ~cell
    pred[ity] = P24_ITY * (86400 + NORMAL_MEAN_LIRF) + (1 - P24_ITY) * sd[ity]
    return np.clip(pred, 0, None)


def score(hold: pd.DataFrame, pred: np.ndarray) -> dict:
    y = hold["TAXITIME_SEC_mvt"].astype(float).values
    rep = class_table(pred, y, classify(y, hold["sched_delay"].values.astype(float)), np.full(len(y), "LIRF"))
    return {"full_rmse": rep["full_rmse"], "clean_rmse": rep["clean_rmse"],
            "class_mse": {k: v * len(y) / N_HOLDOUT for k, v in rep["class_mse"].items()}}


def main() -> None:
    dep, _ = build_features()
    dep = dep.merge(pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS]), on="MVT_ID_mvt", how="left")
    dep["is_fb"] = ((dep["TAXITIME_SEC_mvt"].astype(float) - dep["sched_delay"]).abs() < FB_TOL).astype(np.int8)
    add_fallback_rate_features(dep, float(dep.loc[~dep["month"].isin(HOLDOUT_MONTHS), "is_fb"].mean()))
    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")
    feat_old = E.read_list("lirf_regime_v23.features.txt")
    feat_new = feat_old + TEMPO_COLS
    b_new, iso_new = train_gate(dep, feat_new)
    b_new.save_model(os.path.join(MODELS, "lgbm_p_fb_lirf_v41.txt"))
    with open(os.path.join(MODELS, "lirf_regime_v41.gate_isotonic.pkl"), "wb") as f:
        pickle.dump(iso_new, f)
    with open(os.path.join(MODELS, "lirf_regime_v41.gate_features.txt"), "w") as f:
        f.write("\n".join(feat_new))

    hold = dep[dep["month"].isin(HOLDOUT_MONTHS)].copy()
    feat_norm = E.read_list("lirf_regime_v41.features.txt")
    members = [np.clip(lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_norm_lirf_v41_s{s}.txt"))
                       .predict(hold[feat_norm]), 0, None) for s in SEEDS]
    r_raw = np.mean(members, axis=0)
    r_norm = np.minimum(r_raw, R_NORM_CLIP)
    with open(os.path.join(MODELS, "lirf_regime_v23.isotonic.pkl"), "rb") as f:
        iso_old = pickle.load(f)
    p_old = iso_old.transform(lgb.Booster(model_file=os.path.join(MODELS, "lgbm_p_fb_lirf_v23.txt")).predict(hold[feat_old]))
    p_new = iso_new.transform(b_new.predict(hold[feat_new], num_iteration=b_new.best_iteration))
    is_fb = hold["is_fb"].values
    report = {"auc_holdout": {"control": roc_auc_score(is_fb, p_old), "v41_gate": roc_auc_score(is_fb, p_new)},
              "control_gate_oof_rates": score(hold, head(hold, p_old, r_norm)),
              "v41_gate": score(hold, head(hold, p_new, r_norm)),
              "control_gate_no_clip": score(hold, head(hold, p_old, r_raw))}
    for k, v in report.items():
        log.info("%s %s", k, {kk: (round(vv, 4) if isinstance(vv, float) else {c: round(x) for c, x in vv.items()})
                              for kk, vv in v.items()} if k != "auc_holdout" else {kk: round(vv, 4) for kk, vv in v.items()})
    with open(os.path.join(MODELS, "lirf_regime_v41.gate_holdout.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
