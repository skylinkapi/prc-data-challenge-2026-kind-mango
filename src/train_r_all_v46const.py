"""L4: paired test of a constant-leaf booster meaned with the v44 base.

Trains 3 seeds of the v26 recipe on the v44 feature set with linear_tree=False
(constant leaves). Scores the fixed mean 0.7 * v44_linear + 0.3 * v44_const
against v44 alone on the 2025 hold-out. Writes lgbm_r_all_v46const_s{seed} and
a paired report.
"""
import json
import logging
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS, HOLDOUT_MONTHS, MODELS, STOP_FRAC
from train_r_all_v40 import CACHE, ID_MAP, TEMPO_COLS, V2_FRAME, class_table, classify
from train_r_all_v44 import EXTRA_COLS as EOBT_P2575

W_LINEAR = 0.7
W_CONST = 0.3
log = logging.getLogger(__name__)


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().split()
    dep = pd.read_parquet(CACHE)
    dep["MVT_ID_mvt"] = pd.read_parquet(ID_MAP)["MVT_ID_mvt"].values
    tempo = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    dep = dep.merge(pd.read_parquet(os.path.join(MODELS, "tempo_p2575_train.parquet")),
                    on="MVT_ID_mvt", how="left")
    log.info("frame %d rows, v44 features %d", len(dep), len(feat) + len(TEMPO_COLS) + len(EOBT_P2575))
    return dep, feat


def train_v44_predict(train: pd.DataFrame, stop: pd.DataFrame, test: pd.DataFrame,
                      feat: list[str]) -> np.ndarray:
    """Score the shipped v44 members on the same hold-out split."""
    preds = []
    for seed in DEFAULT_SEEDS:
        b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_all_v44_s{seed}.txt"))
        preds.append(np.clip(b.predict(test[feat]), 0, None))
    return np.mean(preds, axis=0)


def train_const_members(train: pd.DataFrame, stop: pd.DataFrame, test: pd.DataFrame,
                        feat: list[str]) -> tuple[np.ndarray, list[int]]:
    """Three constant-leaf seeds of the v26 recipe; save and return the mean."""
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS)
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS, reference=dt)
    preds, iters = [], []
    for seed in DEFAULT_SEEDS:
        params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
                  "linear_tree": False, "num_leaves": 220,
                  "seed": seed, "bagging_seed": seed, "bagging_freq": 5,
                  "verbosity": -1, "num_threads": -1}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        log.info("v46const seed %d best iter %d in %.0fs", seed,
                 b.best_iteration, time.time() - t0)
        b.save_model(os.path.join(MODELS, f"lgbm_r_all_v46const_s{seed}.txt"))
        preds.append(np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None))
        iters.append(b.best_iteration)
    return np.mean(preds, axis=0), iters


def report_stats(pred: np.ndarray, y: np.ndarray, cls: np.ndarray, apt: np.ndarray) -> dict:
    return class_table(pred, y, cls, apt)


def main() -> None:
    dep, feat = load_frame()
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    non_hold = dep.loc[~hold]
    stop_mask = np.random.default_rng(1234).random(len(non_hold)) < STOP_FRAC
    train, stop, test = non_hold[~stop_mask], non_hold[stop_mask], dep.loc[hold]
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd = test["sched_delay"].values.astype(float)
    apt = test["ADEP_mvt"].astype(str).values
    cls = classify(y, sd)
    v44_cols = feat + TEMPO_COLS + EOBT_P2575

    log.info("scoring shipped v44 members on the same hold-out")
    p_v44 = train_v44_predict(train, stop, test, v44_cols)
    log.info("training v46const members (linear_tree=False)")
    p_const, iters = train_const_members(train, stop, test, v44_cols)
    p_mean = W_LINEAR * p_v44 + W_CONST * p_const

    report = {"w_linear": W_LINEAR, "w_const": W_CONST, "const_iters": iters,
              "v44_alone": report_stats(p_v44, y, cls, apt),
              "v46const_alone": report_stats(p_const, y, cls, apt),
              "v46_mean": report_stats(p_mean, y, cls, apt)}
    served = {}
    for k in ("clean", "fallback", "tail", "24h"):
        v44_srv = report["v44_alone"]["class_mse"][k] - report["v44_alone"]["per_airport"]["LIRF"][k]
        new_srv = report["v46_mean"]["class_mse"][k] - report["v46_mean"]["per_airport"]["LIRF"][k]
        served[k] = new_srv - v44_srv
    report["delta_served_v46_mean_minus_v44"] = served
    for k in ("v44_alone", "v46const_alone", "v46_mean"):
        log.info("%s FULL %.2f CLEAN %.2f", k,
                 report[k]["full_rmse"], report[k]["clean_rmse"])
    log.info("delta served (non-LIRF): %s", {k: round(v) for k, v in served.items()})
    log.info("served clean price: %+.0f MSE (gate >= 400)", -served["clean"])
    with open(os.path.join(MODELS, "lgbm_r_all_v46const.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
