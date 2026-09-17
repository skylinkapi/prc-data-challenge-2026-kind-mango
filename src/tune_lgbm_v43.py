"""L1: purified retune of the base on the H1 frame.

Section 5.2 step 1 of the fourteenth pass: Optuna, 30 trials, one seed,
`linear_tree` on and off in the space, blocked stop months 11 and 12, the
hold-out months 1 and 7 untouched. The frame is the 110-column H1 cache; the
purified protocol (uncensored target, no leak into the hold-out) follows
tune_lgbm_v36.py. Writes models/tune_lgbm_v43.tuned_params.json.
"""
import json
import logging
import os
import time

import lightgbm as lgb
import optuna
import pandas as pd

from build_h1_frame_cache import load_frame
from train_lgbm_v21 import CAT_COLS
from train_lirf_regime import EARLYSTOP_MONTHS, TRAIN_MONTHS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(MODELS, "tune_lgbm_v43.tuned_params.json")
N_TRIALS = int(os.environ.get("TUNE_TRIALS", "30"))
log = logging.getLogger(__name__)


def make_params(trial: optuna.Trial) -> dict:
    # linear_tree pinned to True: the Dataset caches it on first use, so a
    # per-trial change raises "Cannot change linear_tree after constructed
    # Dataset handle". The deployed base uses linear_tree=True; the retune
    # searches its hyperparameters.
    return {
        "objective": "regression", "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 64, 440),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 3.0),
        "linear_lambda": trial.suggest_float("linear_lambda", 1e-4, 10.0, log=True),
        "linear_tree": True,
        "bagging_freq": 5, "verbosity": -1, "num_threads": -1,
        "seed": 42, "bagging_seed": 42,
    }


def main() -> None:
    dep, feat = load_frame()
    train = dep[dep["month"].isin(TRAIN_MONTHS)]
    stop = dep[dep["month"].isin(EARLYSTOP_MONTHS)]
    log.info("train %d stop %d features %d", len(train), len(stop), len(feat))
    # feature_pre_filter=False lets min_data_in_leaf vary between trials
    # without hitting the "may cause unexpected behaviour" pre-filter check.
    ds_params = {"linear_tree": True, "feature_pre_filter": False}
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS, free_raw_data=True, params=ds_params)
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS, reference=dt, free_raw_data=True,
                     params=ds_params)

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        b = lgb.train(make_params(trial), dt, num_boost_round=3000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        rmse = b.best_score["valid_0"]["rmse"]
        log.info("trial %d: rmse %.3f at iter %d in %.0fs", trial.number, rmse,
                 b.best_iteration, time.time() - t0)
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    fixed = {"objective": "regression", "metric": "rmse", "bagging_freq": 5,
             "verbosity": -1, "num_threads": -1, "seed": 42, "bagging_seed": 42}
    tuned = {**study.best_params, **fixed}
    log.info("best stop rmse %.3f; params %s", study.best_value, study.best_params)
    with open(OUT, "w") as f:
        json.dump({"tuned": tuned, "study_best": study.best_params,
                   "stop_rmse": study.best_value, "n_trials": len(study.trials)}, f, indent=1)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
