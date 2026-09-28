"""MB5 on the LIRF `R_norm` regressor (v78): an Optuna study for its LightGBM parameters.

`R_norm` still trains with the v41 parameters (127 leaves, `linear_lambda`
1.0). The study reads the 166 v41 columns and the LIRF genuine rows, trains
on months 2 to 6 and 8 to 10, and early-stops on months 11 and 12. Trial 0
is the v41 parameters. The best trial and trial 0 are then refit and priced
on the genuine rows of months 1 and 7, on the 344,336-row scale.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import BEST_PARAMS, CAT_COLS

from src_v3 import config as C
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS
from src_v3.measure_tail_rules import HOLD, mse_delta
from src_v3.train_r_norm_v75 import load_frame
from src_v3.tune_lgbm_v76 import space

OUT = C.ROOT / "models" / "tune_r_norm_v78.tuned_params.json"
FIXED = {"objective": "regression", "metric": "rmse", "linear_tree": True, "bagging_freq": 5, "verbosity": -1,
         "num_threads": C.NUM_THREADS, "deterministic": C.DETERMINISTIC, "seed": C.GLOBAL_SEED,
         "bagging_seed": C.GLOBAL_SEED, "feature_fraction_seed": C.GLOBAL_SEED}
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame()
    ds = {"categorical_feature": CAT_COLS, "params": {"linear_tree": True, "feature_pre_filter": False}}
    train, stop, h = (dep[dep["month"].isin(m)] for m in (FIT_MONTHS, STOP_MONTHS, HOLD))
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values, **ds)
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values, reference=dt, **ds)

    def fit(params: dict) -> lgb.Booster:
        return lgb.train({**params, **FIXED}, dt, num_boost_round=C.TUNE_V76_MAX_ROUNDS, valid_sets=[dv],
                         callbacks=[lgb.early_stopping(C.TUNE_V76_EARLY_STOP, verbose=False)])

    def objective(trial: optuna.Trial) -> float:
        b = fit(space(trial))
        log.info("trial %d: stop rmse %.3f at %d rounds", trial.number, b.best_score["valid_0"]["rmse"],
                 b.best_iteration)
        return b.best_score["valid_0"]["rmse"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=C.GLOBAL_SEED))
    study.enqueue_trial({**BEST_PARAMS, "num_leaves": 127, "linear_lambda": 1.0,
                         "cat_smooth": 10.0, "min_data_per_group": 100, "max_cat_threshold": 32})
    study.optimize(objective, n_trials=C.TUNE_R_NORM_V78_TRIALS)
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    boosters = {arm: fit(p) for arm, p in (("v41", study.trials[0].params), ("best", study.best_params))}
    pred = {arm: np.clip(b.predict(h[feat], num_iteration=b.best_iteration), 0, None) for arm, b in boosters.items()}
    report = {"stop_rmse": {"v41": study.trials[0].value, "best": study.best_value},
              "best_iteration": {arm: b.best_iteration for arm, b in boosters.items()},
              "genuine_rmse": {arm: float(np.sqrt(np.mean((p - y) ** 2))) for arm, p in pred.items()},
              "best_minus_v41_genuine": mse_delta(pred["best"], pred["v41"], y), "n_holdout": len(h)}
    log.info("%s", json.dumps(report, indent=1))
    OUT.write_text(json.dumps({"tuned": {**study.best_params, **FIXED}, "report": report}, indent=1))


if __name__ == "__main__":
    main()
