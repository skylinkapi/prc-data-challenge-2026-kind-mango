"""MB5 on the clock-era base: an Optuna study for the LightGBM parameters (v76).

The v43 parameters date from a study before the NM clock columns. This
study reads the v73 columns and the MB3 rows, trains on months 2 to 6 and
8 to 10, and early-stops on months 11 and 12. Trial 0 is the v43
parameters. The bounds follow MB5: up to 1,024 leaves, `min_data_in_leaf`
up to 2,000, `linear_lambda` up to 100, and the categorical parameters.
The best trial and the v43 parameters are then refit and priced on the
hold-out months 1 and 7 outside LIRF. With `--anchor-col`, the study runs
on `y - clip(anchor, 0, 7,200)` (v79a) and the report also prices the
best trial against the saved anchored member of `measure_member_v77.py`
(v43 parameters, 2,000 rounds).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import OUT_TRAIN as CLOCKS_TRAIN
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS, holdout_rows
from src_v3.measure_member_v77 import out_path as member_path
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.train_v57_base import TUNED

OUT = C.ROOT / "models" / "tune_lgbm_v76.tuned_params.json"
FIXED = {"objective": "regression", "metric": "rmse", "linear_tree": True, "bagging_freq": 5,
         "verbosity": -1, "num_threads": C.TUNE_V76_THREADS, "deterministic": C.DETERMINISTIC,
         "seed": C.GLOBAL_SEED, "bagging_seed": C.GLOBAL_SEED, "feature_fraction_seed": C.GLOBAL_SEED}
log = logging.getLogger(__name__)


def space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 64, 1024, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 3.0),
        "linear_lambda": trial.suggest_float("linear_lambda", 1e-4, 100.0, log=True),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        "min_data_per_group": trial.suggest_int("min_data_per_group", 10, 1000, log=True),
        "max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 128, log=True),
    }


SPACE_KEYS = ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction",
              "lambda_l1", "lambda_l2", "min_gain_to_split", "linear_lambda", "cat_smooth",
              "min_data_per_group", "max_cat_threshold")


def fit(dt: lgb.Dataset, dv: lgb.Dataset, params: dict) -> lgb.Booster:
    return lgb.train({**params, **FIXED}, dt, num_boost_round=C.TUNE_V76_MAX_ROUNDS, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(C.TUNE_V76_EARLY_STOP, verbose=False)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-col", help="Tune the anchored member on this clock column (v79a: mvt_aobt3).")
    ap.add_argument("--out", default=str(OUT), help="Study file with the parameters and the report.")
    ap.add_argument("--sampler-seed", type=int, default=C.GLOBAL_SEED)
    ap.add_argument("--control", help="Study file whose best parameters are trial 0 (v84: the v76 study).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame((CLOCKS_TRAIN,))
    if args.anchor_col:
        dep = dep[dep[args.anchor_col].notna()].copy()
        dep["TAXITIME_SEC_mvt"] = dep["TAXITIME_SEC_mvt"] - dep[args.anchor_col].clip(0, C.NM_CLOCK_CLIP_S)
    ds = {"categorical_feature": CAT_COLS, "params": {"linear_tree": True, "feature_pre_filter": False}}
    train, stop = dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)]
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values, **ds)
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values, reference=dt, **ds)
    del dep, train, stop
    v43 = {k: v for k, v in json.loads(TUNED.read_text())["study_best"].items()}
    control = ({k: v for k, v in json.loads(Path(args.control).read_text())["tuned"].items() if k in SPACE_KEYS}
               if args.control else {**v43, "cat_smooth": 10.0, "min_data_per_group": 100, "max_cat_threshold": 32})

    def objective(trial: optuna.Trial) -> float:
        b = fit(dt, dv, space(trial))
        trial.set_user_attr("best_iteration", b.best_iteration)
        log.info("trial %d: stop rmse %.3f at %d rounds", trial.number, b.best_score["valid_0"]["rmse"],
                 b.best_iteration)
        return b.best_score["valid_0"]["rmse"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.sampler_seed))
    study.enqueue_trial(control)
    study.optimize(objective, n_trials=C.TUNE_V76_TRIALS)
    arms = {"v43": study.trials[0].params, "best": study.best_params}
    h, _ = holdout_rows()
    h = h.merge(pd.read_parquet(CLOCKS_TRAIN), on="MVT_ID_mvt", how="left", validate="m:1")
    y, apt = h["TAXITIME_SEC_mvt"].values.astype(float), h["ADEP_mvt"].astype(str).values
    boosters = {arm: fit(dt, dv, p) for arm, p in arms.items()}
    raw = {arm: b.predict(h[feat], num_iteration=b.best_iteration) for arm, b in boosters.items()}
    ref = pd.read_parquet(PRED_OUT)
    if args.anchor_col:
        anchor = h[args.anchor_col].clip(0, C.NM_CLOCK_CLIP_S).values
        pred = {arm: np.where(np.isnan(anchor), ref["clocks"].values, np.clip(anchor + r, 0, None))
                for arm, r in raw.items()}
    else:
        pred = {arm: np.clip(r, 0, None) for arm, r in raw.items()}
    report = {
        "stop_rmse": {"v43": study.trials[0].value, "best": study.best_value},
        "best_iteration": {arm: b.best_iteration for arm, b in boosters.items()},
        "best_minus_v43": mse_delta(pred["best"], pred["v43"], y),
        "per_airport": {a: mse_delta(pred["best"][apt == a], pred["v43"][apt == a], y[apt == a])
                        for a in sorted(set(apt))},
        "rows_over_7200": {arm: int((p > 7200).sum()) for arm, p in pred.items()},
        "n_trials": len(study.trials)}
    if args.anchor_col:
        saved = pd.read_parquet(member_path("lgb_anchor"))["pred"].values
        report["best_minus_saved_anchor"] = mse_delta(pred["best"], saved, y)
    report["refit"] = report["best_minus_v43"] <= -C.TUNE_V76_MIN_GAIN_MSE
    log.info("%s", json.dumps(report, indent=1))
    Path(args.out).write_text(json.dumps({"tuned": {**study.best_params, **FIXED}, "report": report,
                               "best_iteration": report["best_iteration"]["best"],
                               "n_fit": int(dt.num_data())}, indent=1))


if __name__ == "__main__":
    main()
