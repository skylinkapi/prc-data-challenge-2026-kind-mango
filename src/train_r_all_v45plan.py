"""L9: paired test of plan_taxi_res (clipped +-3600 s) on the v44 stack.

Single new column keyed on MVT_ID_mvt. Control = v44 (116 cols); experimental
= v44 + plan_taxi_res (117 cols). MODEL_ANALYSIS 4.1 L9 expected 0-500 MSE,
grade B from the twelfth-pass quick test; the raw plan_block and arvt1_mvt
columns stay out (they collapsed a seed in v38).
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

PLAN = os.path.join(MODELS, "plan_taxi_res_train.parquet")
EXTRA_COLS = ["plan_taxi_res"]
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
    dep = dep.merge(pd.read_parquet(PLAN), on="MVT_ID_mvt", how="left")
    cov = dep["plan_taxi_res"].notna().mean()
    log.info("frame %d rows, plan_taxi_res coverage %.3f", len(dep), cov)
    return dep, feat


def train_members(train, stop, test, feat, tag):
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS)
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS, reference=dt)
    preds, iters = [], []
    for seed in DEFAULT_SEEDS:
        params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0, "num_leaves": 220,
                  "seed": seed, "bagging_seed": seed, "bagging_freq": 5,
                  "verbosity": -1, "num_threads": -1}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        log.info("%s seed %d best iter %d in %.0fs", tag, seed,
                 b.best_iteration, time.time() - t0)
        preds.append(np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None))
        iters.append(b.best_iteration)
    return np.mean(preds, axis=0), iters


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

    report = {"extra_cols": EXTRA_COLS}
    control_cols = feat + TEMPO_COLS + EOBT_P2575
    v45_cols = control_cols + EXTRA_COLS
    for tag, cols in (("control_v44", control_cols), ("v45plan", v45_cols)):
        pred, iters = train_members(train, stop, test, cols, tag)
        report[tag] = {"best_iters": iters, **class_table(pred, y, cls, apt)}
        log.info("%s FULL %.2f CLEAN %.2f iters %s", tag,
                 report[tag]["full_rmse"], report[tag]["clean_rmse"], iters)
    delta_class = {k: report["v45plan"]["class_mse"][k] - report["control_v44"]["class_mse"][k]
                   for k in report["control_v44"]["class_mse"]}
    served = {}
    for k in delta_class:
        ctl = report["control_v44"]["class_mse"][k] - report["control_v44"]["per_airport"]["LIRF"][k]
        new = report["v45plan"]["class_mse"][k] - report["v45plan"]["per_airport"]["LIRF"][k]
        served[k] = new - ctl
    report["delta_v45plan_minus_v44"] = {
        "full_rmse": report["v45plan"]["full_rmse"] - report["control_v44"]["full_rmse"],
        "clean_rmse": report["v45plan"]["clean_rmse"] - report["control_v44"]["clean_rmse"],
        "class_mse": delta_class,
    }
    report["delta_served_v45plan_minus_v44"] = served
    log.info("delta class MSE: %s", {k: round(v) for k, v in delta_class.items()})
    log.info("delta served (non-LIRF): %s", {k: round(v) for k, v in served.items()})
    log.info("served clean price: %+.0f MSE (gate >= 300)", -served["clean"])
    with open(os.path.join(MODELS, "lgbm_r_all_v45plan.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
