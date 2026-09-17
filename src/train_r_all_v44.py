"""L3.a: paired test of 25/75-pct neighbour EOBT columns on the v40 base.

Reads the same 97-column frame that train_r_all_v40.py used, joins the 13
v40 tempo columns and the 6 quartile columns from build_tempo_p2575.py, and
trains the v26 recipe twice on the same split: control (110 v40 columns) and
v44 (116 columns). Writes lgbm_r_all_v44_s{seed} and a paired report keyed on
the class MSE gap.
"""
import json
import logging
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS, HOLDOUT_MONTHS, MODELS, STOP_FRAC
from train_r_all_v40 import CACHE, ID_MAP, TEMPO_COLS, V2_FRAME, class_table, classify

P2575 = os.path.join(MODELS, "tempo_p2575_train.parquet")
EXTRA_COLS = ["nb_eobt_p25_apt30", "nb_eobt_p75_apt30",
              "nb_eobt_p25_apt60", "nb_eobt_p75_apt60",
              "nb_eobt_p25_rwy30", "nb_eobt_p75_rwy30"]
log = logging.getLogger(__name__)


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().split()
    dep = pd.read_parquet(CACHE)
    dep["MVT_ID_mvt"] = pd.read_parquet(ID_MAP)["MVT_ID_mvt"].values
    tempo = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    p2575 = pd.read_parquet(P2575)
    dep = dep.merge(p2575, on="MVT_ID_mvt", how="left")
    cov25 = dep["nb_eobt_p25_apt30"].notna().mean()
    log.info("frame %d rows, p25/p75 coverage %.3f", len(dep), cov25)
    return dep, feat


def train_members(train: pd.DataFrame, stop: pd.DataFrame, test: pd.DataFrame,
                  feat: list[str], tag: str) -> tuple[np.ndarray, list[int]]:
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
        if tag != "control":
            b.save_model(os.path.join(MODELS, f"lgbm_r_all_{tag}_s{seed}.txt"))
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
    control_cols = feat + TEMPO_COLS
    v44_cols = feat + TEMPO_COLS + EXTRA_COLS
    for tag, cols in (("control", control_cols), ("v44", v44_cols)):
        pred, iters = train_members(train, stop, test, cols, tag)
        report[tag] = {"best_iters": iters, **class_table(pred, y, cls, apt)}
        log.info("%s FULL %.2f CLEAN %.2f iters %s", tag,
                 report[tag]["full_rmse"], report[tag]["clean_rmse"], iters)
    delta_class = {k: report["v44"]["class_mse"][k] - report["control"]["class_mse"][k]
                   for k in report["control"]["class_mse"]}
    report["delta_v44_minus_control"] = {
        "full_rmse": report["v44"]["full_rmse"] - report["control"]["full_rmse"],
        "clean_rmse": report["v44"]["clean_rmse"] - report["control"]["clean_rmse"],
        "class_mse": delta_class,
    }
    log.info("delta class MSE: %s", {k: round(v) for k, v in delta_class.items()})
    log.info("paired price (control - v44) clean MSE: %+.0f (gate >= 300)",
             -delta_class["clean"])
    with open(os.path.join(MODELS, "lgbm_r_all_v44.features.txt"), "w") as f:
        f.write("\n".join(v44_cols))
    shutil.copy(os.path.join(MODELS, "lgbm_r_all_v26.encoders.pkl"),
                os.path.join(MODELS, "lgbm_r_all_v44.encoders.pkl"))
    with open(os.path.join(MODELS, "lgbm_r_all_v44.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
