"""L1: paired 3-seed base with the retuned hyperparameters on v45's features.

Loads the tuned params from tune_lgbm_v43.tuned_params.json, keeps
linear_tree=True, and trains three seeds on the v45 feature set (117 cols =
v44 features + plan_taxi_res). Scores paired vs the shipped v45 members.
"""
import json
import logging
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_lgbm_v21 import CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS, HOLDOUT_MONTHS, MODELS, STOP_FRAC
from train_r_all_v40 import CACHE, ID_MAP, TEMPO_COLS, V2_FRAME, class_table, classify
from train_r_all_v44 import EXTRA_COLS as EOBT_P2575
from train_r_all_v45plan import EXTRA_COLS as PLAN_COLS
from train_r_all_v45plan import PLAN

TUNED = os.path.join(MODELS, "tune_lgbm_v43.tuned_params.json")
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
    log.info("frame %d rows", len(dep))
    return dep, feat


def train_v45_predict(test: pd.DataFrame, feat: list[str]) -> np.ndarray:
    preds = []
    for s in DEFAULT_SEEDS:
        b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_all_v45_s{s}.txt"))
        preds.append(np.clip(b.predict(test[feat]), 0, None))
    return np.mean(preds, axis=0)


def train_v46_members(train, stop, test, feat, params_base) -> tuple[np.ndarray, list[int]]:
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS, reference=dt,
                     params={"linear_tree": True, "feature_pre_filter": False})
    preds, iters = [], []
    for s in DEFAULT_SEEDS:
        p = {**params_base, "seed": s, "bagging_seed": s,
             "feature_fraction_seed": s, "linear_tree": True,
             "bagging_freq": 5, "verbosity": -1, "num_threads": -1}
        t0 = time.time()
        b = lgb.train(p, dt, num_boost_round=5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        log.info("v46 seed %d best iter %d in %.0fs", s, b.best_iteration, time.time() - t0)
        b.save_model(os.path.join(MODELS, f"lgbm_r_all_v46_s{s}.txt"))
        preds.append(np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None))
        iters.append(b.best_iteration)
    return np.mean(preds, axis=0), iters


def main() -> None:
    with open(TUNED) as f:
        tuned = json.load(f)["tuned"]
    log.info("tuned params: %s", {k: round(v, 5) if isinstance(v, float) else v for k, v in tuned.items()})

    dep, feat = load_frame()
    v45_cols = feat + TEMPO_COLS + EOBT_P2575 + PLAN_COLS
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    non_hold = dep.loc[~hold]
    stop_mask = np.random.default_rng(1234).random(len(non_hold)) < STOP_FRAC
    train, stop, test = non_hold[~stop_mask], non_hold[stop_mask], dep.loc[hold]
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd = test["sched_delay"].values.astype(float)
    apt = test["ADEP_mvt"].astype(str).values
    cls = classify(y, sd)

    log.info("scoring shipped v45 members on the same hold-out")
    p_v45 = train_v45_predict(test, v45_cols)
    log.info("training v46 members with the retuned params")
    p_v46, iters = train_v46_members(train, stop, test, v45_cols, tuned)

    report = {"tuned": tuned,
              "control_v45": class_table(p_v45, y, cls, apt),
              "v46": {"best_iters": iters, **class_table(p_v46, y, cls, apt)}}
    served = {}
    for k in ("clean", "fallback", "tail", "24h"):
        ctl = report["control_v45"]["class_mse"][k] - report["control_v45"]["per_airport"]["LIRF"][k]
        new = report["v46"]["class_mse"][k] - report["v46"]["per_airport"]["LIRF"][k]
        served[k] = new - ctl
    report["delta_served_v46_minus_v45"] = served
    for k in ("control_v45", "v46"):
        log.info("%s FULL %.2f CLEAN %.2f", k, report[k]["full_rmse"], report[k]["clean_rmse"])
    log.info("delta served (non-LIRF): %s", {k: round(v) for k, v in served.items()})
    log.info("served clean price: %+.0f MSE (gate >= 500)", -served["clean"])
    with open(os.path.join(MODELS, "lgbm_r_all_v46.features.txt"), "w") as f:
        f.write("\n".join(v45_cols))
    shutil.copy(os.path.join(MODELS, "lgbm_r_all_v26.encoders.pkl"),
                os.path.join(MODELS, "lgbm_r_all_v46.encoders.pkl"))
    with open(os.path.join(MODELS, "lgbm_r_all_v46.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
