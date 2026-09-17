"""L7: paired test of weather-at-EOBT columns on top of v45 features."""
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
from train_r_all_v45plan import EXTRA_COLS as PLAN_COLS
from train_r_all_v45plan import PLAN

WX_EOBT = os.path.join(MODELS, "weather_eobt_train.parquet")
EXTRA_COLS = ["tmpc_eobt", "vis_km_eobt", "wind_kt_eobt", "wx_precip_eobt", "deicing_gate_eobt"]
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
    dep = dep.merge(pd.read_parquet(WX_EOBT), on="MVT_ID_mvt", how="left")
    return dep, feat


def score_v45(test: pd.DataFrame, feat: list[str]) -> np.ndarray:
    preds = []
    for s in DEFAULT_SEEDS:
        b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_all_v45_s{s}.txt"))
        preds.append(np.clip(b.predict(test[feat]), 0, None))
    return np.mean(preds, axis=0)


def train_v48_members(train, stop, test, feat) -> tuple[np.ndarray, list[int]]:
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS)
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS, reference=dt)
    preds, iters = [], []
    for s in DEFAULT_SEEDS:
        p = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
             "linear_tree": True, "linear_lambda": 1.0, "num_leaves": 220,
             "seed": s, "bagging_seed": s, "feature_fraction_seed": s,
             "bagging_freq": 5, "verbosity": -1, "num_threads": -1}
        t0 = time.time()
        b = lgb.train(p, dt, num_boost_round=5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        log.info("v48wx seed %d best iter %d in %.0fs", s, b.best_iteration, time.time() - t0)
        preds.append(np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None))
        iters.append(b.best_iteration)
    return np.mean(preds, axis=0), iters


def main() -> None:
    dep, feat = load_frame()
    v45_cols = feat + TEMPO_COLS + EOBT_P2575 + PLAN_COLS
    v48_cols = v45_cols + EXTRA_COLS
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    non_hold = dep.loc[~hold]
    stop_mask = np.random.default_rng(1234).random(len(non_hold)) < STOP_FRAC
    train, stop, test = non_hold[~stop_mask], non_hold[stop_mask], dep.loc[hold]
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd = test["sched_delay"].values.astype(float)
    apt = test["ADEP_mvt"].astype(str).values
    cls = classify(y, sd)

    p_v45 = score_v45(test, v45_cols)
    p_v48, iters = train_v48_members(train, stop, test, v48_cols)
    report = {"extra_cols": EXTRA_COLS,
              "control_v45": class_table(p_v45, y, cls, apt),
              "v48wx": {"best_iters": iters, **class_table(p_v48, y, cls, apt)}}
    served = {}
    for k in ("clean", "fallback", "tail", "24h"):
        ctl = report["control_v45"]["class_mse"][k] - report["control_v45"]["per_airport"]["LIRF"][k]
        new = report["v48wx"]["class_mse"][k] - report["v48wx"]["per_airport"]["LIRF"][k]
        served[k] = new - ctl
    report["delta_served_v48wx_minus_v45"] = served
    for k in ("control_v45", "v48wx"):
        log.info("%s FULL %.2f CLEAN %.2f", k, report[k]["full_rmse"], report[k]["clean_rmse"])
    log.info("delta served: %s", {k: round(v) for k, v in served.items()})
    log.info("served clean price: %+.0f MSE (gate >= 300)", -served["clean"])
    with open(os.path.join(MODELS, "lgbm_r_all_v48wx.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
