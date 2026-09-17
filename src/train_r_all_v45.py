"""L9 ship: 3-seed R_all_v45 = v44 features + plan_taxi_res.

Saves lgbm_r_all_v45_s{42,43,44}.txt, lgbm_r_all_v45.features.txt and
lgbm_r_all_v45.encoders.pkl (copied from v26). The paired report already
exists at lgbm_r_all_v45plan.holdout.json.
"""
import logging
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS, HOLDOUT_MONTHS, MODELS, STOP_FRAC
from train_r_all_v40 import CACHE, ID_MAP, TEMPO_COLS, V2_FRAME
from train_r_all_v44 import EXTRA_COLS as EOBT_P2575
from train_r_all_v45plan import EXTRA_COLS as PLAN_COLS
from train_r_all_v45plan import PLAN

log = logging.getLogger(__name__)


def main() -> None:
    with open(CACHE + ".feat.txt") as f:
        feat = f.read().split()
    dep = pd.read_parquet(CACHE)
    dep["MVT_ID_mvt"] = pd.read_parquet(ID_MAP)["MVT_ID_mvt"].values
    tempo = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    dep = dep.merge(pd.read_parquet(os.path.join(MODELS, "tempo_p2575_train.parquet")),
                    on="MVT_ID_mvt", how="left")
    dep = dep.merge(pd.read_parquet(PLAN), on="MVT_ID_mvt", how="left")

    v45_cols = feat + TEMPO_COLS + EOBT_P2575 + PLAN_COLS
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    non_hold = dep.loc[~hold]
    stop_mask = np.random.default_rng(1234).random(len(non_hold)) < STOP_FRAC
    train, stop = non_hold[~stop_mask], non_hold[stop_mask]
    log.info("train %d stop %d features %d", len(train), len(stop), len(v45_cols))

    dt = lgb.Dataset(train[v45_cols], label=train["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS)
    dv = lgb.Dataset(stop[v45_cols], label=stop["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS, reference=dt)
    for seed in DEFAULT_SEEDS:
        params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
                  "linear_tree": True, "linear_lambda": 1.0, "num_leaves": 220,
                  "seed": seed, "bagging_seed": seed, "bagging_freq": 5,
                  "verbosity": -1, "num_threads": -1}
        t0 = time.time()
        b = lgb.train(params, dt, num_boost_round=5000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        b.save_model(os.path.join(MODELS, f"lgbm_r_all_v45_s{seed}.txt"))
        log.info("v45 seed %d best iter %d in %.0fs", seed, b.best_iteration, time.time() - t0)
    with open(os.path.join(MODELS, "lgbm_r_all_v45.features.txt"), "w") as f:
        f.write("\n".join(v45_cols))
    shutil.copy(os.path.join(MODELS, "lgbm_r_all_v26.encoders.pkl"),
                os.path.join(MODELS, "lgbm_r_all_v45.encoders.pkl"))
    log.info("wrote lgbm_r_all_v45 members (%d cols)", len(v45_cols))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
