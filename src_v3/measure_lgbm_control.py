"""Control for the CatBoost price: LightGBM on the same columns, rows and months.

`measure_catboost_blend.py` compares CatBoost on the v65 columns with the
v46 members on the v45 columns. This run fits one LightGBM member with the
v43 parameters on the v65 columns, the MB3 rows and months 2 to 6 and 8 to
10, early-stops on months 11 and 12, and prices it against v46 on the same
hold-out rows. CatBoost minus this delta is the price of the model class.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS

from src_v3 import config as C
from src_v3.catboost_base import HOLDOUT_REPORT, load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS, holdout_rows
from src_v3.measure_tail_rules import mse_delta
from src_v3.train_v57_base import TUNED

log = logging.getLogger(__name__)


def fit_early_stopped(train: pd.DataFrame, stop: pd.DataFrame | None, feat: list[str],
                      rounds: int = 5000) -> lgb.Booster:
    """One LightGBM member with the v43 parameters, early-stopped on `stop`, or `rounds` fixed when `stop` is None."""
    params = {**json.loads(TUNED.read_text())["tuned"], "linear_tree": True, "verbosity": -1,
              "num_threads": C.NUM_THREADS, "deterministic": C.DETERMINISTIC,
              "seed": C.GLOBAL_SEED, "bagging_seed": C.GLOBAL_SEED,
              "feature_fraction_seed": C.GLOBAL_SEED}
    ds = {"categorical_feature": CAT_COLS, "params": {"linear_tree": True, "feature_pre_filter": False}}
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values, **ds)
    if stop is None:
        return lgb.train(params, dt, num_boost_round=rounds)
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values, reference=dt, **ds)
    return lgb.train(params, dt, num_boost_round=rounds, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame()
    train, stop = dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)]
    b = fit_early_stopped(train, stop, feat)
    h, v46_pred = holdout_rows()
    pred = np.clip(b.predict(h[feat], num_iteration=b.best_iteration), 0, None)
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    cat_alone = json.loads(HOLDOUT_REPORT.read_text())["catboost_alone"]
    report = {"best_iteration": int(b.best_iteration), "lgbm_v65_cols_alone": mse_delta(pred, v46_pred, y),
              "catboost_alone": cat_alone}
    report["catboost_minus_lgbm_control"] = cat_alone - report["lgbm_v65_cols_alone"]
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "lgbm_control.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
