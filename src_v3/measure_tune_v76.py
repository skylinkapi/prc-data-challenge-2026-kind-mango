"""Price the MB5 parameters against the served-style clock member on the 2025 hold-out.

`tune_lgbm_v76.py` compares both parameter sets early-stopped; the v43
control stopped at 249 rounds, while the served members train 767 to
2,265 rounds. This run fits the MB5 parameters at their early-stop round
count on the same rows and seed as the saved clock member of
`measure_nm_clocks.py` (v43 parameters, 2,000 rounds), and prices the
difference outside LIRF.
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
from src_v3.build_nm_clocks_v73 import OUT_TRAIN as CLOCKS_TRAIN
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, holdout_rows
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.tune_lgbm_v76 import OUT as TUNED_V76

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    study = json.loads(TUNED_V76.read_text())
    dep, feat = load_frame((CLOCKS_TRAIN,))
    train = dep[dep["month"].isin(FIT_MONTHS)]
    del dep
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values, categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})
    b = lgb.train({**study["tuned"], "num_threads": C.NUM_THREADS}, dt, num_boost_round=study["best_iteration"])
    h, _ = holdout_rows()
    h = h.merge(pd.read_parquet(CLOCKS_TRAIN), on="MVT_ID_mvt", how="left", validate="m:1")
    ref = pd.read_parquet(PRED_OUT)
    y, clocks = ref["y"].values, ref["clocks"].values
    p = np.clip(b.predict(h[feat]), 0, None)
    apt = h["ADEP_mvt"].astype(str).values
    report = {"rounds": study["best_iteration"], "mb5_minus_served_style": mse_delta(p, clocks, y),
              "rmse": {"mb5": float(np.sqrt(np.mean((p - y) ** 2))), "served_style": float(np.sqrt(np.mean((clocks - y) ** 2)))},
              "per_airport": {a: mse_delta(p[apt == a], clocks[apt == a], y[apt == a]) for a in sorted(set(apt))},
              "rows_over_7200": {"mb5": int((p > 7200).sum()), "served_style": int((clocks > 7200).sum())}}
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "tune_v76.holdout.json").write_text(json.dumps(report, indent=1))
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "pred": p}).to_parquet(C.MODELS / "holdout_lgb_mb5_v76.parquet")


if __name__ == "__main__":
    main()
