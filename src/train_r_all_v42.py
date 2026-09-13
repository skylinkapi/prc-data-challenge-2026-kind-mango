"""Forward take-off order on the v40 base, paired against the v40 report.

Same cached frame, split and recipe as train_r_all_v40.py; the v40 members
of that run are the control. Adds order_earlier_eobt_next30 from
features_order_next.py and writes the members as lgbm_r_all_v42.
"""
import json
import logging
import os
import shutil

import numpy as np
import pandas as pd

from features_order_next import OUT as ORDER_NEXT
from train_r_all_v26 import HOLDOUT_MONTHS, MODELS, STOP_FRAC
from train_r_all_v40 import TEMPO_COLS, class_table, classify, load_frame, train_members

NEW_COLS = ["order_earlier_eobt_next30"]
log = logging.getLogger(__name__)


def main() -> None:
    dep, feat = load_frame()
    dep = dep.merge(pd.read_parquet(ORDER_NEXT), on="MVT_ID_mvt", how="left")
    cols = feat + TEMPO_COLS + NEW_COLS
    hold = dep["month"].isin(HOLDOUT_MONTHS)
    non_hold = dep.loc[~hold]
    stop_mask = np.random.default_rng(1234).random(len(non_hold)) < STOP_FRAC
    train, stop, test = non_hold[~stop_mask], non_hold[stop_mask], dep.loc[hold]
    y, sd, apt = (test["TAXITIME_SEC_mvt"].values.astype(float), test["sched_delay"].values.astype(float),
                  test["ADEP_mvt"].astype(str).values)
    pred, iters = train_members(train, stop, test, cols, "v42")
    with open(os.path.join(MODELS, "lgbm_r_all_v40.holdout.json")) as f:
        control = json.load(f)["v40"]
    report = {"control_v40": control, "v42": {"best_iters": iters, **class_table(pred, y, classify(y, sd), apt)}}
    report["delta_v42_minus_v40"] = {k: report["v42"][k] - control[k] for k in ("full_rmse", "clean_rmse")}
    with open(os.path.join(MODELS, "lgbm_r_all_v42.features.txt"), "w") as f:
        f.write("\n".join(cols))
    shutil.copy(os.path.join(MODELS, "lgbm_r_all_v26.encoders.pkl"), os.path.join(MODELS, "lgbm_r_all_v42.encoders.pkl"))
    with open(os.path.join(MODELS, "lgbm_r_all_v42.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)
    log.info("v42 FULL %.2f CLEAN %.2f iters %s delta %s", report["v42"]["full_rmse"], report["v42"]["clean_rmse"],
             iters, report["delta_v42_minus_v40"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
