"""Price the CatBoost blend (L3) on the 2025 hold-out months 1 and 7.

CatBoost trains on the v46 months (2 to 6 and 8 to 10) of the MB3 rows and
early-stops on months 11 and 12, as v46 did. The v46 LightGBM members
score the same hold-out rows. The report gives the MSE delta of
`(1 - w) * v46 + w * CatBoost` for each pre-registered weight, overall and
per airport, and the round count that the full-year fit scales.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_plan_nm_taxi_v65 import OUT_TRAIN as PLAN_NM_TRAIN
from src_v3.catboost_base import fit, load_frame, paths, predict
from src_v3.frames import load_v47_frame
from src_v3.measure_tail_rules import HOLD, N_HOLDOUT, mse_delta, score_holdout
from src_v3.train_v57_base import _load_frame_v57

FIT_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10)
STOP_MONTHS = (11, 12)
V67_ROUND_CAP = 5_000
log = logging.getLogger(__name__)


def holdout_rows() -> tuple[pd.DataFrame, np.ndarray]:
    """v65 frame of the hold-out rows outside LIRF, and the v46 prediction on the same rows."""
    v65 = _load_frame_v57().merge(pd.read_parquet(PLAN_NM_TRAIN), on="MVT_ID_mvt",
                                  how="left", validate="m:1")
    v47 = load_v47_frame()
    if not (v65["TAXITIME_SEC_mvt"].values == v47["TAXITIME_SEC_mvt"].values).all():
        raise AssertionError("The v65 and v47 frames are not in the same row order. "
                             "Align them on the H1 row index, then rerun.")
    keep = (v65["month"].isin(HOLD) & (v65["ADEP_mvt"].astype(str) != "LIRF")
            & (v65["TAXITIME_SEC_mvt"] > 0)).values
    return v65[keep].reset_index(drop=True), score_holdout(v47[keep])


def blend_table(lgb_pred: np.ndarray, cat_pred: np.ndarray, y: np.ndarray, apt: np.ndarray) -> dict:
    """MSE delta of each pre-registered blend against the LightGBM prediction, overall and per airport."""
    table = {}
    for w in C.CATBOOST_BLEND_WEIGHTS:
        p = (1 - w) * lgb_pred + w * cat_pred
        table[str(w)] = {"all": mse_delta(p, lgb_pred, y), **{
            a: mse_delta(p[apt == a], lgb_pred[apt == a], y[apt == a]) for a in sorted(set(apt))}}
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v67", help="Version tag of the hold-out report.")
    ap.add_argument("--max-rounds", type=int, help="Round cap of the early-stopped fit; default is the config value.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame()
    train, stop = dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)]
    model = fit(train, feat, stop=stop, max_rounds=args.max_rounds)
    h, lgb_pred = holdout_rows()
    cat_pred = predict(model, h, feat)
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    apt = h["ADEP_mvt"].astype(str).values
    best = int(model.get_best_iteration()) + 1
    report = {"best_iteration": best, "n_fit": int(len(train)),
              "n_holdout": int(len(h)), "scale": N_HOLDOUT,
              "catboost_alone": mse_delta(cat_pred, lgb_pred, y),
              "blend": blend_table(lgb_pred, cat_pred, y, apt)}
    if best > V67_ROUND_CAP:
        capped = predict(model, h, feat, ntree_end=V67_ROUND_CAP)
        report["blend_first_5000_trees"] = blend_table(lgb_pred, capped, y, apt)
    log.info("%s", json.dumps(report, indent=1))
    paths(args.tag)[1].write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
