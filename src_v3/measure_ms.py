"""Measure MS1-MS4 on the (1,7) fold with the shipped v46 3 members.

v46 was paired-trained on months {2..12} \\ {1, 7} with a random-row stop
(P4). Our fold splits early-stop months 2 and 8 out of training, so
strictly v46 has a small in-sample overlap on months 2 and 8 with our
fold protocol. The measurement is still directional and shows the size of
MS1-MS4's post-processing effect; a fold-honest run is a Phase 2 task.

The runner:
  1. Loads the v47 feature frame + fold (1,7) hold-out rows.
  2. Predicts with the three v46 members individually on those rows.
  3. Builds the 2025 support table on the fold's training months only.
  4. Reports FoldReport for each of:
     - v46 unbounded mean (baseline)
     - MS1 mean-only bound
     - MS3 disagreement gate only
     - full MS1+MS2+MS3 pipeline
  5. Writes models/v3/measure_ms_fold_1_7.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.eval import FoldReport, score_fold
from src_v3.folds import build_folds
from src_v3.frames import load_v47_frame, v47_feature_names, fold_masks
from src_v3.postprocess import apply_ms1, apply_ms2, apply_ms3, apply_pipeline
from src_v3.support import build_support, write_support

MODELS_OLD = C.ROOT / "models"
V46_SEEDS = (42, 43, 44)
log = logging.getLogger(__name__)


def _score_v46_members(test: pd.DataFrame, feat: list[str]) -> np.ndarray:
    preds = []
    for s in V46_SEEDS:
        b = lgb.Booster(model_file=str(MODELS_OLD / f"lgbm_r_all_v46_s{s}.txt"))
        p = np.clip(b.predict(test[feat]), 0, None)
        preds.append(p)
    return np.array(preds)


def _fold_report(pred: np.ndarray, test: pd.DataFrame) -> FoldReport:
    return score_fold(pred,
                      test["TAXITIME_SEC_mvt"].values,
                      test["sched_delay"].values,
                      test["ADEP_mvt"].astype(str).values,
                      (1, 7))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hold", nargs=2, type=int, default=[1, 7])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    fold_map = {f.hold: f for f in build_folds()}
    fold = fold_map[(args.hold[0], args.hold[1])]

    log.info("loading v47 frame")
    dep = load_v47_frame()
    feat = v47_feature_names()
    tr_m, st_m, ho_m = fold_masks(dep["month"], fold.hold, fold.early_stop)
    train_fold = dep[tr_m | st_m]         # v46 was trained on these months
    test = dep[ho_m]
    log.info("fold %s: train_or_stop %d hold %d", fold.hold, len(train_fold), len(test))

    log.info("building 2025 fold-training support")
    support = build_support(train_fold)
    write_support(C.MODELS / f"support_fold_{fold.hold[0]}_{fold.hold[1]}.json", support)

    log.info("scoring v46 members on the hold-out")
    t0 = time.time()
    members = _score_v46_members(test, feat)
    mean = members.mean(axis=0)
    log.info("member scoring in %.0fs", time.time() - t0)

    adep = test["ADEP_mvt"].astype(str).values
    flt_null = test["flt_null"].astype(int).values
    mvt_eobt1 = test["mvt_eobt1"].astype(float).values

    variants: dict[str, np.ndarray] = {"v46_unbounded_mean": mean}
    variants["MS1_only"], _ = apply_ms1(mean.copy(), adep, flt_null, mvt_eobt1, support)
    variants["MS2_only"], _ = apply_ms2(mean.copy(), adep, support)
    variants["MS3_only"], _ = apply_ms3(members, adep)
    variants["MS1_MS2_MS3"], moved = apply_pipeline(members, adep, flt_null, mvt_eobt1, support)

    reports: dict[str, dict] = {}
    for name, pred in variants.items():
        rep = _fold_report(pred, test)
        reports[name] = rep.to_dict()
        log.info("%s FULL RMSE %.3f class %s", name, rep.full_rmse,
                 {c: round(v) for c, v in rep.class_mse.items()})

    # Served class delta (non-LIRF) against unbounded baseline
    baseline = reports["v46_unbounded_mean"]
    for name in ("MS1_only", "MS2_only", "MS3_only", "MS1_MS2_MS3"):
        served_delta = {}
        for cls in ("low", "clean", "fallback", "tail", "24h"):
            ctl = sum(v[cls] for a, v in baseline["per_airport"].items() if a != "LIRF")
            new = sum(v[cls] for a, v in reports[name]["per_airport"].items() if a != "LIRF")
            served_delta[cls] = new - ctl
        reports[name + "_served_delta"] = served_delta
        log.info("%s served delta (non-LIRF): %s", name,
                 {c: round(v, 1) for c, v in served_delta.items()})

    outp = C.MODELS / f"measure_ms_fold_{fold.hold[0]}_{fold.hold[1]}.json"
    payload = {"hold": list(fold.hold), "n_test": int(len(test)),
               "reports": reports, "pipeline_moved": moved}
    outp.write_text(json.dumps(payload, indent=1))
    log.info("wrote %s", outp)


if __name__ == "__main__":
    main()
