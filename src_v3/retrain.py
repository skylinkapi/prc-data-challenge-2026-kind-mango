"""MP3 (light): fold retrain of the v47 recipe.

For one fold, train the v47 recipe (retuned params, v45 feature set) twice
with the same seed and score both models on the fold's hold-out. The delta
between the two paired reports is the identical-retrain noise for that
fold. Full 6-fold MP3 is a scheduling task; this runner takes a single
fold as argument.

The training uses v26 encoders as v47 does (a known leak per L1/L2 that
MP4 fixes; the point of MP3 here is the noise floor of the current recipe,
not the honest one).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.eval import FoldReport, score_fold
from src_v3.frames import load_v47_frame, v47_feature_names, fold_masks
from src_v3.folds import build_folds

log = logging.getLogger(__name__)


def _tuned() -> dict:
    with open(C.ROOT / "models" / "tune_lgbm_v43.tuned_params.json") as f:
        return json.load(f)["tuned"]


def _v46_iters() -> list[int]:
    with open(C.ROOT / "models" / "lgbm_r_all_v46.holdout.json") as f:
        return json.load(f)["v46"]["best_iters"]


def _train_one(train: pd.DataFrame, stop: pd.DataFrame, feat: list[str],
               tuned: dict, seed: int, patience: int = 100,
               max_rounds: int = 5000) -> lgb.Booster:
    cats = [c for c in C.CAT_COLS if c in feat]
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values,
                     categorical_feature=cats,
                     params={"linear_tree": True, "feature_pre_filter": False})
    dv = lgb.Dataset(stop[feat], label=stop["TAXITIME_SEC_mvt"].values,
                     categorical_feature=cats, reference=dt,
                     params={"linear_tree": True, "feature_pre_filter": False})
    params = {**tuned, "seed": seed, "bagging_seed": seed,
              "feature_fraction_seed": seed,
              "linear_tree": True, "bagging_freq": 5,
              "verbosity": -1, "num_threads": C.NUM_THREADS,
              "deterministic": C.DETERMINISTIC}
    return lgb.train(params, dt, num_boost_round=max_rounds, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(patience, verbose=False)])


def run_fold_retrain(hold: tuple[int, int], seed: int = C.GLOBAL_SEED,
                     n_replicates: int = 2, save_reports: bool = True
                     ) -> dict:
    """Train the v47 recipe `n_replicates` times on the (hold, early_stop)
    fold. Return the per-replicate FoldReport for the hold-out rows and the
    pairwise MSE delta.
    """
    folds = {f.hold: f for f in build_folds()}
    if hold not in folds:
        raise ValueError(f"hold {hold} not in {list(folds)}")
    fold = folds[hold]

    dep = load_v47_frame()
    feat = v47_feature_names()
    tr_m, st_m, ho_m = fold_masks(dep["month"], fold.hold, fold.early_stop)
    train = dep[tr_m]
    stop = dep[st_m]
    test = dep[ho_m]
    log.info("fold %s: train %d stop %d hold %d", hold, len(train), len(stop), len(test))

    tuned = _tuned()
    reports: list[FoldReport] = []
    per_replicate: list[dict] = []
    for r in range(n_replicates):
        t0 = time.time()
        b = _train_one(train, stop, feat, tuned, seed)
        pred = np.clip(b.predict(test[feat], num_iteration=b.best_iteration), 0, None)
        rep = score_fold(pred, test["TAXITIME_SEC_mvt"].values,
                         test["sched_delay"].values,
                         test["ADEP_mvt"].astype(str).values, hold)
        reports.append(rep)
        per_replicate.append({"replicate": r, "best_iter": b.best_iteration,
                              "seconds": time.time() - t0, "report": rep.to_dict()})
        log.info("fold %s replicate %d: FULL RMSE %.3f iter %d in %.0fs",
                 hold, r, rep.full_rmse, b.best_iteration, time.time() - t0)

    payload = {"hold": list(hold), "seed": seed, "n_replicates": n_replicates,
               "replicates": per_replicate}
    if n_replicates >= 2:
        d = np.array([r.full_mse for r in reports])
        payload["identical_retrain_mse_delta"] = {
            "min": float(d.min()), "max": float(d.max()),
            "spread": float(d.max() - d.min()),
            "mean": float(d.mean()), "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        }
    if save_reports:
        outp = C.MODELS / f"noise_v47_fold_{hold[0]}_{hold[1]}.json"
        outp.write_text(json.dumps(payload, indent=1))
        log.info("wrote %s", outp)
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hold", nargs=2, type=int, default=[1, 7],
                    help="the hold-out month pair (default: 1 7)")
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--seed", type=int, default=C.GLOBAL_SEED)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_fold_retrain((args.hold[0], args.hold[1]),
                     seed=args.seed, n_replicates=args.replicates)


if __name__ == "__main__":
    main()
