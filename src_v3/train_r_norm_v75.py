"""WINNING_PLAN L6 (track B): the LIRF `R_norm` members with the five NM clock columns.

v55 recipe otherwise: LIRF genuine rows (`abs(y - sd) >= 60`, `y < 80,000`)
of all 12 months, the 166 v41 columns, five seeds, the v41 rounds scaled
by 1/0.88. Writes `lgbm_r_norm_lirf_<tag>_s{42..46}` and its feature list.
`--extra` adds the columns of other parquets instead of the clocks
(third-edition K2, tag v99r: the ADS-B and OPDI pushback clocks).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import BEST_PARAMS, CAT_COLS
from train_r_norm_lirf_seeds import SEEDS

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import FEATURES as CLOCKS, OUT_TRAIN as CLOCKS_TRAIN
from src_v3.catboost_r_norm import load_frame as load_genuine
from src_v3.train_r_norm_v55 import SCALE, V41_HOLDOUT

log = logging.getLogger(__name__)


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    """LIRF genuine rows with the clock columns and categorical dtypes, and the v41 feature names."""
    dep, feat = load_genuine()
    dep = dep.merge(pd.read_parquet(CLOCKS_TRAIN), on="MVT_ID_mvt", how="left", validate="m:1")
    for c in CAT_COLS:
        dep[c] = dep[c].astype("category")
    return dep, feat


def fit(frame: pd.DataFrame, feat: list[str], seed: int, rounds: int) -> lgb.Booster:
    """One `R_norm` member with the v41 parameters and a fixed round count."""
    params = {**BEST_PARAMS, "objective": "regression", "metric": "rmse",
              "linear_tree": True, "linear_lambda": 1.0, "num_leaves": 127,
              "bagging_freq": 5, "verbosity": -1, "num_threads": C.NUM_THREADS,
              "deterministic": C.DETERMINISTIC, "seed": seed, "bagging_seed": seed,
              "feature_fraction_seed": seed}
    dt = lgb.Dataset(frame[feat], label=frame["TAXITIME_SEC_mvt"].astype(float).values,
                     categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})
    return lgb.train(params, dt, num_boost_round=rounds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v75")
    ap.add_argument("--extra", nargs="+", help="Parquets of added columns keyed on MVT_ID_mvt; default the NM clocks.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rounds = [math.ceil(i * SCALE) for i in json.loads(V41_HOLDOUT.read_text())["v41"]["best_iters"]]
    dep, feat = load_frame()
    if args.extra:
        for path in args.extra:
            cols = pd.read_parquet(C.ROOT / path).drop(columns="adsb_day_loaded", errors="ignore")
            dep = dep.merge(cols, on="MVT_ID_mvt", how="left", validate="m:1")
            feat += [c for c in cols.columns if c != "MVT_ID_mvt"]
    else:
        feat += CLOCKS
    log.info("LIRF genuine rows %d, features %d, rounds %s", len(dep), len(feat), rounds)
    for seed, n in zip(SEEDS, rounds):
        fit(dep, feat, seed, n).save_model(str(C.ROOT / "models" / f"lgbm_r_norm_lirf_{args.tag}_s{seed}.txt"))
        log.info("%s seed %d written", args.tag, seed)
    (C.ROOT / "models" / f"lirf_regime_{args.tag}.features.txt").write_text("\n".join(feat))


if __name__ == "__main__":
    main()
