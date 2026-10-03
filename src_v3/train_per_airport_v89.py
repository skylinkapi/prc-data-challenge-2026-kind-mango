"""v89: one LightGBM member per airport, for the plain and the anchored term.

The v85 columns, the MB3 rows of all 12 months, the v76 parameters and
round count scaled by the airport's row ratio, seed 42. Served as a 50/50
mean with the global v85 members of the same term (`predict_v67
--per-airport-tag`). Hold-out (`measure_per_airport_v89.py`): alone +1,112
MSE, the mean with the global member -657 MSE at all 9 airports.
`--seed` fits another seed under the tags `v89s<seed>` and `v89s<seed>a`
(WINNING_PLAN second edition, B2).
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
from train_lgbm_v21 import CAT_COLS

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import FEATURES as CLOCKS, OUT_TRAIN as CLOCKS_TRAIN
from src_v3.build_plan_nm_taxi_v65 import OUT_TRAIN as PLAN_NM_TRAIN
from src_v3.frames import v47_feature_names
from src_v3.train_v57_base import _load_frame_v57
from src_v3.tune_lgbm_v76 import OUT as TUNED_V76

FORWARD_TRAIN = C.ROOT / "models" / "nm_forward_train_v85.parquet"
log = logging.getLogger(__name__)


def model_path(tag: str, airport: str) -> Path:
    return C.ROOT / "models" / f"lgbm_pa_{tag}_{airport}.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", choices=("plain", "anchored"))
    ap.add_argument("--seed", type=int, default=C.GLOBAL_SEED)
    ap.add_argument("--tag", default="v89", help="Version tag (v97: with the OPDI and ADS-B pushback columns).")
    ap.add_argument("--extra", nargs="*", default=[str(FORWARD_TRAIN.relative_to(C.ROOT))],
                    help="Parquets of extra training columns keyed on MVT_ID_mvt, relative to the repo root.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    study = json.loads(TUNED_V76.read_text())
    dep, feat = _load_frame_v57(), v47_feature_names() + ["plan_nm_taxi", *CLOCKS]
    for path in (PLAN_NM_TRAIN, CLOCKS_TRAIN, *(C.ROOT / p for p in args.extra)):
        cols = pd.read_parquet(path)
        dep = dep.merge(cols, on="MVT_ID_mvt", how="left", validate="m:1")
        if path not in (PLAN_NM_TRAIN, CLOCKS_TRAIN):
            feat += [c for c in cols.columns if c != "MVT_ID_mvt"]
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    dep = dep[(dep["ADEP_mvt"].astype(str) != "LIRF") & (y > 0) & (y <= 80_000)]
    tag = (args.tag if args.seed == C.GLOBAL_SEED else f"{args.tag}s{args.seed}") + ("a" if args.base == "anchored" else "")
    if args.base == "anchored":
        dep = dep[dep["mvt_aobt3"].notna()].copy()
        dep["TAXITIME_SEC_mvt"] -= dep["mvt_aobt3"].clip(0, C.NM_CLOCK_CLIP_S)
    n_total = len(dep)
    rounds = math.ceil(study["best_iteration"] * n_total / study["n_fit"])
    for airport, grp in dep.groupby(dep["ADEP_mvt"].astype(str)):
        dt = lgb.Dataset(grp[feat], label=grp["TAXITIME_SEC_mvt"].values, categorical_feature=CAT_COLS,
                         params={"linear_tree": True, "feature_pre_filter": False})
        params = {**study["tuned"], "num_threads": C.PER_AIRPORT_THREADS,
                  "seed": args.seed, "bagging_seed": args.seed, "feature_fraction_seed": args.seed}
        lgb.train(params, dt, num_boost_round=rounds) \
           .save_model(str(model_path(tag, airport)))
        log.info("%s %s: %d rows, %d rounds", tag, airport, len(grp), rounds)
    (C.ROOT / "models" / f"lgbm_pa_{tag}.features.txt").write_text("\n".join(feat))


if __name__ == "__main__":
    main()
