"""MB5 (v76): the plain (v73) and anchored (v74a) LightGBM bases refit with the clock-era study parameters.

`--tag` and `--extra` add column files keyed on `MVT_ID_mvt` (v86: the
forward-window clocks and queue lengths of `build_nm_forward_v85.py`).
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import FEATURES as CLOCKS, OUT_TRAIN as CLOCKS_TRAIN
from src_v3.build_plan_nm_taxi_v65 import OUT_TRAIN as PLAN_NM_TRAIN
from src_v3.frames import v47_feature_names
from src_v3.train_v57_base import _load_frame_v57, train_base
from src_v3.tune_lgbm_v76 import OUT as TUNED_V76


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", choices=("plain", "anchored"))
    ap.add_argument("--tag", default="v76", help="Version tag; the anchored base gets the suffix 'a'.")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Parquets of extra training columns keyed on MVT_ID_mvt, relative to the repo root.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep = _load_frame_v57()
    feat = v47_feature_names() + ["plan_nm_taxi", *CLOCKS]
    for path in (PLAN_NM_TRAIN, CLOCKS_TRAIN, *(C.ROOT / p for p in args.extra)):
        cols = pd.read_parquet(path)
        dep = dep.merge(cols, on="MVT_ID_mvt", how="left", validate="m:1")
        if path not in (PLAN_NM_TRAIN, CLOCKS_TRAIN):
            feat += [c for c in cols.columns if c != "MVT_ID_mvt"]
    changed = f"v73 columns plus {args.extra}, MB5 clock-era parameters"
    if args.base == "plain":
        train_base(dep, feat, args.tag, changed, tuned_path=TUNED_V76)
    else:
        train_base(dep, feat, f"{args.tag}a", f"anchored target, {changed}",
                   anchor_col="mvt_aobt3", tuned_path=TUNED_V76)


if __name__ == "__main__":
    main()
