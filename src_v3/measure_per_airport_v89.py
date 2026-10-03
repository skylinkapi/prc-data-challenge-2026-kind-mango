"""Price per-airport LightGBM members (v89) on the 2025 hold-out months 1 and 7.

One MB5 member per airport (v85 columns: v73 plus the forward windows,
MB3 rows of months 2 to 6 and 8 to 10, the v76 parameters and round
count, seed 42) scores that airport's hold-out rows. The report prices
the per-airport prediction, its mean with the saved global v85 member,
and that mean as the plain LightGBM term of the v85 blend, outside LIRF.
`--seed` fits another seed (WINNING_PLAN second edition, B2) and prices
the mean of every seed on disk against seed 42 inside the v89-style blend.
"""
from __future__ import annotations

import argparse
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
from src_v3.fit_blend_v77 import blend
from src_v3.measure_catboost_blend import FIT_MONTHS, holdout_rows
from src_v3.measure_extra_v81 import MB5_ANCHOR
from src_v3.measure_member_v77 import out_path
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.tune_lgbm_v76 import OUT as TUNED_V76

GLOBAL_V85 = C.MODELS / "holdout_lgb_v85.parquet"
FORWARD_V85 = C.ROOT / "models" / "nm_forward_train_v85.parquet"
log = logging.getLogger(__name__)


def holdout_path(seed: int) -> Path:
    suffix = "" if seed == C.GLOBAL_SEED else f"_s{seed}"
    return C.MODELS / f"holdout_lgb_per_airport_v89{suffix}.parquet"


def price_seeds(glob: np.ndarray, members: dict, y: np.ndarray, apt: np.ndarray) -> dict:
    """Seed mean against seed 42, both as the per-airport half of the plain term of the v89-style blend."""
    seeds = [s for s in C.PER_AIRPORT_SEEDS if holdout_path(s).exists()]
    per = {s: pd.read_parquet(holdout_path(s))["pred"].values for s in seeds}
    one = blend({**members, "lgb": 0.5 * (per[C.GLOBAL_SEED] + glob)}, *C.BLEND_V77_SERVED)
    avg = blend({**members, "lgb": 0.5 * (np.mean(list(per.values()), axis=0) + glob)}, *C.BLEND_V77_SERVED)
    return {"seeds": seeds, "blend_seed_mean_minus_seed42": mse_delta(avg, one, y),
            "per_airport": {a: mse_delta(avg[apt == a], one[apt == a], y[apt == a]) for a in sorted(set(apt))}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=C.GLOBAL_SEED)
    ap.add_argument("--threads", type=int, default=C.TUNE_V76_THREADS)
    args = ap.parse_args()
    study = json.loads(TUNED_V76.read_text())
    params = {**study["tuned"], "num_threads": args.threads,
              "seed": args.seed, "bagging_seed": args.seed, "feature_fraction_seed": args.seed}
    dep, feat = load_frame((CLOCKS_TRAIN, FORWARD_V85))
    train = dep[dep["month"].isin(FIT_MONTHS)]
    del dep
    h, _ = holdout_rows()
    for path in (CLOCKS_TRAIN, FORWARD_V85):
        h = h.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    apt = h["ADEP_mvt"].astype(str).values
    per = np.full(len(h), np.nan)
    for a in sorted(set(apt)):
        tr = train[train["ADEP_mvt"].astype(str) == a]
        dt = lgb.Dataset(tr[feat], label=tr["TAXITIME_SEC_mvt"].values, categorical_feature=CAT_COLS,
                         params={"linear_tree": True, "feature_pre_filter": False})
        b = lgb.train(params, dt, num_boost_round=study["best_iteration"])
        per[apt == a] = np.clip(b.predict(h.loc[apt == a, feat]), 0, None)
        log.info("%s: %d training rows", a, len(tr))
    ref = pd.read_parquet(PRED_OUT)
    y, glob = ref["y"].values, pd.read_parquet(GLOBAL_V85)["pred"].values
    mean = 0.5 * (per + glob)
    members = {"lgb": glob, "lgb_anchor": pd.read_parquet(MB5_ANCHOR)["pred_plain_stop"].values,
               **{k: pd.read_parquet(out_path(p))["pred"].values for k, p in (("cat", "cat_plain"), ("cat_anchor", "cat_anchor"))}}
    served = blend(members, 0.5, 0.5, 0.5)
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "pred": per}).to_parquet(holdout_path(args.seed))
    if args.seed != C.GLOBAL_SEED:
        report = price_seeds(glob, members, y, apt)
        log.info("%s", json.dumps(report, indent=1))
        (C.MODELS / "per_airport_seeds_b2.holdout.json").write_text(json.dumps(report, indent=1))
        return
    report = {"per_minus_global": mse_delta(per, glob, y), "mean_minus_global": mse_delta(mean, glob, y),
              "blend_mean_minus_v85": mse_delta(blend({**members, "lgb": mean}, 0.5, 0.5, 0.5), served, y),
              "per_airport_mean": {a: mse_delta(mean[apt == a], glob[apt == a], y[apt == a]) for a in sorted(set(apt))},
              "rows_over_7200": {"per": int((per > 7200).sum()), "global": int((glob > 7200).sum())}}
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "per_airport_v89.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
