"""Price extra base columns on the MB5 member and inside the v76 blend on the 2025 hold-out.

The treatment is the MB5 LightGBM member (`tune_lgbm_v76.py` parameters and
round count, seed 42, MB3 rows of months 2 to 6 and 8 to 10) with the v73
columns plus the columns of the given parquets. The control is the saved
MB5 member without them (`measure_tune_v76.py`). The report prices the
member alone and as the plain LightGBM term of the four-member v76 blend,
outside LIRF.
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
from src_v3.measure_member_v77 import out_path
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.tune_lgbm_v76 import OUT as TUNED_V76

MB5_MEMBER = C.MODELS / "holdout_lgb_mb5_v76.parquet"
MB5_ANCHOR = C.MODELS / "holdout_lgb_anchor_mb5.parquet"
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag")
    ap.add_argument("extra", nargs="+", help="Parquets of extra training columns keyed on MVT_ID_mvt.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    study = json.loads(TUNED_V76.read_text())
    extra = tuple(C.ROOT / p for p in args.extra)
    dep, feat = load_frame((CLOCKS_TRAIN, *extra))
    train = dep[dep["month"].isin(FIT_MONTHS)]
    del dep
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values, categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})
    b = lgb.train({**study["tuned"], "num_threads": C.NUM_THREADS}, dt, num_boost_round=study["best_iteration"])
    h, _ = holdout_rows()
    for path in (CLOCKS_TRAIN, *extra):
        h = h.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    ref = pd.read_parquet(PRED_OUT)
    y, apt = ref["y"].values, h["ADEP_mvt"].astype(str).values
    new, old = np.clip(b.predict(h[feat]), 0, None), pd.read_parquet(MB5_MEMBER)["pred"].values
    members = {"lgb": old, "lgb_anchor": pd.read_parquet(MB5_ANCHOR)["pred_plain_stop"].values,
               **{k: pd.read_parquet(out_path(p))["pred"].values for k, p in (("cat", "cat_plain"), ("cat_anchor", "cat_anchor"))}}
    served = blend(members, 0.5, 0.5, 0.5)
    report = {"columns": feat[-sum(len(pd.read_parquet(p).columns) - 1 for p in extra):],
              "member_minus_mb5": mse_delta(new, old, y),
              "blend_minus_v76": mse_delta(blend({**members, "lgb": new}, 0.5, 0.5, 0.5), served, y),
              "per_airport": {a: mse_delta(new[apt == a], old[apt == a], y[apt == a]) for a in sorted(set(apt))},
              "rows_over_7200": {"new": int((new > 7200).sum()), "mb5": int((old > 7200).sum())}}
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / f"extra_{args.tag}.holdout.json").write_text(json.dumps(report, indent=1))
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "pred": new}).to_parquet(C.MODELS / f"holdout_lgb_{args.tag}.parquet")


if __name__ == "__main__":
    main()
