"""Price per-airport LightGBM members (v89) on the 2025 hold-out months 1 and 7.

One MB5 member per airport (v85 columns: v73 plus the forward windows,
MB3 rows of months 2 to 6 and 8 to 10, the v76 parameters and round
count, seed 42) scores that airport's hold-out rows. The report prices
the per-airport prediction, its mean with the saved global v85 member,
and that mean as the plain LightGBM term of the v85 blend, outside LIRF.
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
from src_v3.build_nm_forward_v85 import OUT_TRAIN as FORWARD_V85
from src_v3.catboost_base import load_frame
from src_v3.fit_blend_v77 import blend
from src_v3.measure_catboost_blend import FIT_MONTHS, holdout_rows
from src_v3.measure_extra_v81 import MB5_ANCHOR
from src_v3.measure_member_v77 import out_path
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.tune_lgbm_v76 import OUT as TUNED_V76

GLOBAL_V85 = C.MODELS / "holdout_lgb_v85.parquet"
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    study = json.loads(TUNED_V76.read_text())
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
        b = lgb.train({**study["tuned"], "num_threads": C.TUNE_V76_THREADS}, dt, num_boost_round=study["best_iteration"])
        per[apt == a] = np.clip(b.predict(h.loc[apt == a, feat]), 0, None)
        log.info("%s: %d training rows", a, len(tr))
    ref = pd.read_parquet(PRED_OUT)
    y, glob = ref["y"].values, pd.read_parquet(GLOBAL_V85)["pred"].values
    mean = 0.5 * (per + glob)
    members = {"lgb": glob, "lgb_anchor": pd.read_parquet(MB5_ANCHOR)["pred_plain_stop"].values,
               **{k: pd.read_parquet(out_path(p))["pred"].values for k, p in (("cat", "cat_plain"), ("cat_anchor", "cat_anchor"))}}
    served = blend(members, 0.5, 0.5, 0.5)
    report = {"per_minus_global": mse_delta(per, glob, y), "mean_minus_global": mse_delta(mean, glob, y),
              "blend_mean_minus_v85": mse_delta(blend({**members, "lgb": mean}, 0.5, 0.5, 0.5), served, y),
              "per_airport_mean": {a: mse_delta(mean[apt == a], glob[apt == a], y[apt == a]) for a in sorted(set(apt))},
              "rows_over_7200": {"per": int((per > 7200).sum()), "global": int((glob > 7200).sum())}}
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "per_airport_v89.holdout.json").write_text(json.dumps(report, indent=1))
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "pred": per}).to_parquet(C.MODELS / "holdout_lgb_per_airport_v89.parquet")


if __name__ == "__main__":
    main()
