"""Price the round count of the anchored MB5 member on the 2025 hold-out.

v76a trains the anchored target with the MB5 parameters for 2,284 rounds
(scaled), the early stop of the plain target. The anchored study stops at
1,202 rounds with the same parameters. One fit on months 2 to 6 and 8 to
10 is read at both counts and priced outside LIRF against the 2,284-round
reading; rows without `AOBT_3` take the saved clock member.
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
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, holdout_rows
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.tune_lgbm_v76 import OUT as TUNED_V76

ANCHORED_STUDY = C.ROOT / "models" / "tune_lgbm_v79a.tuned_params.json"
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    plain, anchored = (json.loads(p.read_text()) for p in (TUNED_V76, ANCHORED_STUDY))
    rounds = {"plain_stop": plain["best_iteration"], "anchored_stop": anchored["best_iteration"]}
    dep, feat = load_frame((CLOCKS_TRAIN,))
    train = dep[dep["month"].isin(FIT_MONTHS) & dep["mvt_aobt3"].notna()].copy()
    del dep
    train["TAXITIME_SEC_mvt"] -= train["mvt_aobt3"].clip(0, C.NM_CLOCK_CLIP_S)
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values, categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})
    b = lgb.train({**plain["tuned"], "num_threads": C.NUM_THREADS}, dt, num_boost_round=max(rounds.values()))
    h, _ = holdout_rows()
    h = h.merge(pd.read_parquet(CLOCKS_TRAIN), on="MVT_ID_mvt", how="left", validate="m:1")
    ref = pd.read_parquet(PRED_OUT)
    y, anchor = ref["y"].values, h["mvt_aobt3"].clip(0, C.NM_CLOCK_CLIP_S).values
    pred = {k: np.where(np.isnan(anchor), ref["clocks"].values, np.clip(anchor + b.predict(h[feat], num_iteration=n), 0, None))
            for k, n in rounds.items()}
    apt = h["ADEP_mvt"].astype(str).values
    report = {"rounds": rounds, "anchored_minus_plain_stop": mse_delta(pred["anchored_stop"], pred["plain_stop"], y),
              "per_airport": {a: mse_delta(pred["anchored_stop"][apt == a], pred["plain_stop"][apt == a], y[apt == a])
                              for a in sorted(set(apt))},
              "rows_over_7200": {k: int((p > 7200).sum()) for k, p in pred.items()}}
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "anchor_rounds.holdout.json").write_text(json.dumps(report, indent=1))
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, **{f"pred_{k}": p for k, p in pred.items()}}) \
      .to_parquet(C.MODELS / "holdout_lgb_anchor_mb5.parquet")


if __name__ == "__main__":
    main()
