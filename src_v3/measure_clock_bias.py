"""Paired price of the clock-bias columns (v76) on the 2025 hold-out months 1 and 7.

Hold-out rows read the map of the ten months outside the hold-out, as
2026 rows read the 12-month map. Training rows read the map of the other
nine of those ten months (leave one month out). The treatment is one LightGBM member with the v43 parameters on
the v73 columns plus the three bias columns, MB3 rows of months 2 to 6
and 8 to 10, 2,000 rounds, seed 42. It is priced against the saved clock
member of `measure_nm_clocks.py`, which has the same rows, rounds and seed.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_clock_bias_v76 import FEATURES as BIAS, bias_columns, bias_oof, load_rows
from src_v3.build_nm_clocks_v73 import FEATURES as CLOCKS, OUT_TRAIN as CLOCKS_TRAIN
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, holdout_rows
from src_v3.measure_lgbm_control import fit_early_stopped
from src_v3.measure_mf2 import ROUNDS
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import HOLD, mse_delta

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rows = load_rows(sorted(C.TRAIN_DIR.glob("training_*.parquet")))
    fit = rows[~rows["month"].isin(HOLD)]
    bias = pd.concat([bias_oof(fit), bias_columns(rows[rows["month"].isin(HOLD)], fit)], ignore_index=True)
    dep, feat = load_frame((CLOCKS_TRAIN,))
    train = dep[dep["month"].isin(FIT_MONTHS)].merge(bias, on="MVT_ID_mvt", how="left", validate="m:1")
    del dep
    cols = feat + BIAS
    b = fit_early_stopped(train, None, cols, rounds=ROUNDS)
    h, _ = holdout_rows()
    h = h.merge(pd.read_parquet(CLOCKS_TRAIN), on="MVT_ID_mvt", how="left", validate="m:1") \
         .merge(bias, on="MVT_ID_mvt", how="left", validate="m:1")
    ref = pd.read_parquet(PRED_OUT)
    if not (ref["y"].values == h["TAXITIME_SEC_mvt"].values).all():
        raise AssertionError("The saved clock predictions are not in hold-out row order. "
                             "Rerun measure_nm_clocks, then rerun.")
    y, clocks = ref["y"].values, ref["clocks"].values
    apt = h["ADEP_mvt"].astype(str).values
    p = np.clip(b.predict(h[cols]), 0, None)
    report = {"columns": CLOCKS + BIAS, "bias_minus_clocks": mse_delta(p, clocks, y),
              "per_airport": {a: mse_delta(p[apt == a], clocks[apt == a], y[apt == a])
                              for a in sorted(set(apt))}}
    report["rows_over_7200"] = {"clocks": int((clocks > 7200).sum()), "bias": int((p > 7200).sum())}
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "clock_bias.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
