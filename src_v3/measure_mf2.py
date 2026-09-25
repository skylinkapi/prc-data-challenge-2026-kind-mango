"""Paired price of the MF2 runway-queue columns on the 2025 hold-out months 1 and 7.

Control and treatment are one LightGBM member each with the v43 parameters,
on the MB3 rows of months 2 to 6 and 8 to 10. Both train a fixed 2,000
rounds, and the price is read at 500, 1,000, 1,500 and 2,000 rounds: an
early-stopped control stopped at 438 rounds against 1,959 for the
treatment (`mf2_earlystop.holdout.json`), so equal rounds give the fair
comparison. The served base also trains a fixed round count.
The control reads the v65 columns; the treatment adds the seven MF2
columns. v71 is built only if the treatment beats the control by at least
`MIN_GAIN_MSE` at every checkpoint, on the hold-out rows outside LIRF.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_runway_queue_v71 import FEATURES as MF2, OUT_TRAIN
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, holdout_rows
from src_v3.measure_lgbm_control import fit_early_stopped
from src_v3.measure_tail_rules import mse_delta

MIN_GAIN_MSE = 150
ROUNDS = 2_000
CHECKPOINTS = (500, 1_000, 1_500, 2_000)
log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mf2 = pd.read_parquet(OUT_TRAIN)
    dep, feat = load_frame()
    dep = dep.merge(mf2, on="MVT_ID_mvt", how="left", validate="m:1")
    train = dep[dep["month"].isin(FIT_MONTHS)]
    h, _ = holdout_rows()
    h = h.merge(mf2, on="MVT_ID_mvt", how="left", validate="m:1")
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    apt = h["ADEP_mvt"].astype(str).values
    cols = {"control": feat, "mf2": feat + MF2}
    boosters = {arm: fit_early_stopped(train, None, c, rounds=ROUNDS) for arm, c in cols.items()}
    report = {"n_holdout": int(len(h)), "min_gain_mse": MIN_GAIN_MSE, "mf2_minus_control": {}}
    for n in CHECKPOINTS:
        p = {arm: np.clip(b.predict(h[cols[arm]], num_iteration=n), 0, None) for arm, b in boosters.items()}
        report["mf2_minus_control"][str(n)] = {
            "all": mse_delta(p["mf2"], p["control"], y),
            **{a: mse_delta(p["mf2"][apt == a], p["control"][apt == a], y[apt == a]) for a in sorted(set(apt))}}
    report["build_v71"] = all(v["all"] <= -MIN_GAIN_MSE for v in report["mf2_minus_control"].values())
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "mf2.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
