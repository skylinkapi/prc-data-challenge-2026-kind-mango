"""Paired price of the NM clock columns in the LIRF `R_norm` term (WINNING_PLAN L6) on the 2025 hold-out.

Two arms, one member each with the v41 parameters, on the LIRF genuine
rows of months 2 to 6 and 8 to 10, at a fixed 500 rounds: the 166 v41
columns (control) and the same plus the five clock columns. The report
gives clocks minus control at three round counts on the genuine rows of
months 1 and 7, on the 344,336-row scale. `--extra` prices the columns of
other parquets instead (third-edition K2: the ADS-B and OPDI clocks).
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import FEATURES as CLOCKS
from src_v3.measure_catboost_blend import FIT_MONTHS
from src_v3.measure_tail_rules import HOLD, mse_delta
from src_v3.train_r_norm_v75 import fit, load_frame

ROUNDS = 500
CHECKPOINTS = (250, 400, 500)
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="clocks")
    ap.add_argument("--extra", nargs="+", help="Parquets of treatment columns keyed on MVT_ID_mvt; default the NM clocks.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame()
    treat = list(CLOCKS)
    if args.extra:
        treat = []
        for path in args.extra:
            cols = pd.read_parquet(C.ROOT / path).drop(columns="adsb_day_loaded", errors="ignore")
            dep = dep.merge(cols, on="MVT_ID_mvt", how="left", validate="m:1")
            treat += [c for c in cols.columns if c != "MVT_ID_mvt"]
    train, h = dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(HOLD)]
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    cols = {"control": feat, "clocks": feat + treat}
    preds = {arm: {n: np.clip(b.predict(h[c], num_iteration=n), 0, None) for n in CHECKPOINTS}
             for arm, c in cols.items() for b in [fit(train, c, C.GLOBAL_SEED, ROUNDS)]}
    report = {"n_holdout": len(h),
              "rmse": {arm: {str(n): float(np.sqrt(np.mean((p[n] - y) ** 2))) for n in CHECKPOINTS}
                       for arm, p in preds.items()},
              "clocks_minus_control": {str(n): mse_delta(preds["clocks"][n], preds["control"][n], y)
                                       for n in CHECKPOINTS}}
    log.info("%s", json.dumps(report, indent=1))
    report["columns"] = treat
    (C.MODELS / f"lirf_{args.tag}.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
