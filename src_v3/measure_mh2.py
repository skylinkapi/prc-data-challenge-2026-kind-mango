"""MH2: price a fallback head outside LIRF on the 2025 hold-out months 1 and 7.

A LightGBM classifier learns `p` = P(|y - sd| < 60) on the MB3 rows of
months 2 to 6 and 8 to 10 with the v65 columns, early-stops on months 11
and 12, and takes an isotonic map fit on its months 11 and 12 scores. On
the hold-out rows the served-style base is `0.5 v46 + 0.5 CatBoost A`
(`catboost_pair_holdout.parquet`). The head serves
`base + alpha * p * (sd - base)`; alpha 1 is the LIRF mixture, alpha 0.5
allows for the fallback rows the base already learned.

The report gives the MSE delta per airport and alpha. v72 is built only if
the airports with a negative delta sum to at least `MIN_GAIN_MSE`.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS

from src_v3 import config as C
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS, holdout_rows
from src_v3.measure_catboost_pair import PREDICTIONS
from src_v3.measure_tail_rules import mse_delta
from src_v3.train_p_fb_v56 import GATE_PARAMS

FB_TOL = C.FB_TOL_DEFAULT
ALPHAS = (1.0, 0.5)
MIN_GAIN_MSE = 150
log = logging.getLogger(__name__)


def is_fallback(frame: pd.DataFrame) -> np.ndarray:
    y, sd = frame["TAXITIME_SEC_mvt"].astype(float), frame["sched_delay"].astype(float)
    return (((y - sd).abs() < FB_TOL) & (y > 0)).values.astype(np.int8)


def fit_head(train: pd.DataFrame, stop: pd.DataFrame, feat: list[str]) -> tuple[lgb.Booster, IsotonicRegression]:
    """Classifier early-stopped on `stop`, and its isotonic map fit on the `stop` scores."""
    ds = {"categorical_feature": CAT_COLS, "params": {"feature_pre_filter": False}}
    dt = lgb.Dataset(train[feat], label=is_fallback(train), **ds)
    dv = lgb.Dataset(stop[feat], label=is_fallback(stop), reference=dt, **ds)
    b = lgb.train(GATE_PARAMS, dt, num_boost_round=5000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        b.predict(stop[feat], num_iteration=b.best_iteration), is_fallback(stop))
    return b, iso


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame()
    train, stop = dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)]
    head, iso = fit_head(train, stop, feat)
    h, _ = holdout_rows()
    saved = pd.read_parquet(PREDICTIONS)
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    if not np.array_equal(saved["y"].values, y):
        raise AssertionError("The saved hold-out predictions are not in the hold-out row order. "
                             "Rerun measure_catboost_pair, then rerun this check.")
    base = 0.5 * saved["v46"].values + 0.5 * saved["cat_a"].values
    p = iso.transform(head.predict(h[feat], num_iteration=head.best_iteration))
    sd = h["sched_delay"].astype(float).values
    ok = ~np.isnan(sd)
    apt = h["ADEP_mvt"].astype(str).values
    report = {"best_iteration": int(head.best_iteration), "fallback_rate_holdout": float(is_fallback(h).mean()),
              "min_gain_mse": MIN_GAIN_MSE, "by_alpha": {}}
    for a in ALPHAS:
        pred = np.where(ok, base + a * p * (np.where(ok, sd, 0) - base), base)
        per = {x: mse_delta(pred[apt == x], base[apt == x], y[apt == x]) for x in sorted(set(apt))}
        report["by_alpha"][str(a)] = {"all": mse_delta(pred, base, y), "by_airport": per,
                                      "gain_where_negative": sum(v for v in per.values() if v < 0)}
    best = min(report["by_alpha"].values(), key=lambda r: r["gain_where_negative"])
    report["build_v72"] = best["gain_where_negative"] <= -MIN_GAIN_MSE
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "mh2.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
