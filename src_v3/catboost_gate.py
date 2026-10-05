"""Third-edition N7: a CatBoost member in the LIRF gate, next to the LightGBM g3 gate.

The served gate is the mean of calibrated members (`predict_v30.predict_p_fb`). This module adds a
CatBoost classifier on the 176 g3 columns. Its ordered target statistics and category combinations read
the operator, stand, type and destination columns in a way the LightGBM gate does not. `price` sets the
rounds with an early stop on months 11 and 12 (fit months 2 to 6 and 8 to 10). It then scores every month
out of fold for both classes with the six month-pair folds of the v64 protocol, calibrates each with a
nested isotonic map, and prices the served mixture `p sd + (1 - p) R_norm` on the LIRF rows of months 1
and 7 outside Step A for each CatBoost share. `fit` trains on all 12 months and fits the isotonic map on
the out-of-fold CatBoost scores of `price`.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS
from train_r_all_v40 import TEMPO_COLS, V2_FRAME

from src_v3 import config as C
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS
from src_v3.measure_gate_clocks import nested_map, r_norm_v41
from src_v3.measure_tail_rules import HOLD, mse_delta
from src_v3.train_p_fb_v56 import load_gate_frame
from src_v3.train_p_fb_v64 import EPS, SD_BINS, score_oof

G3_EXTRA = C.ROOT / "models" / "arr_art_op_link_train_g3.parquet"
G3_FEATURES = C.ROOT / "models" / "lirf_regime_g3.features.txt"
G3_BOOSTER = C.ROOT / "models" / "lgbm_p_fb_lirf_g3.txt"
OOF = C.MODELS / "catboost_gate_oof.parquet"
REPORT = C.MODELS / "catboost_gate.holdout.json"
log = logging.getLogger(__name__)


def cat_strings(x: pd.DataFrame) -> pd.DataFrame:
    """Categorical columns as strings; a missing value becomes 'nan' for object and category dtypes alike."""
    x = x.copy()
    for c in CAT_COLS:
        if c in x:
            x[c] = x[c].astype(object).where(x[c].notna(), "nan").astype(str)
    return x


def pool(frame: pd.DataFrame, feat: list[str], label: np.ndarray | None = None) -> Pool:
    return Pool(cat_strings(frame[feat]), label=label, cat_features=[c for c in feat if c in CAT_COLS])


def fit(frame: pd.DataFrame, feat: list[str], iterations: int | None = None,
        stop: pd.DataFrame | None = None) -> CatBoostClassifier:
    """Early stop on `stop` when given; otherwise a fixed `iterations`."""
    params = dict(C.CATBOOST_GATE_PARAMS)
    if iterations is not None:
        params.update(iterations=iterations, od_type=None, od_wait=None)
    m = CatBoostClassifier(**{k: v for k, v in params.items() if v is not None})
    m.fit(pool(frame, feat, frame["is_fb"].values),
          eval_set=None if stop is None else pool(stop, feat, stop["is_fb"].values), use_best_model=stop is not None)
    return m


def score(model_file: str, frame: pd.DataFrame, feat: list[str]) -> np.ndarray:
    """Raw fallback probability of a saved CatBoost gate member."""
    m = CatBoostClassifier()
    m.load_model(model_file)
    return m.predict_proba(pool(frame, feat))[:, 1]


def gate_frame() -> tuple[pd.DataFrame, list[str]]:
    dep, _ = load_gate_frame()
    dep = dep.merge(pd.read_parquet(G3_EXTRA), on="MVT_ID_mvt", how="left", validate="m:1")
    return dep, [x for x in G3_FEATURES.read_text().splitlines() if x]


def score_oof_cat(dep: pd.DataFrame, feat: list[str], rounds: int) -> np.ndarray:
    oof = np.full(len(dep), np.nan)
    for hold in C.FOLDS:
        h = dep["month"].isin(hold).values
        oof[h] = fit(dep[~h], feat, iterations=rounds).predict_proba(pool(dep[h], feat))[:, 1]
        log.info("CatBoost fold %s scored %d rows", hold, h.sum())
    return oof


def mixture(dep: pd.DataFrame, arms: dict) -> dict:
    """MSE change of the served mixture of each arm against the LightGBM gate, all and per hold-out month."""
    served = (dep["month"].isin(HOLD) & (dep["TAXITIME_SEC_mvt"] > 0)
              & ~((dep["flt_null"] == 1) & (dep["sched_delay"] > C.STEP_A_SD_THRESHOLD))).values
    h = dep[served].merge(pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS]), on="MVT_ID_mvt",
                          how="left", validate="m:1")
    r, t = r_norm_v41(h), h["TAXITIME_SEC_mvt"].values.astype(float)
    sd, month = h["sched_delay"].values.astype(float), h["month"].values
    mix = {arm: q[served] * sd + (1 - q[served]) * r for arm, q in arms.items()}
    return {arm: {"all": mse_delta(m, mix["lgb"], t),
                  **{str(k): mse_delta(m[month == k], mix["lgb"][month == k], t[month == k]) for k in HOLD}}
            for arm, m in mix.items() if arm != "lgb"}


def price() -> None:
    dep, feat = gate_frame()
    rounds = fit(dep[dep["month"].isin(FIT_MONTHS)], feat, stop=dep[dep["month"].isin(STOP_MONTHS)]).get_best_iteration() + 1
    log.info("CatBoost gate: %d rounds at the early stop", rounds)
    y, month, sd = dep["is_fb"].values, dep["month"], dep["sched_delay"].astype(float).values
    oof = {"lgb": score_oof(dep, feat, lgb.Booster(model_file=str(G3_BOOSTER)).num_trees()),
           "cat": score_oof_cat(dep, feat, rounds)}
    pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].values, **oof}).to_parquet(OOF)
    cal = {k: nested_map(v, y, month) for k, v in oof.items()}
    arms = {"lgb": cal["lgb"], **{str(w): (1 - w) * cal["lgb"] + w * cal["cat"] for w in C.CATBOOST_GATE_SHARES}}
    report = {"rounds": rounds, "log_loss": {}}
    for label, lo, hi in zip(["sd<=3600", "3600<sd<=14400", "sd>14400"], SD_BINS, SD_BINS[1:]):
        m = (sd > lo) & (sd <= hi)
        report["log_loss"][label] = {arm: float(log_loss(y[m], np.clip(q[m], EPS, 1 - EPS), labels=[0, 1]))
                                     for arm, q in arms.items()}
    report["corr_oof"] = float(np.corrcoef(cal["lgb"], cal["cat"])[0, 1])
    report["mixture_minus_lgb"] = mixture(dep, arms)
    log.info("%s", json.dumps(report, indent=1))
    REPORT.write_text(json.dumps(report, indent=1))


def fit_full(tag: str) -> None:
    """Full-year member and its isotonic map from the out-of-fold scores of `price`."""
    dep, feat = gate_frame()
    oof = pd.read_parquet(OOF)
    if not np.array_equal(oof["MVT_ID_mvt"].values, dep["MVT_ID_mvt"].values):
        raise AssertionError(f"{OOF.name} is not in the row order of the gate frame. Rerun `catboost_gate price`.")
    fit(dep, feat, iterations=json.loads(REPORT.read_text())["rounds"]).save_model(
        str(C.ROOT / "models" / f"catboost_p_fb_lirf_{tag}.cbm"))
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof["cat"].values, dep["is_fb"].values)
    with open(C.ROOT / "models" / f"lirf_regime_{tag}.isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    log.info("wrote the %s CatBoost gate member", tag)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("price", "fit"))
    ap.add_argument("--tag", default="g3c", help="fit: tag of the member files.")
    args = ap.parse_args()
    price() if args.mode == "price" else fit_full(args.tag)


if __name__ == "__main__":
    main()
