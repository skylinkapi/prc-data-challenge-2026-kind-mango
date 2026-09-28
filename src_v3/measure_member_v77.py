"""Hold-out predictions of one clock-era base member, for the blend-weight fit of v77.

Each member trains on the MB3 rows of months 2 to 6 and 8 to 10 with the
v73 columns and scores the hold-out rows of months 1 and 7 outside LIRF,
in the row order of `measure_nm_clocks.py` (its saved clock member is the
plain LightGBM member). LightGBM members train a fixed 2,000 rounds;
`lgb_const` uses constant leaves (no `linear_tree`), which cannot extrapolate;
the `C.CATBOOST_VARIANTS_V80` members are plain CatBoost with other parameters;
`xgb` is XGBoost (`C.XGB_PARAMS`), early-stopped on months 11 and 12;
CatBoost members early-stop on months 11 and 12 up to the v67 round cap.
Anchored members learn `y - clip(MVT - AOBT_3, 0, 7,200)` and serve
`anchor + offset`, or the plain member of the same class where `AOBT_3`
is null.
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
from src_v3.catboost_base import fit, load_frame, predict
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS, V67_ROUND_CAP, holdout_rows
from src_v3.measure_lgbm_control import fit_early_stopped
from src_v3.measure_mf2 import ROUNDS
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.train_v57_base import TUNED

MEMBERS = ("lgb_anchor", "cat_plain", "cat_anchor", "lgb_const", *C.CATBOOST_VARIANTS_V80, "xgb")
log = logging.getLogger(__name__)


def out_path(member: str):
    return C.MODELS / f"holdout_{member}_v77.parquet"


def anchor(frame: pd.DataFrame) -> np.ndarray:
    return frame["mvt_aobt3"].clip(0, C.NM_CLOCK_CLIP_S).values


def fit_xgb(train: pd.DataFrame, stop: pd.DataFrame, h: pd.DataFrame, feat: list[str]) -> tuple:
    """XGBoost member; categorical codes follow the training categories, unseen levels become NaN."""
    import xgboost as xgb  # not a served dependency; only this closed hold-out member reads it

    cats = {c: pd.Categorical(train[c].astype(str)).categories for c in CAT_COLS}

    def matrix(frame: pd.DataFrame, label: bool = True):
        x = frame[feat].copy()
        for c in CAT_COLS:
            x[c] = pd.Categorical(x[c].astype(str), categories=cats[c])
        return xgb.DMatrix(x, label=frame["TAXITIME_SEC_mvt"].values if label else None, enable_categorical=True)

    b = xgb.train(C.XGB_PARAMS, matrix(train), num_boost_round=C.XGB_MAX_ROUNDS,
                  evals=[(matrix(stop), "stop")], early_stopping_rounds=C.XGB_EARLY_STOP, verbose_eval=250)
    raw = b.predict(matrix(h, label=False), iteration_range=(0, b.best_iteration + 1))
    return np.clip(raw, 0, None), None, b.best_iteration + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("member", choices=MEMBERS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dep, feat = load_frame((CLOCKS_TRAIN,))
    is_anchor = args.member.endswith("anchor")
    if is_anchor:
        dep = dep[dep["mvt_aobt3"].notna()].copy()
        dep["TAXITIME_SEC_mvt"] = dep["TAXITIME_SEC_mvt"] - anchor(dep)
    train, stop = dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)]
    del dep
    h, _ = holdout_rows()
    h = h.merge(pd.read_parquet(CLOCKS_TRAIN), on="MVT_ID_mvt", how="left", validate="m:1")
    ref = pd.read_parquet(PRED_OUT)
    if not (ref["y"].values == h["TAXITIME_SEC_mvt"].values).all():
        raise AssertionError("The saved clock predictions are not in hold-out row order. "
                             "Rerun measure_nm_clocks, then rerun.")
    if args.member == "xgb":
        raw, plain, rounds = fit_xgb(train, stop, h, feat)
    elif args.member == "lgb_const":
        params = {**json.loads(TUNED.read_text())["tuned"], "linear_tree": False, "verbosity": -1,
                  "num_threads": C.NUM_THREADS, "deterministic": C.DETERMINISTIC, "seed": C.GLOBAL_SEED,
                  "bagging_seed": C.GLOBAL_SEED, "feature_fraction_seed": C.GLOBAL_SEED}
        dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values, categorical_feature=CAT_COLS,
                         params={"feature_pre_filter": False})
        raw, plain, rounds = lgb.train(params, dt, num_boost_round=ROUNDS).predict(h[feat]), None, ROUNDS
        raw = np.clip(raw, 0, None)
    elif args.member.startswith("lgb"):
        raw = fit_early_stopped(train, None, feat, rounds=ROUNDS).predict(h[feat])
        plain, rounds = ref["clocks"].values, ROUNDS
    else:
        model = fit(train, feat, stop=stop, max_rounds=V67_ROUND_CAP,
                    params=C.CATBOOST_VARIANTS_V80.get(args.member))
        raw, rounds = predict(model, h, feat, clip=not is_anchor), int(model.get_best_iteration()) + 1
        plain = raw if not is_anchor else pd.read_parquet(out_path("cat_plain"))["pred"].values
    pred = np.where(np.isnan(anchor(h)), plain, np.clip(anchor(h) + raw, 0, None)) if is_anchor else raw
    log.info("%s: %d rounds, hold-out RMSE %.2f", args.member, rounds,
             np.sqrt(np.mean((pred - ref["y"].values) ** 2)))
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "month": h["month"].values,
                  "pred": pred}).to_parquet(out_path(args.member))


if __name__ == "__main__":
    main()
