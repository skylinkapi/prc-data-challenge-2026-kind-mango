"""v97: price the base members refit with the OPDI and ADS-B pushback columns on the 2025 hold-out.

`fit <member>` trains one member on months 2 to 6 and 8 to 10 (CatBoost
early-stops on 11 and 12) with the served v89 recipe plus the columns of
`opdi_own_train_v90` and `adsb_clock_train` (350 of 365 days of 2025 carry
ADS-B), and writes its hold-out prediction. Members run as separate
processes. `price` builds the v97 hold-out blend from the five members,
cross-fits the v95 stack on it, and compares blend plus stack with the
v89 blend plus the same stack, outside LIRF, on the hold-out scale.
`fit --tag T --extra F...` writes `holdout_<member>_T.parquet` with more
column files (third-edition K3); `measure_member_swap.py` prices it.
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
from src_v3 import stack_pushback_v90 as sp
from src_v3.build_adsb_clock import OUT_TRAIN as ADSB_TRAIN
from src_v3.build_nm_clocks_v73 import OUT_TRAIN as CLOCKS_TRAIN
from src_v3.build_opdi_own import OUT_TRAIN as OPDI_TRAIN
from src_v3.catboost_base import fit as fit_catboost, load_frame, predict
from src_v3.fit_blend_v77 import blend
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS, V67_ROUND_CAP, holdout_rows
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.tune_lgbm_v76 import OUT as TUNED_V76
from src_v3.tune_stack_v93 import frame as stack_frame

FORWARD_TRAIN = C.ROOT / "models" / "nm_forward_train_v85.parquet"
LGB_EXTRA = (CLOCKS_TRAIN, FORWARD_TRAIN, OPDI_TRAIN, ADSB_TRAIN)   # served LightGBM columns plus pushback clocks
CAT_EXTRA = (CLOCKS_TRAIN, OPDI_TRAIN, ADSB_TRAIN)                  # the served CatBoost has no forward columns (v87)
MEMBERS = ("lgb", "lgb_anchor", "lgb_per_airport", "cat", "cat_anchor")
log = logging.getLogger(__name__)


def out_path(member: str, tag: str = "v97") -> Path:
    return C.MODELS / f"holdout_{member}_{tag}.parquet"


def anchor(frame: pd.DataFrame) -> np.ndarray:
    return frame["mvt_aobt3"].clip(0, C.NM_CLOCK_CLIP_S).values


def lgb_fit(train: pd.DataFrame, feat: list[str]) -> lgb.Booster:
    study = json.loads(TUNED_V76.read_text())
    dt = lgb.Dataset(train[feat], label=train["TAXITIME_SEC_mvt"].values, categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})
    return lgb.train({**study["tuned"], "num_threads": C.NUM_THREADS}, dt, num_boost_round=study["best_iteration"])


def fit_member(member: str, more: tuple = ()) -> np.ndarray:
    """Hold-out prediction of one member, in the row order of `PRED_OUT`; anchored members fall back to the plain one."""
    extra = (*(CAT_EXTRA if member.startswith("cat") else LGB_EXTRA), *more)
    dep, feat = load_frame(extra)
    if member.endswith("anchor"):
        dep = dep[dep["mvt_aobt3"].notna()].copy()
        dep["TAXITIME_SEC_mvt"] -= anchor(dep)
    train = dep[dep["month"].isin(FIT_MONTHS)]
    stop = dep[dep["month"].isin(STOP_MONTHS)]
    del dep
    h, _ = holdout_rows()
    for path in extra:
        h = h.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    if member == "lgb_per_airport":
        apt, raw = h["ADEP_mvt"].astype(str).values, np.full(len(h), np.nan)
        for a in sorted(set(apt)):
            raw[apt == a] = lgb_fit(train[train["ADEP_mvt"].astype(str) == a], feat).predict(h.loc[apt == a, feat])
            log.info("per-airport %s done", a)
    elif member.startswith("lgb"):
        raw = lgb_fit(train, feat).predict(h[feat])
    else:
        model = fit_catboost(train, feat, stop=stop, max_rounds=V67_ROUND_CAP)
        raw = predict(model, h, feat, clip=False)
    if not member.endswith("anchor"):
        return np.clip(raw, 0, None)
    plain = pd.read_parquet(out_path(member.removesuffix("_anchor")))["pred"].values
    return np.where(np.isnan(anchor(h)), plain, np.clip(anchor(h) + raw, 0, None))


def blend_path(per: float) -> Path:
    return C.MODELS / ("blend_v97_holdout.parquet" if per else "blend_v97np_holdout.parquet")


def price(per: float = 0.5) -> dict:
    """v97 blend and v97 blend plus stack against the v89 blend plus stack, outside LIRF.

    `per` is the per-airport share of the plain LightGBM term; 0 drops the per-airport members (v97 as served).
    """
    ref = pd.read_parquet(PRED_OUT)
    m = {k: pd.read_parquet(out_path(k))["pred"].values for k in MEMBERS}
    new = blend({"lgb": (1 - per) * m["lgb"] + per * m["lgb_per_airport"], "lgb_anchor": m["lgb_anchor"],
                 "cat": m["cat"], "cat_anchor": m["cat_anchor"]}, *C.BLEND_V77_SERVED)
    path = blend_path(per)
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "y": ref["y"].values, "base": new}).to_parquet(path)
    old = pd.read_parquet(sp.BLEND_HOLDOUT)
    y = ref["y"].values
    report = {"blend_v97_minus_v89": mse_delta(new, old["base"].values, y)}
    totals = {}
    for name, source in (("v89", sp.BLEND_HOLDOUT), ("v97", path)):
        s, _ = sp.crossfit(stack_frame(C.AUTO_MLP_WEIGHT, False, blend_path=source))
        full = pd.read_parquet(source).dropna(subset=["MVT_ID_mvt"]).set_index("MVT_ID_mvt")
        full.loc[s["MVT_ID_mvt"].values, "base"] = np.fmax(s["stack"].values, sp.stand_bound(s))
        totals[name] = full
    ids = totals["v97"].index
    report["blend_and_stack_v97_minus_v89"] = mse_delta(totals["v97"]["base"].values, totals["v89"].loc[ids, "base"].values,
                                                        totals["v97"]["y"].values)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("fit", "price"))
    ap.add_argument("member", nargs="?", choices=MEMBERS)
    ap.add_argument("--tag", default="v97")
    ap.add_argument("--extra", nargs="+", default=[], help="More column files for the member (K3).")
    args = ap.parse_args()
    if args.mode == "fit":
        ref = pd.read_parquet(PRED_OUT)
        pred = fit_member(args.member, tuple(C.ROOT / x for x in args.extra))
        log.info("%s hold-out RMSE %.2f", args.member, np.sqrt(np.mean((pred - ref["y"].values) ** 2)))
        pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "pred": pred}).to_parquet(out_path(args.member, args.tag))
        return
    report = price()
    log.info("%s", json.dumps(report, indent=1))
    (C.MODELS / "adsb_base_v97.holdout.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
