"""Third-edition N3: non-negative blend weights of the six base members per coverage class.

The v105 base outside LIRF has fixed shares: 0.8 times the v97 tree blend plus 0.2 times the MLP. `price`
fits non-negative least-squares weights of the six hold-out members (plain, per-airport and anchored
LightGBM, plain and anchored CatBoost, MLP) per coverage class of the error budget, on rows with
`0 < y <= Y_CLEAN_MAX`. Each hold-out month takes the weights fitted on the other month. The price runs end
to end: the cross-fitted base, the v95 stack cross-fitted January against July, and the A5 bound, against
the v105 hold-out path. `serve` writes the served base: the v104 base outside LIRF moves by the difference
between the weighted members and the v105 shares, both scored on the ranking terms of
`predict_v67 --dump-terms`; the stack then runs with `--mlp-weight 0`.
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src_v3 import config as C
from src_v3.measure_adsb_base_v97 import out_path
from src_v3.measure_budget_v97 import coverage, rank_coverage
from src_v3.measure_tail_rules import mse_delta
from src_v3.mlp_member import COLUMN_SETS
from src_v3.mlp_member import out_paths as mlp_paths
from src_v3.stack_pushback_v90 import crossfit, stand_bound
from src_v3.tune_stack_v93 import frame as stack_frame

TREES = ("lgb", "lgb_per_airport", "lgb_anchor", "cat", "cat_anchor")
MEMBERS = (*TREES, "mlp")
BLEND = COLUMN_SETS["v97"][1]
ADSB_V97 = C.ADSB_DIR / "v97" / "adsb_clock_train.parquet"   # the table the v105 stack read
REPORT = C.MODELS / "blend_nnls.holdout.json"
HOLDOUT_OUT = C.MODELS / "blend_nnls_holdout.parquet"
log = logging.getLogger(__name__)


def holdout_members() -> np.ndarray:
    """Hold-out predictions of the six members, one column each, in the row order of the v97 blend."""
    cols = [pd.read_parquet(out_path(k))["pred"].values for k in TREES]
    return np.column_stack([*cols, pd.read_parquet(mlp_paths("v97")[0])["pred"].values])


def stacked(base: np.ndarray) -> pd.DataFrame:
    """Hold-out path with `base` outside LIRF (MLP already inside): cross-fitted v95 stack and A5 bound."""
    rows = pd.read_parquet(BLEND).assign(base=base)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blend.parquet"
        rows.to_parquet(path)
        f = stack_frame(0.0, False, adsb_train=ADSB_V97, blend_path=path)
    s, _ = crossfit(f)
    f = f.set_index("MVT_ID_mvt")
    f.loc[s["MVT_ID_mvt"].values, "base"] = np.fmax(s["stack"].values, stand_bound(s))
    return f


def fit_weights(m: np.ndarray, y: np.ndarray, cls: np.ndarray, rows: np.ndarray) -> dict:
    """NNLS weights per coverage class on `rows` with a normal label."""
    fit = rows & (y > 0) & (y <= C.Y_CLEAN_MAX)
    return {c: nnls(m[fit & (cls == c)], y[fit & (cls == c)])[0] for c in np.unique(cls)}


def apply(m: np.ndarray, cls: np.ndarray, w: dict) -> np.ndarray:
    out = np.empty(len(m))
    for c, wc in w.items():
        out[cls == c] = m[cls == c] @ wc
    return out


def price() -> None:
    ref = pd.read_parquet(BLEND)
    m, y = holdout_members(), ref["y"].values
    v105 = 0.8 * ref["base"].values + C.MLP_BLEND_W * m[:, -1]
    f = stack_frame(0.0, False, adsb_train=ADSB_V97, blend_path=BLEND)
    keyed = f.set_index("MVT_ID_mvt").reindex(ref["MVT_ID_mvt"])
    cls = np.where(ref["MVT_ID_mvt"].isna(), "none", coverage(keyed.reset_index()))
    jan = (keyed["MVT_TIME_UTC_mvt"].dt.month == 1).values
    arms = {}
    for name, c in (("per_class", cls), ("global", np.full(len(cls), "all"))):
        w = {"1": fit_weights(m, y, c, ~jan), "7": fit_weights(m, y, c, jan)}
        cross = np.where(jan, apply(m, c, w["1"]), apply(m, c, w["7"]))
        arms[name] = (cross, {mo: {k: v.round(3).tolist() for k, v in wm.items()} for mo, wm in w.items()})
    old = stacked(v105)
    yy, month, cov = old["y"].values, old["MVT_TIME_UTC_mvt"].dt.month.values, coverage(old.reset_index())
    report = {"members": list(MEMBERS), "v105_shares": [0.1, 0.1, 0.2, 0.2, 0.2, 0.2], "arms": {}}
    for name, (cross, w) in arms.items():
        new = stacked(cross).loc[old.index, "base"].values
        report["arms"][name] = {
            "weights_by_fit_month": w, "base_only": mse_delta(cross, v105, y),
            "all": mse_delta(new, old["base"].values, yy),
            **{f"month_{k}": mse_delta(new[month == k], old["base"].values[month == k], yy[month == k]) for k in (1, 7)},
            "by_coverage": {c: mse_delta(new[cov == c], old["base"].values[cov == c], yy[cov == c]) for c in np.unique(cov)}}
    report["serve_weights"] = {k: v.round(4).tolist() for k, v in fit_weights(m, y, cls, np.ones(len(y), bool)).items()}
    log.info("%s", json.dumps(report, indent=1))
    REPORT.write_text(json.dumps(report, indent=1))


def serve(terms: str, base: str, out: str) -> None:
    """Hold-out blend for the stack and served base, both with the per-class weights fitted on January and July."""
    w = {c: np.array(v) for c, v in json.loads(REPORT.read_text())["serve_weights"].items()}
    ref = pd.read_parquet(BLEND)
    keyed = stack_frame(0.0, False, adsb_train=ADSB_V97, blend_path=BLEND).set_index("MVT_ID_mvt").reindex(ref["MVT_ID_mvt"])
    cls = np.where(ref["MVT_ID_mvt"].isna(), "none", coverage(keyed.reset_index()))
    ref.assign(base=apply(holdout_members(), cls, w)).to_parquet(HOLDOUT_OUT)
    t = (pd.read_parquet(C.MODELS / terms).merge(pd.read_parquet(mlp_paths("v97")[1]).rename(columns={"pred": "mlp"}),
                                                 on="MVT_ID_mvt", how="left", validate="1:1")
         .merge(rank_coverage()[["MVT_ID_mvt", "cov"]], on="MVT_ID_mvt", how="left", validate="1:1"))
    if t[[*MEMBERS, "cov"]].isna().any().any():
        raise ValueError(f"{terms} misses a member or a coverage class on rows outside LIRF. Rerun predict_v67 --dump-terms.")
    served = pd.read_parquet(C.ROOT / "submission" / base)
    t = t.merge(served.rename(columns={"TAXITIME_SEC_mvt": "b"}), on="MVT_ID_mvt", how="left", validate="1:1")
    log.info("served term minus base file outside LIRF: p99 |d| %.1f s (MS1 and MS2)", np.quantile(np.abs(t["served"] - t["b"]), 0.99))
    # v105 serves 0.8 b + 0.2 MLP outside LIRF before the stack; the weighted members replace 0.8 served + 0.2 MLP.
    t["new"] = 0.8 * t["b"] + apply(t[list(MEMBERS)].to_numpy(), t["cov"].values, w) - 0.8 * t["served"]
    served = served.merge(t[["MVT_ID_mvt", "new"]], on="MVT_ID_mvt", how="left", validate="1:1")
    served["TAXITIME_SEC_mvt"] = served.pop("new").fillna(served["TAXITIME_SEC_mvt"])
    served.to_parquet(C.ROOT / "submission" / out)
    log.info("wrote %s and %s; %d rows outside LIRF", HOLDOUT_OUT.name, out, len(t))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("price", "serve"))
    ap.add_argument("--terms", default="rank_terms_v104.parquet", help="serve: member terms of predict_v67 --dump-terms.")
    ap.add_argument("--base", default="kind-mango_v104_base.parquet", help="serve: base file that v105 stacks on.")
    ap.add_argument("--out", default="kind-mango_v110_base.parquet")
    args = ap.parse_args()
    price() if args.mode == "price" else serve(args.terms, args.base, args.out)


if __name__ == "__main__":
    main()
