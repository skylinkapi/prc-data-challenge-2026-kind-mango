"""v67 = v65 + CatBoost blend on the base (WINNING_PLAN L3, MB7).

Outside LIRF the base term becomes `(1 - w) * LightGBM mean + w * CatBoost`.
The difference goes onto the v65 pre-MS output, so the LIRF head, Step A
and ITY340 stay as served. MS1 and MS2 then apply as in v57.
With `--r-norm-weight`, v68 also blends a LIRF CatBoost member into `R_norm`.
With `--ms3`, the LightGBM term is the member median where the members
disagree by more than 3,600 s (MS3, v73).
With `--anchor-model`, the LightGBM term blends in members that learned
`y - clip(MVT - AOBT_3, 0, 7,200)` (WINNING_PLAN L5, v74); rows without
`AOBT_3` keep the plain LightGBM term. `--catboost-anchor-tag` does the
same for the CatBoost term (v75). `--per-airport-tag` mixes one member per
airport into the plain and the anchored LightGBM terms (v89).
"""
from __future__ import annotations

import argparse
import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src_v3 import config as C
from src_v3.catboost_base import paths, predict
from src_v3.catboost_r_norm import blend_r_norm
from src_v3.postprocess import apply_ms3
from src_v3.predict_v57 import serve_v30, write_ms
from src_v3.predict_v62 import CAT_COLS, load_training_categories, score_members

BASE = "lgbm_r_all_v65"
EXTRA = ("models/plan_nm_taxi_rank_v65.parquet",)
GATE_ISO = "lirf_regime_v64.isotonic.pkl"
log = logging.getLogger(__name__)


def blend_anchor(dump: pd.DataFrame, term: np.ndarray, offset: np.ndarray, weight: float) -> np.ndarray:
    """`(1 - weight) * term + weight * (anchor + offset)`; rows without an anchor keep `term`."""
    anchor = dump["mvt_aobt3"].clip(0, C.NM_CLOCK_CLIP_S).values
    anchored = np.where(np.isnan(anchor), term, np.clip(anchor + offset, 0, None))
    base = (dump["ADEP_mvt"].astype(str) != "LIRF").values
    log.info("anchored minus plain term outside LIRF: mean %.1f s, p99 |d| %.0f s",
             (anchored - term)[base].mean(), np.quantile(np.abs(anchored - term)[base], 0.99))
    return (1 - weight) * term + weight * anchored


def score_per_airport(dump: pd.DataFrame, tag: str, clip: bool) -> np.ndarray:
    """Prediction of each airport's member on its own rows; LIRF and airports without a member stay NaN."""
    feat = (C.ROOT / "models" / f"lgbm_pa_{tag}.features.txt").read_text().splitlines()
    cats = load_training_categories()
    x = dump[feat].copy()
    for c in CAT_COLS:
        x[c] = pd.Categorical(x[c], categories=cats[c])
    adep = dump["ADEP_mvt"].astype(str).values
    out = np.full(len(dump), np.nan)
    for a in np.unique(adep):
        path = C.ROOT / "models" / f"lgbm_pa_{tag}_{a}.txt"
        if path.exists():
            out[adep == a] = lgb.Booster(model_file=str(path)).predict(x[adep == a])
    return np.clip(out, 0, None) if clip else out


def score_seeds(dump: pd.DataFrame, tags: list[str], clip: bool) -> np.ndarray:
    """Mean of the per-airport members over the seed tags (B2); one tag serves v89 unchanged."""
    return np.mean([score_per_airport(dump, t, clip) for t in tags], axis=0)


def mix(term: np.ndarray, per: np.ndarray, weight: float) -> np.ndarray:
    return np.where(np.isnan(per), term, (1 - weight) * term + weight * per)


def load_catboost(tag: str) -> tuple[CatBoostRegressor, list[str]]:
    model_path, _, meta_path = paths(tag)
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model, json.loads(meta_path.read_text())["features"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weight", type=float, required=True, choices=C.CATBOOST_BLEND_WEIGHTS)
    ap.add_argument("--r-norm-weight", type=float, choices=C.CATBOOST_BLEND_WEIGHTS,
                    help="Blend the LIRF CatBoost member into R_norm at this weight (v68).")
    ap.add_argument("--catboost-tag", default="v67", help="Version tag of the base CatBoost model.")
    ap.add_argument("--base-model", default=BASE, help="LightGBM base members (v71: lgbm_r_all_v71).")
    ap.add_argument("--extra", nargs="+", default=list(EXTRA),
                    help="Parquets of extra ranking columns keyed on MVT_ID_mvt.")
    ap.add_argument("--ms3", action="store_true",
                    help="Serve the LightGBM member median where the members disagree by more than 3,600 s.")
    ap.add_argument("--anchor-model", help="LightGBM members trained on the AOBT_3 residual (v74: lgbm_r_all_v74a).")
    ap.add_argument("--per-airport-tag", nargs="+",
                    help="Per-airport LightGBM members, one tag per seed (v89: v89; v92: v89 v89s43 v89s44).")
    ap.add_argument("--per-airport-weight", type=float, default=0.5)
    ap.add_argument("--catboost-anchor-tag", help="CatBoost trained on the AOBT_3 residual (v75: v75a).")
    ap.add_argument("--anchor-weight", type=float, default=0.5,
                    help="Weight of the anchored model inside the LightGBM term and the CatBoost term.")
    ap.add_argument("--gate-booster", default="lgbm_p_fb_lirf_v56.txt", help="LIRF gate booster (v83: lgbm_p_fb_lirf_v83.txt).")
    ap.add_argument("--gate-iso", default=GATE_ISO, help="Isotonic map of the gate (v83: lirf_regime_v83.isotonic.pkl).")
    ap.add_argument("--gate-features", default="lirf_regime_v23.features.txt",
                    help="Feature list of the gate (v83: lirf_regime_v83.features.txt).")
    ap.add_argument("--gate-catboost", nargs=2, metavar=("MODEL", "ISO"),
                    help="CatBoost gate member and its isotonic map, averaged with the gate booster (N7: g3c).")
    ap.add_argument("--r-norm-tag", default="v55", help="LIRF R_norm members (K2: v99r, with the ADS-B and OPDI columns).")
    ap.add_argument("--dump-terms", help="Write the per-member ranking terms outside LIRF to this models/v3 parquet (N3).")
    ap.add_argument("--pre-ms", default="kind-mango_v67_pre_ms.parquet")
    ap.add_argument("--out", default="kind-mango_v67.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    dump_path = C.MODELS / f"{args.base_model}_rank_features.parquet"
    serve_v30(args.pre_ms, args.base_model, dump_features=str(dump_path), stepa_normal_rnorm=True,
              gate_iso=args.gate_iso, extra_files=tuple(args.extra),
              r_norm_post=None if args.r_norm_weight is None else blend_r_norm(args.r_norm_weight),
              gate_booster=args.gate_booster, gate_features=args.gate_features, r_norm_tag=args.r_norm_tag,
              gate_catboost=args.gate_catboost)
    dump = pd.read_parquet(dump_path)
    pre = pd.read_parquet(C.ROOT / "submission" / args.pre_ms)
    members = score_members(dump, args.base_model)
    lgb_mean = members.mean(axis=0)
    adep = dump["ADEP_mvt"].astype(str).values
    is_base = adep != "LIRF"
    lgb_term = lgb_mean
    if args.ms3:
        lgb_term, moved = apply_ms3(members, adep)
        log.info("MS3 moved %s", moved)
    terms = {"lgb": lgb_term}
    if args.per_airport_tag:
        terms["lgb_per_airport"] = score_seeds(dump, args.per_airport_tag, clip=True)
        lgb_term = mix(lgb_term, terms["lgb_per_airport"], args.per_airport_weight)
    if args.anchor_model:
        offsets = score_members(dump, args.anchor_model, clip=False)
        offset, moved = apply_ms3(offsets, adep) if args.ms3 else (offsets.mean(axis=0), {})
        log.info("anchored members: MS3 moved %s", moved)
        terms["lgb_anchor"] = blend_anchor(dump, terms["lgb"], offset, 1.0)
        if args.per_airport_tag:
            offset = mix(offset, score_seeds(dump, [f"{t}a" for t in args.per_airport_tag], clip=False),
                         args.per_airport_weight)
        lgb_term = blend_anchor(dump, lgb_term, offset, args.anchor_weight)
    served = dump[["MVT_ID_mvt"]].merge(pre, on="MVT_ID_mvt", how="left",
                                        validate="1:1")["TAXITIME_SEC_mvt"].values
    gap = np.abs(lgb_mean - served)[is_base].max()
    if gap > 1e-6:
        raise AssertionError(f"Member scoring does not match the served base: max gap {gap:.3f} s. "
                             "Check the categorical cast, then rerun.")

    model, feat = load_catboost(args.catboost_tag)
    cat = terms["cat"] = predict(model, dump, feat)
    if args.catboost_anchor_tag:
        model, feat = load_catboost(args.catboost_anchor_tag)
        cat_offset = predict(model, dump, feat, clip=False)
        terms["cat_anchor"] = blend_anchor(dump, terms["cat"], cat_offset, 1.0)
        cat = blend_anchor(dump, cat, cat_offset, args.anchor_weight)
    delta = np.where(is_base, args.weight * (cat - lgb_mean) + (1 - args.weight) * (lgb_term - lgb_mean), 0.0)
    log.info("CatBoost minus LightGBM outside LIRF: mean %.1f s, p99 |d| %.0f s",
             (cat - lgb_mean)[is_base].mean(), np.quantile(np.abs(cat - lgb_mean)[is_base], 0.99))
    pre = pre.merge(pd.DataFrame({"MVT_ID_mvt": dump["MVT_ID_mvt"].values, "delta": delta}),
                    on="MVT_ID_mvt", how="left", validate="1:1")
    pre["TAXITIME_SEC_mvt"] += pre.pop("delta").fillna(0)
    if args.dump_terms:
        terms["served"] = args.weight * cat + (1 - args.weight) * lgb_term
        pd.DataFrame({"MVT_ID_mvt": dump["MVT_ID_mvt"].values, **terms})[is_base].to_parquet(C.MODELS / args.dump_terms)
    write_ms(pre, args.out)


if __name__ == "__main__":
    main()
