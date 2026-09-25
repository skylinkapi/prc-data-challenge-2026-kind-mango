"""v67 = v65 + CatBoost blend on the base (WINNING_PLAN L3, MB7).

Outside LIRF the base term becomes `(1 - w) * LightGBM mean + w * CatBoost`.
The difference goes onto the v65 pre-MS output, so the LIRF head, Step A
and ITY340 stay as served. MS1 and MS2 then apply as in v57.
With `--r-norm-weight`, v68 also blends a LIRF CatBoost member into `R_norm`.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src_v3 import config as C
from src_v3.catboost_base import paths, predict
from src_v3.catboost_r_norm import blend_r_norm
from src_v3.predict_v57 import serve_v30, write_ms
from src_v3.predict_v62 import score_members

BASE = "lgbm_r_all_v65"
EXTRA = ("models/plan_nm_taxi_rank_v65.parquet",)
GATE_ISO = "lirf_regime_v64.isotonic.pkl"
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weight", type=float, required=True, choices=C.CATBOOST_BLEND_WEIGHTS)
    ap.add_argument("--r-norm-weight", type=float, choices=C.CATBOOST_BLEND_WEIGHTS,
                    help="Blend the LIRF CatBoost member into R_norm at this weight (v68).")
    ap.add_argument("--catboost-tag", default="v67", help="Version tag of the base CatBoost model.")
    ap.add_argument("--base-model", default=BASE, help="LightGBM base members (v71: lgbm_r_all_v71).")
    ap.add_argument("--extra", nargs="+", default=list(EXTRA),
                    help="Parquets of extra ranking columns keyed on MVT_ID_mvt.")
    ap.add_argument("--pre-ms", default="kind-mango_v67_pre_ms.parquet")
    ap.add_argument("--out", default="kind-mango_v67.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    dump_path = C.MODELS / f"{args.base_model}_rank_features.parquet"
    serve_v30(args.pre_ms, args.base_model, dump_features=str(dump_path), stepa_normal_rnorm=True,
              gate_iso=GATE_ISO, extra_files=tuple(args.extra),
              r_norm_post=None if args.r_norm_weight is None else blend_r_norm(args.r_norm_weight))
    dump = pd.read_parquet(dump_path)
    pre = pd.read_parquet(C.ROOT / "submission" / args.pre_ms)
    lgb_mean = score_members(dump, args.base_model).mean(axis=0)
    is_base = (dump["ADEP_mvt"].astype(str) != "LIRF").values
    served = dump[["MVT_ID_mvt"]].merge(pre, on="MVT_ID_mvt", how="left",
                                        validate="1:1")["TAXITIME_SEC_mvt"].values
    gap = np.abs(lgb_mean - served)[is_base].max()
    if gap > 1e-6:
        raise AssertionError(f"Member scoring does not match the served base: max gap {gap:.3f} s. "
                             "Check the categorical cast, then rerun.")

    model_path, _, meta_path = paths(args.catboost_tag)
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    feat = json.loads(meta_path.read_text())["features"]
    cat = predict(model, dump, feat)
    delta = np.where(is_base, args.weight * (cat - lgb_mean), 0.0)
    log.info("CatBoost minus LightGBM outside LIRF: mean %.1f s, p99 |d| %.0f s",
             (cat - lgb_mean)[is_base].mean(), np.quantile(np.abs(cat - lgb_mean)[is_base], 0.99))
    pre = pre.merge(pd.DataFrame({"MVT_ID_mvt": dump["MVT_ID_mvt"].values, "delta": delta}),
                    on="MVT_ID_mvt", how="left", validate="1:1")
    pre["TAXITIME_SEC_mvt"] += pre.pop("delta").fillna(0)
    write_ms(pre, args.out)


if __name__ == "__main__":
    main()
