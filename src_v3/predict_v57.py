"""v57 = v56 recipe with the plan_taxi_res route medians refit on all 12
months (MB8 for route medians) and the base retrained on the updated
frame. Stacks with v55 R_norm and v56 gate.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict_v30 import main as predict_v30_main
from predict_v45 import RANK_FRAME as V2_RANK
from train_r_all_v40 import TEMPO_COLS
from train_r_all_v44 import EXTRA_COLS as EOBT_P2575
from train_r_all_v45plan import EXTRA_COLS as PLAN_COLS

from src_v3 import config as C
from src_v3.postprocess import apply_ms1, apply_ms2, assert_ms4
from src_v3.predict_v51 import _load_train_min
from src_v3.support import build_support

R_NORM_FILES = [f"lgbm_r_norm_lirf_v55_s{s}.txt" for s in (42, 43, 44, 45, 46)]
P_FB_BOOSTER = "lgbm_p_fb_lirf_v56.txt"
log = logging.getLogger(__name__)


def serve_v30(pre_ms: str, base_model: str, dump_features: str | None = None,
              stepa_normal_rnorm: bool = False,
              gate_iso: str = "lirf_regime_v56.isotonic.pkl",
              extra_files: tuple[str, ...] = (), r_norm_post=None) -> None:
    """Write the pre-MS file of the v57 stack with the given base members."""
    tempo = pd.read_parquet(V2_RANK, columns=["MVT_ID_mvt", *TEMPO_COLS])
    p2575 = pd.read_parquet(C.ROOT / "models" / "tempo_p2575_rank.parquet",
                            columns=["MVT_ID_mvt", *EOBT_P2575])
    plan = pd.read_parquet(C.ROOT / "models" / "plan_taxi_res_rank_v57.parquet",
                           columns=["MVT_ID_mvt", *PLAN_COLS])
    extra = tempo.merge(p2575, on="MVT_ID_mvt", how="left") \
                 .merge(plan, on="MVT_ID_mvt", how="left")
    for path in extra_files:
        extra = extra.merge(pd.read_parquet(C.ROOT / path), on="MVT_ID_mvt",
                            how="left", validate="1:1")
    predict_v30_main(
        pre_ms,
        r_norm_files=R_NORM_FILES,
        base_model=base_model,
        extra_columns=extra,
        r_norm_features="lirf_regime_v41.features.txt",
        r_norm_clip=None,
        p_fb_members=[(P_FB_BOOSTER, gate_iso)],
        p_fb_features="lirf_regime_v23.features.txt",
        dump_features=dump_features,
        stepa_normal_rnorm=stepa_normal_rnorm,
        r_norm_post=r_norm_post,
    )


def write_ms(pre: pd.DataFrame, out: str) -> None:
    """Apply MS1 and MS2 to a pre-MS frame, check MS4 and write the submission."""
    v2 = pd.read_parquet(V2_RANK, columns=["MVT_ID_mvt", "ADEP_mvt",
                                            "flt_null", "mvt_eobt1"])
    template = pd.read_parquet(C.SUB_TMPL, columns=["MVT_ID_mvt"])
    m = template.merge(pre.rename(columns={"TAXITIME_SEC_mvt": "y"}),
                       on="MVT_ID_mvt", how="left", validate="1:1") \
                .merge(v2, on="MVT_ID_mvt", how="left", validate="1:1")
    support = build_support(_load_train_min())
    adep = m["ADEP_mvt"].astype(str).values
    flt_null = m["flt_null"].fillna(1).astype(int).values
    mvt_eobt1 = m["mvt_eobt1"].fillna(0).astype(float).values
    pred, _ = apply_ms1(m["y"].values.astype(float), adep, flt_null, mvt_eobt1, support)
    pred, _ = apply_ms2(pred, adep, support)
    assert_ms4(pred, template["MVT_ID_mvt"].values, m["MVT_ID_mvt"].values)
    pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].values,
                  "TAXITIME_SEC_mvt": pred.astype(np.float64)}
                 ).to_parquet(C.ROOT / "submission" / out)
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre-ms", default="kind-mango_v57_pre_ms.parquet")
    ap.add_argument("--out", default="kind-mango_v57.parquet")
    ap.add_argument("--base-model", default="lgbm_r_all_v57")
    ap.add_argument("--stepa-normal-rnorm", action="store_true",
                    help="Step A normal term reads R_norm, not the mixture (T1, L12).")
    ap.add_argument("--gate-iso", default="lirf_regime_v56.isotonic.pkl",
                    help="Isotonic map for the v56 gate booster.")
    ap.add_argument("--extra", action="append", default=[],
                    help="Parquet of extra ranking columns keyed on MVT_ID_mvt, relative to the repo root.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    serve_v30(args.pre_ms, args.base_model, stepa_normal_rnorm=args.stepa_normal_rnorm,
              gate_iso=args.gate_iso, extra_files=tuple(args.extra))
    write_ms(pd.read_parquet(C.ROOT / "submission" / args.pre_ms), args.out)


if __name__ == "__main__":
    main()
