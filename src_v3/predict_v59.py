"""v59 = v57 recipe with 4 arrival-drift features on top of v57.

Base = lgbm_r_all_v59 (v57 recipe + drift columns in training).
Everything else stays at v57: v55 R_norm, v56 gate, v57 plan medians,
MS1+MS2 post-processing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre-ms", default="kind-mango_v59_pre_ms.parquet")
    ap.add_argument("--out", default="kind-mango_v59.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from predict_v30 import main as predict_v30_main
    from predict_v45 import RANK_FRAME as V2_RANK
    from train_r_all_v40 import TEMPO_COLS
    from train_r_all_v44 import EXTRA_COLS as EOBT_P2575
    from train_r_all_v45plan import EXTRA_COLS as PLAN_COLS

    import numpy as np
    import pandas as pd
    from src_v3 import config as C
    from src_v3.postprocess import apply_ms1, apply_ms2, assert_ms4
    from src_v3.support import build_support
    from src_v3.predict_v51 import _load_train_min

    log = logging.getLogger(__name__)
    DRIFT_COLS = ["arr_txi_1d_med", "arr_txi_7d_med",
                  "arr_txi_1d_drift", "arr_txi_7d_drift"]

    tempo = pd.read_parquet(V2_RANK, columns=["MVT_ID_mvt", *TEMPO_COLS])
    p2575 = pd.read_parquet(C.ROOT / "models" / "tempo_p2575_rank.parquet",
                            columns=["MVT_ID_mvt", *EOBT_P2575])
    plan = pd.read_parquet(C.ROOT / "models" / "plan_taxi_res_rank_v57.parquet",
                           columns=["MVT_ID_mvt", *PLAN_COLS])
    drift = pd.read_parquet(C.ROOT / "models" / "arr_drift_rank.parquet",
                            columns=["MVT_ID_mvt", *DRIFT_COLS])
    extra = tempo.merge(p2575, on="MVT_ID_mvt", how="left") \
                 .merge(plan, on="MVT_ID_mvt", how="left") \
                 .merge(drift, on="MVT_ID_mvt", how="left")

    R_NORM_FILES = [f"lgbm_r_norm_lirf_v55_s{s}.txt" for s in (42, 43, 44, 45, 46)]
    P_FB_MEMBERS = [("lgbm_p_fb_lirf_v56.txt", "lirf_regime_v56.isotonic.pkl")]

    log.info("=== v59: v57 base + arrival drift features ===")
    predict_v30_main(
        args.pre_ms,
        r_norm_files=R_NORM_FILES,
        base_model="lgbm_r_all_v59",
        extra_columns=extra,
        r_norm_features="lirf_regime_v41.features.txt",
        r_norm_clip=None,
        p_fb_members=P_FB_MEMBERS,
        p_fb_features="lirf_regime_v23.features.txt",
    )

    log.info("=== apply MS1 + MS2 ===")
    pre = pd.read_parquet(C.ROOT / "submission" / args.pre_ms)
    v2 = pd.read_parquet(V2_RANK, columns=["MVT_ID_mvt", "ADEP_mvt",
                                            "flt_null", "mvt_eobt1"])
    template = pd.read_parquet(C.SUB_TMPL, columns=["MVT_ID_mvt"])
    m = template.merge(pre.rename(columns={"TAXITIME_SEC_mvt": "y"}),
                       on="MVT_ID_mvt", how="left", validate="1:1") \
                .merge(v2, on="MVT_ID_mvt", how="left", validate="1:1")
    train = _load_train_min()
    support = build_support(train)
    pred_vec = m["y"].values.astype(float)
    adep = m["ADEP_mvt"].astype(str).values
    flt_null = m["flt_null"].fillna(1).astype(int).values
    mvt_eobt1 = m["mvt_eobt1"].fillna(0).astype(float).values
    v59, _1 = apply_ms1(pred_vec, adep, flt_null, mvt_eobt1, support)
    v59, _2 = apply_ms2(v59, adep, support)
    assert_ms4(v59, template["MVT_ID_mvt"].values, m["MVT_ID_mvt"].values)
    out = pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].values,
                        "TAXITIME_SEC_mvt": v59.astype(np.float64)})
    (C.ROOT / "submission" / args.out).write_bytes(b"")
    out.to_parquet(C.ROOT / "submission" / args.out)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
