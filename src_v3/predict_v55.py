"""v55 = v51 recipe with R_norm_LIRF refit on all 12 months (MB8).

Everything else stays exactly at v51: MB3 base (lgbm_r_all_v51), v23 p_fb
gate, Step A band table, ITY340 rule, MS1 + MS2 post-processing. Only
R_norm_LIRF gets swapped to the 12-month-refit v55 files.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from src_v3.predict_v51 import _load_train_min  # reuse helper
from src_v3.predict_v51 import main as _  # only to force import chain if needed

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre-ms", default="kind-mango_v55_pre_ms.parquet")
    ap.add_argument("--out", default="kind-mango_v55.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from predict_v30 import main as predict_v30_main
    from predict_v45 import build_extra

    import numpy as np
    import pandas as pd
    from src_v3 import config as C
    from src_v3.postprocess import apply_ms1, apply_ms2, assert_ms4
    from src_v3.support import build_support

    R_NORM_FILES = [f"lgbm_r_norm_lirf_v55_s{s}.txt" for s in (42, 43, 44, 45, 46)]

    log.info("=== v55: MB3 base + v55 R_norm (12-month refit) ===")
    predict_v30_main(
        args.pre_ms,
        r_norm_files=R_NORM_FILES,
        base_model="lgbm_r_all_v51",
        extra_columns=build_extra(),
        r_norm_features="lirf_regime_v41.features.txt",
        r_norm_clip=None,
    )

    log.info("=== apply MS1 + MS2 ===")
    pre = pd.read_parquet(C.ROOT / "submission" / args.pre_ms)
    v2 = pd.read_parquet(C.ROOT / "models" / "v2" / "cache" / "frame_rank.parquet",
                         columns=["MVT_ID_mvt", "ADEP_mvt", "flt_null", "mvt_eobt1"])
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
    v55, _1 = apply_ms1(pred_vec, adep, flt_null, mvt_eobt1, support)
    v55, _2 = apply_ms2(v55, adep, support)
    assert_ms4(v55, template["MVT_ID_mvt"].values, m["MVT_ID_mvt"].values)
    out = pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].values,
                        "TAXITIME_SEC_mvt": v55.astype(np.float64)})
    final_path = C.ROOT / "submission" / args.out
    out.to_parquet(final_path)
    log.info("wrote %s (%d rows, %d KB)", final_path, len(out),
             int(final_path.stat().st_size / 1024))


if __name__ == "__main__":
    main()
