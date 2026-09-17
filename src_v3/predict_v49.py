"""v49 = v47 base + v49 LIRF head (drifted columns removed) + MS1 + MS2.

Reuses predict_v30.main with:
  - base_model = lgbm_r_all_v47 (unchanged)
  - r_norm_files, r_norm_features -> the v49 head files
  - p_fb_members, p_fb_features -> the v49 gate + isotonic
  - r_norm_clip = None (v41/v47 removed the 4,431 s cap)
  - extra_columns = the v45 join (tempo + p2575 + plan_taxi_res)
Then applies MS1 (per-airport upper bounds) and MS2 (clean floor) exactly
as v48 did on top of v47.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict_v30 import main as predict_v30_main
from predict_v45 import build_extra
from train_r_norm_lirf_seeds import SEEDS

from src_v3 import config as C
from src_v3.postprocess import apply_ms1, apply_ms2, assert_ms4
from src_v3.support import build_support

log = logging.getLogger(__name__)
V49_R_NORM_FILES = [f"lgbm_r_norm_lirf_v49_s{s}.txt" for s in SEEDS]
V49_P_FB_MEMBERS = [("lgbm_p_fb_lirf_v49.txt", "lirf_regime_v49.isotonic.pkl")]
V49_R_NORM_FEATURES = "lirf_regime_v49.features.txt"
V49_P_FB_FEATURES = "lirf_regime_v49.gate_features.txt"


def _load_train_min() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(C.TRAIN_DIR / "*.parquet"))):
        t = pd.read_parquet(f, columns=["ADEP_mvt", "PHASE_mvt",
                                        "MVT_TIME_UTC_mvt",
                                        "SCHED_TIME_UTC_mvt",
                                        "TAXITIME_SEC_mvt"])
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(C.TARGETS)]
        t = t[t["TAXITIME_SEC_mvt"].astype(float) > 0]
        t["mvt_ts"] = pd.to_datetime(t["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
        t["sched_ts"] = pd.to_datetime(t["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
        t["sched_delay"] = (t["mvt_ts"] - t["sched_ts"]).dt.total_seconds()
        parts.append(t[["ADEP_mvt", "TAXITIME_SEC_mvt", "sched_delay"]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="kind-mango_v49_pre_ms.parquet",
                    help="intermediate file before MS1+MS2 (default: pre_ms)")
    ap.add_argument("--final", default="kind-mango_v49.parquet",
                    help="final MS1+MS2 submission (default: kind-mango_v49.parquet)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    log.info("=== predict_v49: v47 base + v49 LIRF head ===")
    predict_v30_main(
        args.out,
        r_norm_files=V49_R_NORM_FILES,
        base_model="lgbm_r_all_v47",
        extra_columns=build_extra(),
        r_norm_features=V49_R_NORM_FEATURES,
        r_norm_clip=None,
        p_fb_members=V49_P_FB_MEMBERS,
        p_fb_features=V49_P_FB_FEATURES,
    )

    log.info("=== apply MS1 + MS2 to the served prediction ===")
    pre = pd.read_parquet(C.ROOT / "submission" / args.out)
    v2 = pd.read_parquet(C.ROOT / "models" / "v2" / "cache" / "frame_rank.parquet",
                         columns=["MVT_ID_mvt", "ADEP_mvt", "flt_null", "mvt_eobt1"])
    template = pd.read_parquet(C.SUB_TMPL, columns=["MVT_ID_mvt"])
    m = template.merge(pre.rename(columns={"TAXITIME_SEC_mvt": "y"}),
                       on="MVT_ID_mvt", how="left", validate="1:1") \
                .merge(v2, on="MVT_ID_mvt", how="left", validate="1:1")

    log.info("build 2025 support")
    train = _load_train_min()
    support = build_support(train)

    pred = m["y"].values.astype(float)
    adep = m["ADEP_mvt"].astype(str).values
    flt_null = m["flt_null"].fillna(1).astype(int).values
    mvt_eobt1 = m["mvt_eobt1"].fillna(0).astype(float).values

    v49, ms1_moved = apply_ms1(pred, adep, flt_null, mvt_eobt1, support)
    v49, ms2_moved = apply_ms2(v49, adep, support)
    n_moved = int((np.abs(v49 - pred) > 1e-9).sum())
    max_move = float(np.max(np.abs(v49 - pred)))
    log.info("MS1 moved %s", {k: v["rows_moved"] for k, v in ms1_moved.items()})
    log.info("MS2 moved %d rows", sum(v["rows_moved"] for v in ms2_moved.values()))
    log.info("total moves vs pre-MS: %d rows, max |diff| %.0f s", n_moved, max_move)
    assert_ms4(v49, template["MVT_ID_mvt"].values, m["MVT_ID_mvt"].values)
    out = pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].values,
                        "TAXITIME_SEC_mvt": v49.astype(np.float64)})
    final_path = C.ROOT / "submission" / args.final
    out.to_parquet(final_path)
    log.info("wrote %s (%d rows, %d KB)", final_path, len(out),
             int(final_path.stat().st_size / 1024))


if __name__ == "__main__":
    main()
