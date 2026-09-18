"""v52 = MB3+MB4 base + v41 LIRF head + MS1 + MS2.

Builds the ranking-side MB4 calendar features (doy_sin/doy_cos/hour_local/
is_public_hol) once, saves them as `models/calendar_extra_rank.parquet`,
and merges them into the served predict path via extra_columns.
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict_v30 import main as predict_v30_main
from predict_v45 import build_extra as v45_build_extra
from train_r_norm_lirf_seeds import SEEDS as LIRF_SEEDS

from src_v3 import config as C
from src_v3.calendar_extra import add_calendar_extra
from src_v3.postprocess import apply_ms1, apply_ms2, assert_ms4
from src_v3.support import build_support

MODELS_OLD = C.ROOT / "models"
V2_RANK = MODELS_OLD / "v2" / "cache" / "frame_rank.parquet"
CAL_RANK = MODELS_OLD / "calendar_extra_rank.parquet"
EXTRA_FEATURES = ["doy_sin", "doy_cos", "hour_local", "is_public_hol"]
log = logging.getLogger(__name__)
R_NORM_FILES = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in LIRF_SEEDS]


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


def _build_rank_calendar() -> pd.DataFrame:
    if CAL_RANK.exists():
        log.info("cache hit: %s", CAL_RANK)
        return pd.read_parquet(CAL_RANK)
    log.info("building ranking calendar features")
    ranking = pd.read_parquet(V2_RANK, columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_ts"])
    ranking = add_calendar_extra(ranking)
    out = ranking[["MVT_ID_mvt", *EXTRA_FEATURES]]
    out.to_parquet(CAL_RANK)
    log.info("wrote %s (%d rows, %d cols)", CAL_RANK, len(out), len(out.columns))
    return out


def build_extra_v52() -> pd.DataFrame:
    tempo_plan = v45_build_extra()          # tempo + p2575 + plan
    cal = _build_rank_calendar()            # doy_sin/cos, hour_local, is_public_hol
    return tempo_plan.merge(cal, on="MVT_ID_mvt", how="left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre-ms", default="kind-mango_v52_pre_ms.parquet")
    ap.add_argument("--out", default="kind-mango_v52.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    log.info("=== predict_v52: MB3+MB4 base + v41 LIRF head ===")
    predict_v30_main(
        args.pre_ms,
        r_norm_files=R_NORM_FILES,
        base_model="lgbm_r_all_v52",
        extra_columns=build_extra_v52(),
        r_norm_features="lirf_regime_v41.features.txt",
        r_norm_clip=None,
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
    v52, _ = apply_ms1(pred_vec, adep, flt_null, mvt_eobt1, support)
    v52, _ = apply_ms2(v52, adep, support)
    n_moved = int((np.abs(v52 - pred_vec) > 1e-9).sum())
    max_move = float(np.max(np.abs(v52 - pred_vec)))
    log.info("post-processing moves: %d rows, max |diff| %.0f s", n_moved, max_move)
    assert_ms4(v52, template["MVT_ID_mvt"].values, m["MVT_ID_mvt"].values)
    out = pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].values,
                        "TAXITIME_SEC_mvt": v52.astype(np.float64)})
    final_path = C.ROOT / "submission" / args.out
    out.to_parquet(final_path)
    log.info("wrote %s (%d rows, %d KB)", final_path, len(out),
             int(final_path.stat().st_size / 1024))


if __name__ == "__main__":
    main()
