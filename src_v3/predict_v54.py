"""v54 = MB1 (anchored offset) base + MB3 filter + v41 LIRF head + MS1 + MS2.

The v54 boosters output the offset `y - anchor` (anchor = mvt_eobt1 if
not-null else sched_delay). This runner:

  1. Loads the v54 rank-side feature frame from `models/v3/
     v54_rank_features.parquet` (dumped by a previous predict_v30 run) or
     dumps it fresh if the cache is missing.
  2. Scores the three v54 boosters directly (bypassing predict_v30's
     final clip at zero, which would destroy the negative offsets that
     make up the median case).
  3. Reads mvt_eobt1 and sched_delay from the dump, computes the
     anchor row-by-row, and adds it to the offset predictions.
  4. Clips the served value to `anchor + [q0.001, q0.999]` per airport
     using `v54_offset_bounds.json`.
  5. Takes LIRF rows from the shipped v51 parquet (LIRF head unchanged).
  6. Applies MS1 + MS2 as v51 did.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict_v30 import main as predict_v30_main, load_training_categories
from predict_v45 import build_extra
from train_lgbm_v21 import CAT_COLS as SERVED_CAT_COLS
from train_r_norm_lirf_seeds import SEEDS as LIRF_SEEDS

from src_v3 import config as C
from src_v3.postprocess import apply_ms1, apply_ms2, assert_ms4
from src_v3.support import build_support

MODELS_OLD = C.ROOT / "models"
V2_RANK = MODELS_OLD / "v2" / "cache" / "frame_rank.parquet"
BOUNDS = MODELS_OLD / "v54_offset_bounds.json"
DUMP = C.MODELS / "v54_rank_features.parquet"
V54_SEEDS = (42, 43, 44)
R_NORM_FILES = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in LIRF_SEEDS]
log = logging.getLogger(__name__)


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


def _ensure_dump() -> None:
    if DUMP.exists():
        log.info("dump cache hit: %s", DUMP)
        return
    log.info("dumping v54 served features via predict_v30")
    predict_v30_main(
        "kind-mango_v54_dump.parquet",
        r_norm_files=R_NORM_FILES,
        base_model="lgbm_r_all_v54",
        extra_columns=build_extra(),
        r_norm_features="lirf_regime_v41.features.txt",
        r_norm_clip=None,
        per_member_base_clip=False,
        dump_features=str(DUMP),
    )
    (C.ROOT / "submission" / "kind-mango_v54_dump.parquet").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="kind-mango_v54.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    _ensure_dump()

    log.info("loading dump and feature list")
    with open(MODELS_OLD / "lgbm_r_all_v54.features.txt") as f:
        feat = f.read().splitlines()
    dump = pd.read_parquet(DUMP)
    dump = dump[["MVT_ID_mvt"] + [c for c in feat if c in dump.columns]
                + ["mvt_eobt1", "sched_delay", "ADEP_mvt"]]
    # De-duplicate columns (ADEP_mvt is in feat and in the extras list).
    dump = dump.loc[:, ~dump.columns.duplicated()]
    train_cats = load_training_categories()
    for c in SERVED_CAT_COLS:
        if c in dump.columns:
            dump[c] = pd.Categorical(dump[c].astype(str), categories=train_cats[c])

    log.info("scoring 3 v54 boosters directly (no clip)")
    preds = []
    for s in V54_SEEDS:
        t0 = time.time()
        b = lgb.Booster(model_file=str(MODELS_OLD / f"lgbm_r_all_v54_s{s}.txt"))
        p = b.predict(dump[feat])
        preds.append(p)
        log.info("v54 seed %d scored in %.0fs", s, time.time() - t0)
    offset = np.mean(preds, axis=0)
    log.info("offset stats: min=%.0f  max=%.0f  mean=%.1f",
             offset.min(), offset.max(), offset.mean())

    log.info("computing anchor and served y for non-LIRF rows")
    sd = dump["sched_delay"].astype(float).values
    eobt1 = dump["mvt_eobt1"].astype(float).values
    anchor = np.where(np.isnan(eobt1), sd, eobt1)
    adep_dump = dump["ADEP_mvt"].astype(str).values
    y_base = anchor + offset
    with open(BOUNDS) as f:
        bounds = json.load(f)
    for a, b in bounds.items():
        if a == "LIRF":
            continue
        mask = (adep_dump == a)
        if not mask.any():
            continue
        y_base[mask] = np.clip(y_base[mask],
                               anchor[mask] + b["q_low"],
                               anchor[mask] + b["q_high"])
    y_base = np.clip(y_base, 0, None)
    log.info("y_base stats: min=%.0f  max=%.0f  mean=%.1f",
             y_base.min(), y_base.max(), y_base.mean())

    log.info("merging: LIRF rows from v51, non-LIRF from v54")
    v51 = pd.read_parquet(C.ROOT / "submission" / "kind-mango_v51.parquet")
    template = pd.read_parquet(C.SUB_TMPL, columns=["MVT_ID_mvt"])
    v54_df = pd.DataFrame({"MVT_ID_mvt": dump["MVT_ID_mvt"].values,
                           "ADEP_mvt": adep_dump, "y_base": y_base})
    m = template.merge(v51.rename(columns={"TAXITIME_SEC_mvt": "y_v51"}),
                       on="MVT_ID_mvt", how="left", validate="1:1") \
                .merge(v54_df, on="MVT_ID_mvt", how="left", validate="1:1")
    lirf_mask = (m["ADEP_mvt"].astype(str) == "LIRF").values
    y_pred = np.where(lirf_mask, m["y_v51"].values.astype(float),
                      m["y_base"].values.astype(float))

    log.info("applying MS1 + MS2")
    v2 = pd.read_parquet(V2_RANK, columns=["MVT_ID_mvt", "ADEP_mvt",
                                            "flt_null", "mvt_eobt1"])
    aligned = template.merge(pd.DataFrame({"MVT_ID_mvt": m["MVT_ID_mvt"].values,
                                            "y": y_pred}),
                             on="MVT_ID_mvt", how="left", validate="1:1"
                            ).merge(v2, on="MVT_ID_mvt", how="left",
                                     validate="1:1")
    train = _load_train_min()
    support = build_support(train)
    pred_vec = aligned["y"].values.astype(float)
    adep2 = aligned["ADEP_mvt"].astype(str).values
    flt_null = aligned["flt_null"].fillna(1).astype(int).values
    mvt_eobt1 = aligned["mvt_eobt1"].fillna(0).astype(float).values
    v54, _ = apply_ms1(pred_vec, adep2, flt_null, mvt_eobt1, support)
    v54, _ = apply_ms2(v54, adep2, support)
    assert_ms4(v54, template["MVT_ID_mvt"].values, aligned["MVT_ID_mvt"].values)
    out = pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].values,
                        "TAXITIME_SEC_mvt": v54.astype(np.float64)})
    final_path = C.ROOT / "submission" / args.out
    out.to_parquet(final_path)
    log.info("wrote %s (%d rows, %d KB)", final_path, len(out),
             int(final_path.stat().st_size / 1024))


if __name__ == "__main__":
    main()
