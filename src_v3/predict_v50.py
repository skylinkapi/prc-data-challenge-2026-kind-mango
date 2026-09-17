"""v50 = v48 + MS3 (member-disagreement median).

v48 already serves v47_mean + MS1 + MS2. MS3 requires per-member v47
predictions on the 2026 ranking frame. This runner:

  1. Calls predict_v30.main with dump_features=<path> and the v47 base to
     regenerate the served feature frame and save it as a parquet.
  2. Loads that dump, loads the 3 v47 boosters, scores each individually.
  3. Where the three-member spread exceeds 3,600 s outside LIRF, replaces
     the ensemble mean with the member median; otherwise keeps the mean.
  4. Wraps LIRF rows back onto the v48 LIRF-head serve (unchanged, since
     MS3 only touches non-LIRF).
  5. Applies MS1 + MS2 as v48 did.

MS3 fits the winning pattern: label-free defensive gating on rows where
the base disagrees, i.e. out-of-distribution rows. If it stacks with MS1
we should see a small further live gain.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import pickle
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
from src_v3.postprocess import apply_ms1, apply_ms2, apply_ms3, assert_ms4
from src_v3.support import build_support

MODELS_OLD = C.ROOT / "models"
V47_SEEDS = (42, 43, 44)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    R_NORM_FILES_V41 = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in LIRF_SEEDS]
    dump_path = C.MODELS / "v47_rank_features.parquet"
    pre_ms_path = C.ROOT / "submission" / "kind-mango_v50_pre_ms.parquet"

    if dump_path.exists() and pre_ms_path.exists():
        log.info("=== predict_v50 step 1: dump + pre-MS cache hit ===")
    else:
        log.info("=== predict_v50 step 1: dump v47 served features ===")
        predict_v30_main(
            pre_ms_path.name,
            r_norm_files=R_NORM_FILES_V41,
            base_model="lgbm_r_all_v47",
            extra_columns=build_extra(),
            r_norm_features="lirf_regime_v41.features.txt",
            r_norm_clip=None,
            dump_features=str(dump_path),
        )

    log.info("=== step 2: score 3 v47 members individually ===")
    with open(MODELS_OLD / "lgbm_r_all_v47.features.txt") as f:
        feat = f.read().splitlines()
    dump = pd.read_parquet(dump_path)
    # ADEP_mvt is already in `feat`; keep only MVT_ID_mvt as an extra key.
    dump = dump[["MVT_ID_mvt"] + [c for c in feat if c in dump.columns]]
    # Apply the training-set categorical vocab so LightGBM's stored
    # `pandas_categorical` matches. This mirrors predict_v30.py:110,152-154.
    log.info("aligning categorical vocab with the training set")
    train_cats = load_training_categories()
    for c in SERVED_CAT_COLS:
        if c in dump.columns:
            dump[c] = pd.Categorical(dump[c].astype(str), categories=train_cats[c])

    members: list[np.ndarray] = []
    for s in V47_SEEDS:
        t0 = time.time()
        b = lgb.Booster(model_file=str(MODELS_OLD / f"lgbm_r_all_v47_s{s}.txt"))
        p = np.clip(b.predict(dump[feat]), 0, None)
        members.append(p)
        log.info("v47 seed %d scored in %.0fs", s, time.time() - t0)
    members_arr = np.array(members)
    mean = members_arr.mean(axis=0)

    log.info("=== step 3: MS3 disagreement gate ===")
    adep_dump = dump["ADEP_mvt"].astype(str).values
    ms3_pred, ms3_moved = apply_ms3(members_arr, adep_dump)
    n_ms3 = int(sum(v["rows_moved"] for v in ms3_moved.values()))
    log.info("MS3 moved %d rows: %s", n_ms3,
             {k: v["rows_moved"] for k, v in ms3_moved.items()})

    log.info("=== step 4: wrap LIRF back onto v48 (LIRF head unchanged) ===")
    v48_pred = pd.read_parquet(C.ROOT / "submission" / "kind-mango_v48.parquet"
                                ).rename(columns={"TAXITIME_SEC_mvt": "v48"})
    pre = pd.read_parquet(pre_ms_path).rename(columns={"TAXITIME_SEC_mvt": "pre"})

    # Merge dump ADEP into an align frame keyed on MVT_ID_mvt
    align = dump[["MVT_ID_mvt", "ADEP_mvt"]].assign(base_ms3=ms3_pred,
                                                    base_mean=mean)
    # Non-LIRF rows only: use MS3 correction over the base MEAN, then combine
    # into the pre-MS shipped final by using the RATIO of MS3/mean to the
    # served value. Actually simpler: recompute serve_final = (pre - base_mean
    # + base_ms3) so LIRF rows are unchanged (base_ms3 == base_mean for LIRF).
    #
    # But pre is the served value with base_mean already substituted by LIRF
    # head for LIRF rows. Adding (base_ms3 - base_mean) preserves LIRF (both
    # zero there since MS3 skips LIRF) and shifts non-LIRF where MS3 diverges.
    align["delta"] = align["base_ms3"] - align["base_mean"]
    m = pre.merge(align[["MVT_ID_mvt", "delta"]], on="MVT_ID_mvt", how="left",
                  validate="1:1").merge(v48_pred, on="MVT_ID_mvt", how="left",
                                        validate="1:1")
    corrected = (m["pre"].values.astype(float)
                 + m["delta"].fillna(0).values.astype(float))

    log.info("=== step 5: MS1 + MS2 post-processing ===")
    v2 = pd.read_parquet(MODELS_OLD / "v2" / "cache" / "frame_rank.parquet",
                         columns=["MVT_ID_mvt", "ADEP_mvt", "flt_null",
                                  "mvt_eobt1"])
    template = pd.read_parquet(C.SUB_TMPL, columns=["MVT_ID_mvt"])
    aligned = template.merge(pd.DataFrame({"MVT_ID_mvt": m["MVT_ID_mvt"].values,
                                            "y": corrected}),
                             on="MVT_ID_mvt", how="left", validate="1:1"
                            ).merge(v2, on="MVT_ID_mvt", how="left",
                                     validate="1:1")

    train = _load_train_min()
    support = build_support(train)
    pred_vec = aligned["y"].values.astype(float)
    adep = aligned["ADEP_mvt"].astype(str).values
    flt_null = aligned["flt_null"].fillna(1).astype(int).values
    mvt_eobt1 = aligned["mvt_eobt1"].fillna(0).astype(float).values
    v50, _ = apply_ms1(pred_vec, adep, flt_null, mvt_eobt1, support)
    v50, _ = apply_ms2(v50, adep, support)

    n_moved_vs_v48 = int((np.abs(v50 - m["v48"].values) > 1e-9).sum())
    max_move_vs_v48 = float(np.max(np.abs(v50 - m["v48"].values)))
    log.info("v50 vs v48: %d rows moved, max |diff| %.0f s",
             n_moved_vs_v48, max_move_vs_v48)
    assert_ms4(v50, template["MVT_ID_mvt"].values, aligned["MVT_ID_mvt"].values)
    out = pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].values,
                        "TAXITIME_SEC_mvt": v50.astype(np.float64)})
    outp = C.ROOT / "submission" / "kind-mango_v50.parquet"
    out.to_parquet(outp)
    log.info("wrote %s (%d rows, %d KB)", outp, len(out),
             int(outp.stat().st_size / 1024))

    # Report
    (C.MODELS / "predict_v50.json").write_text(json.dumps({
        "v47_members_scored": list(V47_SEEDS),
        "ms3_rows_moved_per_airport": {k: v["rows_moved"] for k, v in ms3_moved.items()},
        "ms3_total_rows_moved": n_ms3,
        "rows_diff_vs_v48": n_moved_vs_v48,
        "max_abs_diff_vs_v48_s": max_move_vs_v48,
    }, indent=1))


if __name__ == "__main__":
    main()
