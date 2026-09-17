"""Apply MS1-MS4 to the shipped v47 submission and report the label-free
effect on the 2026 ranking rows. This is the honest MS1-MS4 sizing: on the
2026 predictions where the extreme rows sit.

Uses the 2025 training frame for support (frozen once per airport across
all 12 months of 2025). Reads submission/kind-mango_v47.parquet and the
v2 ranking frame for flt_null and mvt_eobt1. Writes
models/v3/apply_ms_v47.json.

For MS3 (member disagreement) we need the three v47 members individually.
The runner runs each on the ranking frame; that adds a few minutes.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.postprocess import apply_ms1, apply_ms2, apply_ms3, apply_pipeline
from src_v3.support import build_support

MODELS_OLD = C.ROOT / "models"
V47_SEEDS = (42, 43, 44)
log = logging.getLogger(__name__)


def _load_train_min() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(C.TRAIN_DIR / "*.parquet"))):
        t = pd.read_parquet(f, columns=["MVT_ID_mvt", "ADEP_mvt",
                                        "PHASE_mvt", "MVT_TIME_UTC_mvt",
                                        "SCHED_TIME_UTC_mvt",
                                        "TAXITIME_SEC_mvt"])
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(C.TARGETS)]
        t = t[t["TAXITIME_SEC_mvt"].astype(float) > 0]
        t["mvt_ts"] = pd.to_datetime(t["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
        t["sched_ts"] = pd.to_datetime(t["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
        t["sched_delay"] = (t["mvt_ts"] - t["sched_ts"]).dt.total_seconds()
        parts.append(t[["ADEP_mvt", "TAXITIME_SEC_mvt", "sched_delay"]])
    return pd.concat(parts, ignore_index=True)


def _load_v47_members_scored_on_rank() -> tuple[np.ndarray, pd.DataFrame]:
    """Score the three v47 boosters on the ranking frame and return the
    members array (3, n) and the meta dataframe with adep, flt_null,
    mvt_eobt1."""
    v2 = pd.read_parquet(MODELS_OLD / "v2" / "cache" / "frame_rank.parquet",
                         columns=["MVT_ID_mvt", "ADEP_mvt", "flt_null",
                                  "mvt_eobt1"])
    with open(MODELS_OLD / "lgbm_r_all_v47.features.txt") as f:
        feat = f.read().splitlines()

    # v47 was scored by predict_v30.main; that predict path assembled the
    # served features in memory. Reproducing that here means running the
    # whole feature pipeline. For MS1-MS4 sizing we only need the ADEP,
    # flt_null and mvt_eobt1 metadata and the served final prediction.
    #
    # We already have the served final prediction in submission/
    # kind-mango_v47.parquet. Member-level predictions are not saved for
    # v47's ranking run, so MS3 (member disagreement) is measured only on
    # the fold (measure_ms.py). Here we apply MS1 and MS2 to the shipped
    # served ensemble mean.
    pred = pd.read_parquet(C.ROOT / "submission" / "kind-mango_v47.parquet")
    m = v2.merge(pred.rename(columns={"TAXITIME_SEC_mvt": "y_hat"}),
                 on="MVT_ID_mvt", how="left", validate="1:1")
    return m["y_hat"].values.astype(float), m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    log.info("loading 2025 training frame for support")
    t0 = time.time()
    train = _load_train_min()
    log.info("2025 train %d rows in %.0fs", len(train), time.time() - t0)
    support = build_support(train)
    (C.MODELS / "support_all_2025.json").write_text(json.dumps(support, indent=1))

    log.info("scoring v47 shipped output")
    pred, meta = _load_v47_members_scored_on_rank()
    adep = meta["ADEP_mvt"].astype(str).values
    flt_null = meta["flt_null"].fillna(1).astype(int).values
    mvt_eobt1 = meta["mvt_eobt1"].fillna(0).astype(float).values

    # LIRF rows carry the LIRF head's serve; MS1 does not touch LIRF, but
    # the report still lists them for the diagnostic.
    variants: dict[str, np.ndarray] = {"v47_shipped": pred.copy()}
    v_ms1, moved1 = apply_ms1(pred.copy(), adep, flt_null, mvt_eobt1, support)
    v_ms2, moved2 = apply_ms2(v_ms1.copy(), adep, support)
    variants["MS1"] = v_ms1
    variants["MS1_MS2"] = v_ms2

    diffs = {}
    for name, p in variants.items():
        d = p - pred
        n_moved = int((np.abs(d) > 1e-9).sum())
        max_up = float(d.max())
        max_down = float(d.min())
        big_moves = np.where(np.abs(d) >= 60.0)[0]
        big_rows = []
        for i in big_moves[:200]:
            big_rows.append({
                "row": int(i),
                "airport": str(adep[i]),
                "before": float(pred[i]),
                "after": float(p[i]),
                "diff_s": float(d[i]),
                "flt_null": int(flt_null[i]),
                "mvt_eobt1": float(mvt_eobt1[i]),
            })
        # count over 7,200 after each variant per airport
        per_apt = {}
        for a in np.unique(adep):
            m = (adep == a)
            per_apt[str(a)] = {
                "n": int(m.sum()),
                "before_over_7200": int((pred[m] > 7200).sum()),
                "after_over_7200": int((p[m] > 7200).sum()),
                "mean_shift_s": float(d[m].mean()),
                "max_abs_diff_s": float(np.abs(d[m]).max()),
            }
        diffs[name] = {"rows_moved": n_moved, "max_up_s": max_up,
                       "max_down_s": max_down,
                       "big_moves_ge_60s": big_rows,
                       "n_big_moves": len(big_moves),
                       "per_airport": per_apt}

    payload = {"variants": diffs, "ms1_moved": moved1, "ms2_moved": moved2}
    outp = C.MODELS / "apply_ms_v47.json"
    outp.write_text(json.dumps(payload, indent=1))
    log.info("wrote %s", outp)
    log.info("v47_shipped rows over 7200: %d", int((pred > 7200).sum()))
    log.info("MS1 rows over 7200: %d, rows moved: %d, max down %.0f s",
             int((variants["MS1"] > 7200).sum()),
             diffs["MS1"]["rows_moved"], -diffs["MS1"]["max_down_s"])
    log.info("MS1+MS2 rows over 7200: %d, rows moved: %d",
             int((variants["MS1_MS2"] > 7200).sum()),
             diffs["MS1_MS2"]["rows_moved"])


if __name__ == "__main__":
    main()
