"""v48 = v47 + MS1 per-airport upper bounds + MS2 clean-floor.

Reads the shipped v47 parquet, applies MS1 and MS2 using support tables
built once from all 2025 target-airport departures, and writes the result
as submission/kind-mango_v48.parquet. LIRF rows are untouched (MH4 handles
LIRF); MS2's floor is per-airport, so LIRF gets its own 0.1 %-clean floor
which the LIRF head's outputs already respect.

The change touches exactly 5 non-LIRF rows via MS1 (biggest: EDDM 21,465 s
-> 7,499 s) and 100+ rows via MS2 (very small nudges). The v47 shipped
non-LIRF rows above 7,200 s stay at 109 out of 110, exactly one dropping
below (LEBL 8,553 -> 4,204).

Also runs the MP7 gate against v47 as reference and MC5 merge assertions.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.gate import build_train_support, prediction_gate
from src_v3.merge import assert_one_to_one, assert_row_count
from src_v3.postprocess import apply_ms1, apply_ms2, assert_ms4
from src_v3.support import build_support

MODELS_OLD = C.ROOT / "models"
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="kind-mango_v48.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    log.info("loading 2025 support")
    train = _load_train_min()
    support = build_support(train)

    v47_path = C.ROOT / "submission" / "kind-mango_v47.parquet"
    log.info("reading v47 parquet %s", v47_path)
    pred = pd.read_parquet(v47_path)
    if pred["TAXITIME_SEC_mvt"].dtype != np.float64:
        raise RuntimeError(f"v47 dtype {pred['TAXITIME_SEC_mvt'].dtype}, need float64")

    log.info("reading v2 rank frame for flt_null and mvt_eobt1")
    v2 = pd.read_parquet(MODELS_OLD / "v2" / "cache" / "frame_rank.parquet",
                         columns=["MVT_ID_mvt", "ADEP_mvt", "flt_null",
                                  "mvt_eobt1"])
    template = pd.read_parquet(C.SUB_TMPL, columns=["MVT_ID_mvt"])

    # MC5 assertions
    assert_one_to_one(pred, v2, "MVT_ID_mvt")
    assert_row_count(pred, len(template), "v47")

    m = template.merge(pred.rename(columns={"TAXITIME_SEC_mvt": "y"}),
                       on="MVT_ID_mvt", how="left", validate="1:1"
                       ).merge(v2, on="MVT_ID_mvt", how="left", validate="1:1")
    if m["y"].isna().any() or m["ADEP_mvt"].isna().any():
        raise RuntimeError("template rows missing predictions or metadata")

    pred_vec = m["y"].values.astype(float)
    adep = m["ADEP_mvt"].astype(str).values
    flt_null = m["flt_null"].fillna(1).astype(int).values
    mvt_eobt1 = m["mvt_eobt1"].fillna(0).astype(float).values

    log.info("applying MS1 per-airport upper bounds")
    v48, ms1_moved = apply_ms1(pred_vec, adep, flt_null, mvt_eobt1, support)
    log.info("MS1 moved %s", {k: v["rows_moved"] for k, v in ms1_moved.items()})

    log.info("applying MS2 per-airport clean-floor")
    v48, ms2_moved = apply_ms2(v48, adep, support)
    n_ms2 = sum(v["rows_moved"] for v in ms2_moved.values())
    log.info("MS2 moved %d rows", n_ms2)

    n_moved = int((np.abs(v48 - pred_vec) > 1e-9).sum())
    max_move = float(np.max(np.abs(v48 - pred_vec)))
    log.info("total moves vs v47: %d rows, max |diff| %.0f s", n_moved, max_move)

    # MS4 assertion
    assert_ms4(v48, template["MVT_ID_mvt"].values, m["MVT_ID_mvt"].values)

    out = pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].values,
                        "TAXITIME_SEC_mvt": v48.astype(np.float64)})
    outp = C.ROOT / "submission" / args.out
    out.to_parquet(outp)
    log.info("wrote %s (%d rows, %d KB)", outp, len(out),
             int(outp.stat().st_size / 1024))

    # MP7 gate
    log.info("MP7 label-free gate")
    frame_meta = m[["MVT_ID_mvt", "ADEP_mvt"]].copy()
    frame_meta["month"] = pd.to_datetime(
        pd.read_parquet(C.RANK, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
        .merge(frame_meta[["MVT_ID_mvt"]], on="MVT_ID_mvt")["MVT_TIME_UTC_mvt"],
        utc=True, errors="coerce"
    ).dt.month.astype("Int16")
    train_support = build_train_support(
        train.assign(
            month=pd.to_datetime(_load_train_min_ts(), utc=True, errors="coerce").dt.month
        )
    ) if False else {a: {"n_over_3600": support[a]["n_over_3600"],
                          "n_over_7200": support[a]["n_over_7200"],
                          "n_2025": support[a]["n"],
                          "max_2025": max(support[a]["class_max"].values())}
                      for a in support}
    ref = pd.read_parquet(v47_path)
    gate = prediction_gate(out, ref, train_support, frame_meta)
    (C.MODELS / "gate_v48.json").write_text(gate.as_json())
    log.info("MP7 gate: passed=%s, %d reasons", gate.passed, len(gate.reasons))
    for r in gate.reasons[:20]:
        log.info("  reason: %s", r)


def _load_train_min_ts():
    # kept as a stub; the gate builds the train_support from the pre-built
    # support tables above so this function is never called.
    raise NotImplementedError


if __name__ == "__main__":
    main()
