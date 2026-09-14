"""H3: label-free 2026 checks on a submission file against the best file.

Compares a candidate parquet with the reference (default: the live best,
kind-mango_v41.parquet) on the 344,841 ranking rows: per-airport mean shift,
counts over 7,200 and 80,000 s, rows at the zero clip, and the rows that
differ per airport. Writes <candidate>.check.json next to the candidate.
"""
import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_RANK = os.path.join(ROOT, "models", "v2", "cache", "frame_rank.parquet")
DEFAULT_REF = os.path.join(ROOT, "submission", "kind-mango_v41.parquet")
log = logging.getLogger(__name__)


def load_pred(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df["TAXITIME_SEC_mvt"].dtype != np.float64:
        raise RuntimeError(f"{path}: dtype is {df['TAXITIME_SEC_mvt'].dtype}, the scorer needs float64")
    return df


def check(cand_path: str, ref_path: str) -> dict:
    cand, ref = load_pred(cand_path), load_pred(ref_path)
    meta = pd.read_parquet(FRAME_RANK, columns=["MVT_ID_mvt", "ADEP_mvt", "month"])
    m = meta.merge(cand.rename(columns={"TAXITIME_SEC_mvt": "new"}), on="MVT_ID_mvt", how="left",
                   validate="1:1").merge(
        ref.rename(columns={"TAXITIME_SEC_mvt": "ref"}), on="MVT_ID_mvt", how="left", validate="1:1")
    if m["new"].isna().any() or m["ref"].isna().any():
        raise RuntimeError("a file does not cover every template row")
    new, ref_v = m["new"].values, m["ref"].values
    diff = new - ref_v
    moved = np.abs(diff) > 1e-9
    out = {"file": os.path.basename(cand_path), "ref": os.path.basename(ref_path),
           "n": int(len(m)), "rows_moved": int(moved.sum()),
           "over_7200": int((new > 7200).sum()), "over_80000": int((new > 80000).sum()),
           "at_zero_clip": int((new == 0).sum()),
           "ref_over_7200": int((ref_v > 7200).sum()), "ref_over_80000": int((ref_v > 80000).sum()),
           "ref_at_zero_clip": int((ref_v == 0).sum()), "per_airport": {}}
    for a, g in m.groupby(m["ADEP_mvt"].astype(str)):
        i = g.index.values
        out["per_airport"][a] = {
            "n": int(len(g)), "mean_new": float(new[i].mean()), "mean_ref": float(ref_v[i].mean()),
            "mean_shift": float(diff[i].mean()), "rows_moved": int(moved[i].sum()),
            "mean_abs_diff": float(np.abs(diff[i]).mean()), "max_abs_diff": float(np.abs(diff[i]).max())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate")
    ap.add_argument("--ref", default=DEFAULT_REF)
    args = ap.parse_args()
    rep = check(args.candidate, args.ref)
    out = args.candidate.replace(".parquet", ".check.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    log.info("%s vs %s: rows moved %d, over 7,200 %d, over 80,000 %d, at clip %d",
             rep["file"], rep["ref"], rep["rows_moved"], rep["over_7200"],
             rep["over_80000"], rep["at_zero_clip"])
    for a, r in sorted(rep["per_airport"].items()):
        log.info("  %s: shift %+8.2f s  moved %7d  mean|d| %7.2f  max|d| %8.1f",
                 a, r["mean_shift"], r["rows_moved"], r["mean_abs_diff"], r["max_abs_diff"])
    log.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
