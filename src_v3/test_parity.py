"""Parity test: rebuild a served file through its command and compare it with the recorded hash.

`--record` writes the SHA-256 of the uploaded `kind-mango_<tag>.parquet` (ids
and predictions, template order, float64) to `submission/kind-mango_<tag>.parity.json`.
Without it the test rebuilds the file into a scratch name, hashes it, and
fails when the hash differs from the record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys

import numpy as np
import pandas as pd

from src_v3 import config as C

SUB = C.ROOT / "submission"
# `predict_v67` arguments of each served file, as uploaded (README results table).
SERVE_ARGS = {
    "v67": ["--weight", "0.5"],
    "v89": ["--weight", "0.5", "--ms3", "--base-model", "lgbm_r_all_v85", "--catboost-tag", "v73",
            "--catboost-anchor-tag", "v75a", "--anchor-model", "lgbm_r_all_v85a", "--anchor-weight", "0.5",
            "--per-airport-tag", "v89", "--per-airport-weight", "0.5",
            "--extra", "models/plan_nm_taxi_rank_v65.parquet", "models/nm_clocks_rank_v73.parquet",
            "models/nm_forward_rank_v85.parquet"],
}
log = logging.getLogger(__name__)


def digest(path) -> dict:
    """Row count and SHA-256 of the id and prediction columns."""
    f = pd.read_parquet(path)
    return {"rows": int(len(f)),
            "ids_sha256": hashlib.sha256(f["MVT_ID_mvt"].to_numpy(np.float64).tobytes()).hexdigest(),
            "pred_sha256": hashlib.sha256(f["TAXITIME_SEC_mvt"].to_numpy(np.float64).tobytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", choices=sorted(SERVE_ARGS), required=True, help="Served file to check.")
    ap.add_argument("--record", action="store_true", help="Record the hash of the uploaded file.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    record = SUB / f"kind-mango_{args.tag}.parity.json"
    if args.record:
        record.write_text(json.dumps(digest(SUB / f"kind-mango_{args.tag}.parquet"), indent=1))
        return
    if not record.exists():
        raise SystemExit(f"No parity record for {args.tag}. The file {record.name} is missing. "
                         f"Run with --tag {args.tag} --record on the uploaded file, then rerun.")
    out, pre_ms = f"kind-mango_{args.tag}_parity.parquet", f"kind-mango_{args.tag}_parity_pre_ms.parquet"
    subprocess.run([sys.executable, "-m", "src_v3.predict_v67", *SERVE_ARGS[args.tag],
                    "--pre-ms", pre_ms, "--out", out], check=True, cwd=C.ROOT)
    got, want = digest(SUB / out), json.loads(record.read_text())
    for name in (out, pre_ms):
        (SUB / name).unlink()
    if got != want:
        raise SystemExit(f"Parity failed. The rebuilt {args.tag} file differs from the record ({got} != {want}). "
                         "Compare the served artefacts with REPRODUCE.md, then rerun.")
    log.info("parity ok: %d rows match the recorded %s hash", got["rows"], args.tag)


if __name__ == "__main__":
    main()
