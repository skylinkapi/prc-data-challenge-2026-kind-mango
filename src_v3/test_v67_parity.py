"""Parity test: rebuild v67 through the served command and compare it with the recorded hash.

`--record` writes the SHA-256 of the uploaded `kind-mango_v67.parquet` (ids
and predictions, template order, float64) to `submission/kind-mango_v67.parity.json`.
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
RECORD = SUB / "kind-mango_v67.parity.json"
SERVE = [sys.executable, "-m", "src_v3.predict_v67", "--weight", "0.5",
         "--pre-ms", "kind-mango_v67_parity_pre_ms.parquet", "--out", "kind-mango_v67_parity.parquet"]
log = logging.getLogger(__name__)


def digest(path) -> dict:
    """Row count and SHA-256 of the id and prediction columns."""
    f = pd.read_parquet(path)
    return {"rows": int(len(f)),
            "ids_sha256": hashlib.sha256(f["MVT_ID_mvt"].to_numpy(np.float64).tobytes()).hexdigest(),
            "pred_sha256": hashlib.sha256(f["TAXITIME_SEC_mvt"].to_numpy(np.float64).tobytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true", help="Record the hash of the uploaded v67 file.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.record:
        RECORD.write_text(json.dumps(digest(SUB / "kind-mango_v67.parquet"), indent=1))
        return
    subprocess.run(SERVE, check=True, cwd=C.ROOT)
    got, want = digest(SUB / "kind-mango_v67_parity.parquet"), json.loads(RECORD.read_text())
    for name in ("kind-mango_v67_parity.parquet", "kind-mango_v67_parity_pre_ms.parquet"):
        (SUB / name).unlink()
    if got != want:
        raise SystemExit(f"Parity failed. The rebuilt v67 file differs from the record ({got} != {want}). "
                         "Compare the served artefacts with REPRODUCE.md, then rerun.")
    log.info("parity ok: %d rows match the recorded v67 hash", got["rows"])


if __name__ == "__main__":
    main()
