"""Parity test: rebuild a recorded file through its served commands and compare it with the recorded hash.

`--record` writes the SHA-256 of the uploaded `kind-mango_v<N>.parquet` (ids
and predictions, template order, float64) to `submission/kind-mango_v<N>.parity.json`.
Without it the test rebuilds the file under scratch names, hashes it, and
fails when the hash differs from the record. `RECIPES` holds the served
commands of every recorded version (C1 of the second-edition plan).
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
PY = [sys.executable, "-m"]
V89_BASE = ["src_v3.predict_v67", "--weight", "0.5", "--ms3", "--base-model", "lgbm_r_all_v85",
            "--catboost-tag", "v73", "--catboost-anchor-tag", "v75a", "--anchor-model", "lgbm_r_all_v85a",
            "--anchor-weight", "0.5", "--per-airport-weight", "0.5", "--extra", "models/plan_nm_taxi_rank_v65.parquet",
            "models/nm_clocks_rank_v73.parquet", "models/nm_forward_rank_v85.parquet"]
RECIPES = {
    67: lambda p: [["src_v3.predict_v67", "--weight", "0.5", "--pre-ms", f"{p}_pre_ms.parquet", "--out", f"{p}.parquet"]],
    92: lambda p: [[*V89_BASE, "--per-airport-tag", *C.V92_PER_AIRPORT_TAGS,
                    "--pre-ms", f"{p}_base_pre_ms.parquet", "--out", f"{p}_base.parquet"],
                   ["src_v3.stack_pushback_v90", "serve", *C.V92_STACK_ARGS, "--base", f"{p}_base.parquet",
                    "--out", f"{p}.parquet"]],
    **{v: (lambda args: lambda p: [[*V89_BASE, "--per-airport-tag", "v89", "--pre-ms", f"{p}_base_pre_ms.parquet",
                                    "--out", f"{p}_base.parquet"],
                                   ["src_v3.stack_pushback_v90", "serve", *args, "--base", f"{p}_base.parquet",
                                    "--out", f"{p}.parquet"]])(args)
       for v, args in ((94, C.V94_STACK_ARGS), (95, C.V95_STACK_ARGS))},
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
    ap.add_argument("--version", type=int, choices=sorted(RECIPES), default=max(RECIPES))
    ap.add_argument("--record", action="store_true", help="Record the hash of the uploaded file.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    record = SUB / f"kind-mango_v{args.version}.parity.json"
    if args.record:
        record.write_text(json.dumps(digest(SUB / f"kind-mango_v{args.version}.parquet"), indent=1))
        return
    prefix = f"kind-mango_v{args.version}_parity"
    for step in RECIPES[args.version](prefix):
        subprocess.run([*PY, *step], check=True, cwd=C.ROOT)
    got, want = digest(SUB / f"{prefix}.parquet"), json.loads(record.read_text())
    for scratch in SUB.glob(f"{prefix}*.parquet"):
        scratch.unlink()
    if got != want:
        raise SystemExit(f"Parity failed. The rebuilt v{args.version} file differs from the record ({got} != {want}). "
                         "Compare the served artefacts with REPRODUCE.md, then rerun.")
    log.info("parity ok: %d rows match the recorded v%d hash", got["rows"], args.version)


if __name__ == "__main__":
    main()
