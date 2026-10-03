"""Unattended release of the next stack version once the January and July 2025 ADS-B days are on disk.

1. Wait until the download process `--wait-pid` exits.
2. Rebuild the ADS-B clock tables and freeze the training table under `external/adsblol/v<N>/`.
3. Price the stack rounds; keep `STACK_V90_ROUNDS` unless another count wins by `AUTO_BAR_MSE`.
4. Serve the v94 recipe (A5 stand bound, MLP at `AUTO_MLP_WEIGHT`) on the new table.
5. Gate the file against the reference; any failed gate stops the run before the upload.
6. Upload, wait for the score (retry after the UTC midnight reset on the daily limit), write the record.
The process keeps Windows awake while it runs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3 import stack_pushback_v90 as sp
from src_v3.tune_stack_v93 import frame

SUB = C.ROOT / "submission"
log = logging.getLogger(__name__)


def keep_awake() -> None:
    """ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no system sleep while this process lives."""
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def is_running(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def run(*args: str) -> None:
    log.info("run: %s", " ".join(args))
    subprocess.run([sys.executable, *args], check=True, cwd=C.ROOT)


def pick_rounds() -> tuple[int, dict]:
    """Price each round count on the frozen frame; a change needs a gain of `AUTO_BAR_MSE` over the default."""
    f = frame(C.AUTO_MLP_WEIGHT, False)
    prices = {n: sp.price(f, rounds=n)["all"] for n in C.AUTO_ROUNDS}
    best = min(prices, key=prices.get)
    keep = prices[C.STACK_V90_ROUNDS] - prices[best] < C.AUTO_BAR_MSE
    chosen = C.STACK_V90_ROUNDS if keep else best
    log.info("stack prices by rounds %s; chosen %d", prices, chosen)
    if prices[chosen] > C.AUTO_MAX_STACK_PRICE:
        raise SystemExit(f"The stack price {prices[chosen]:.0f} MSE is above the sanity bar {C.AUTO_MAX_STACK_PRICE}. "
                         "The clock tables are suspect. Inspect logs/auto_release.log, then run by hand.")
    return chosen, prices


def gates(out: str, ref: str) -> dict:
    """Label-free gates against the reference file; raise on the first failure."""
    subprocess.run([sys.executable, "src/check_submission_2026.py", str(SUB / out), "--ref", str(SUB / ref)],
                   check=True, cwd=C.ROOT)
    chk = json.loads((SUB / out).with_suffix(".check.json").read_text())
    new, old = pd.read_parquet(SUB / out), pd.read_parquet(SUB / ref)
    big = int((np.abs(new["TAXITIME_SEC_mvt"].values - old["TAXITIME_SEC_mvt"].values) > C.AUTO_BIG_MOVE_S).sum())
    checks = {"rows": len(new) == chk["n"] == len(old), "float64": str(new["TAXITIME_SEC_mvt"].dtype) == "float64",
              "no_nan": not new["TAXITIME_SEC_mvt"].isna().any(),
              "lirf_unchanged": chk["per_airport"]["LIRF"]["max_abs_diff"] == 0,
              "over_80000": chk["over_80000"] == chk["ref_over_80000"],
              "over_7200": chk["over_7200"] <= chk["ref_over_7200"] + C.AUTO_MAX_NEW_7200,
              "big_moves": big <= C.AUTO_MAX_BIG_MOVES}
    log.info("gates %s (rows moved by more than %d s: %d)", checks, C.AUTO_BIG_MOVE_S, big)
    if not all(checks.values()):
        raise SystemExit(f"Gates failed: {[k for k, v in checks.items() if not v]}. The file is not uploaded. "
                         "Inspect the check JSON, then decide by hand.")
    return checks


def minio_client():
    from minio import Minio
    kv = dict(line.split(" = ", 1) for line in (C.ROOT / ".osn_credentials.txt").read_text().splitlines() if " = " in line)
    return Minio("s3.opensky-network.org", access_key=kv["access_key"], secret_key=kv["secret_key"], secure=True), \
        kv.get("bucket", "prc-2026-kind-mango")


def upload(out: str) -> dict:
    """Upload and wait for the result; on the daily limit, wait for the UTC reset and upload again once."""
    from minio.error import S3Error
    client, bucket = minio_client()
    for attempt in (1, 2):
        client.fput_object(bucket, out, str(SUB / out))
        log.info("uploaded %s (attempt %d)", out, attempt)
        for _ in range(C.AUTO_RESULT_POLLS):
            time.sleep(C.AUTO_RESULT_WAIT_S)
            try:
                result = json.loads(client.get_object(bucket, f"{out}_result.json").read())
                break
            except S3Error as e:
                if e.code != "NoSuchKey":  # the result object does not exist until the scorer writes it
                    raise
        else:
            raise SystemExit(f"No result for {out} after {C.AUTO_RESULT_POLLS * C.AUTO_RESULT_WAIT_S} s. Check the bucket by hand.")
        if result.get("status") == "Succeeded":
            return result
        if attempt == 2 or "LIMIT" not in json.dumps(result).upper():
            raise SystemExit(f"The scorer rejected {out}: {result}. Fix the cause, then upload by hand.")
        client.remove_object(bucket, f"{out}_result.json")
        now = datetime.now(timezone.utc)
        reset = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
        log.info("daily limit reached; waiting until %s", reset.isoformat())
        time.sleep((reset - now).total_seconds())
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--wait-pid", type=int, required=True, help="Process of the January and July 2025 downloads.")
    ap.add_argument("--ref", default="kind-mango_v94.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(C.ROOT / "logs" / "auto_release.log"), logging.StreamHandler()])
    keep_awake()
    while is_running(args.wait_pid):
        time.sleep(C.AUTO_POLL_S)
    days = sorted(p.stem for p in (C.ADSB_DIR / "points").glob("2025.0[17].*.parquet"))
    log.info("download process ended; %d January and July 2025 days on disk", len(days))
    run("-m", "src_v3.build_adsb_clock")
    frozen = C.ADSB_DIR / f"v{args.version}" / "adsb_clock_train.parquet"
    frozen.parent.mkdir(exist_ok=True)
    shutil.copy(C.ADSB_DIR / "adsb_clock_train.parquet", frozen)
    rounds, prices = pick_rounds()
    out = f"kind-mango_v{args.version}.parquet"
    stack_args = ["--sources", "both", "--stand-bound", "--mlp-weight", str(C.AUTO_MLP_WEIGHT),
                  "--adsb-train", frozen.relative_to(C.ROOT).as_posix(), "--rounds", str(rounds)]
    run("-m", "src_v3.stack_pushback_v90", "serve", *stack_args, "--base", "kind-mango_v89.parquet", "--out", out)
    checks = gates(out, args.ref)
    result = upload(out)
    (SUB / f"kind-mango_v{args.version}.result.json").write_text(json.dumps(result, indent=1))
    (SUB / f"kind-mango_v{args.version}.recipe.json").write_text(json.dumps(
        {"stack_args": stack_args, "base": "kind-mango_v89.parquet", "train_days_2025": days,
         "stack_prices_by_rounds": prices, "gates": checks, "score": result["score"]}, indent=1))
    log.info("done: %s scored %.4f s", out, result["score"])


if __name__ == "__main__":
    main()
