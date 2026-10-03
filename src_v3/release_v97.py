"""Unattended v97 release: the base members refit with the OPDI and ADS-B pushback columns.

1. Wait for `--wait-pid` (the hold-out chain of `measure_adsb_base_v97`), read its price, and stop
   unless blend plus stack beats v95's recipe by `AUTO_BAR_MSE`.
2. Refit the five served base members on all 2025 rows in two groups of parallel processes.
3. Serve the base with `predict_v67`, then the v95 stack trained on the v97 hold-out blend.
4. Gate against the reference file and upload (`auto_release.gates`, `auto_release.upload`).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time

from src_v3 import config as C
from src_v3.auto_release import gates, is_running, keep_awake, upload
from src_v3.measure_adsb_base_v97 import blend_path, price as measure_price

SUB = C.ROOT / "submission"
PUSHBACK = ("models/opdi_own_train_v90.parquet", "external/adsblol/adsb_clock_train.parquet")
FORWARD = "models/nm_forward_train_v85.parquet"
CLOCKS = "models/nm_clocks_train_v73.parquet"
log = logging.getLogger(__name__)


def group(*jobs: list[str]) -> None:
    """Run the module commands in parallel and fail if any of them fails."""
    procs = [subprocess.Popen([sys.executable, "-m", *j], cwd=C.ROOT) for j in jobs]
    log.info("started %s", [" ".join(j[:3]) for j in jobs])
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"A refit failed (exit codes {codes}). Inspect logs/release_v97.log, then rerun the failed step.")


def refit(tag: str, per_airport: bool) -> None:
    """Fit every member whose last output file is missing; members fitted ahead of the price are kept."""
    lgb_extra = ["--extra", FORWARD, *PUSHBACK]
    cat_extra = ["--extra", CLOCKS, *PUSHBACK, "--rounds-from", "v67"]
    jobs = [(f"lgbm_r_all_{tag}_s44.txt", ["src_v3.train_v76_base", "plain", "--tag", tag, *lgb_extra]),
            (f"lgbm_r_all_{tag}a_s44.txt", ["src_v3.train_v76_base", "anchored", "--tag", tag, *lgb_extra]),
            (f"catboost_r_all_{tag}c.cbm", ["src_v3.catboost_base", "--tag", f"{tag}c", *cat_extra]),
            (f"lgbm_pa_{tag}_LTFM.txt", ["src_v3.train_per_airport_v89", "plain", "--tag", tag, *lgb_extra]),
            (f"lgbm_pa_{tag}a_LTFM.txt", ["src_v3.train_per_airport_v89", "anchored", "--tag", tag, *lgb_extra]),
            (f"catboost_r_all_{tag}ca.cbm", ["src_v3.catboost_base", "--tag", f"{tag}ca", *cat_extra,
                                             "--anchor-col", "mvt_aobt3"])]
    jobs = [j for j in jobs if per_airport or j[1][0] != "src_v3.train_per_airport_v89"]
    missing = [cmd for out, cmd in jobs if not (C.ROOT / "models" / out).exists()]
    log.info("%d of %d members to fit", len(missing), len(jobs))
    for i in range(0, len(missing), 3):
        group(*missing[i:i + 3])


def serve(tag: str, out: str, per_airport: bool) -> list[str]:
    rank = ["models/plan_nm_taxi_rank_v65.parquet", "models/nm_clocks_rank_v73.parquet",
            "models/nm_forward_rank_v85.parquet", "models/opdi_own_rank_v90.parquet",
            "external/adsblol/adsb_clock_rank.parquet"]
    base = f"kind-mango_{tag}_base.parquet"
    per = ["--per-airport-tag", tag, "--per-airport-weight", "0.5"] if per_airport else []
    subprocess.run([sys.executable, "-m", "src_v3.predict_v67", "--weight", "0.5", "--ms3", "--base-model", f"lgbm_r_all_{tag}",
                    "--catboost-tag", f"{tag}c", "--catboost-anchor-tag", f"{tag}ca", "--anchor-model", f"lgbm_r_all_{tag}a",
                    "--anchor-weight", "0.5", *per, "--extra", *rank,
                    "--pre-ms", f"kind-mango_{tag}_base_pre_ms.parquet", "--out", base], check=True, cwd=C.ROOT)
    frozen = C.ADSB_DIR / tag / "adsb_clock_train.parquet"
    frozen.parent.mkdir(exist_ok=True)
    shutil.copy(C.ADSB_DIR / "adsb_clock_train.parquet", frozen)
    stack = ["--sources", "both", "--stand-bound", "--mlp-weight", str(C.AUTO_MLP_WEIGHT), "--rounds", str(C.STACK_V90_ROUNDS),
             "--adsb-train", frozen.relative_to(C.ROOT).as_posix(),
             "--blend-holdout", blend_path(0.5 if per_airport else 0.0).relative_to(C.ROOT).as_posix()]
    subprocess.run([sys.executable, "-m", "src_v3.stack_pushback_v90", "serve", *stack, "--base", base, "--out", out],
                   check=True, cwd=C.ROOT)
    return stack


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait-pid", type=int, required=True)
    ap.add_argument("--ref", default="kind-mango_v95.parquet")
    ap.add_argument("--prefit-pids", type=int, nargs="*", default=[],
                    help="Chains that fit members ahead of the price; the refit waits for them, then fills the gaps.")
    ap.add_argument("--no-per-airport", action="store_true",
                    help="Serve without per-airport members (2026-10-03: they could not finish by the morning).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(C.ROOT / "logs" / "release_v97.log"), logging.StreamHandler()])
    keep_awake()
    while is_running(args.wait_pid):
        time.sleep(C.AUTO_POLL_S)
    per_airport = not args.no_per_airport
    price = json.loads((C.MODELS / "adsb_base_v97.holdout.json").read_text()) if per_airport else measure_price(0.0)
    log.info("hold-out price %s", price)
    if price["blend_and_stack_v97_minus_v89"] > -C.AUTO_BAR_MSE:
        raise SystemExit(f"Blend plus stack gains {price['blend_and_stack_v97_minus_v89']:.0f} MSE, under the "
                         f"{C.AUTO_BAR_MSE} MSE bar. No refit, no upload.")
    while any(is_running(p) for p in args.prefit_pids):
        time.sleep(C.AUTO_POLL_S)
    refit("v97", per_airport)
    out = "kind-mango_v97.parquet"
    stack = serve("v97", out, per_airport)
    checks = gates(out, args.ref)
    result = upload(out)
    (SUB / "kind-mango_v97.result.json").write_text(json.dumps(result, indent=1))
    (SUB / "kind-mango_v97.recipe.json").write_text(json.dumps(
        {"holdout_price": price, "stack_args": stack, "gates": checks, "score": result["score"]}, indent=1))
    log.info("done: %s scored %.4f s", out, result["score"])


if __name__ == "__main__":
    main()
