"""v98: the v97 members with serve-time parameters chosen on the 2025 hold-out, released after v97 scores.

1. Blend shares (CatBoost, anchored, per-airport) on the v97 hold-out members: a grid point replaces the v97
   shares only if it is better in January and in July alone.
2. On that blend, MLP weight and stack rounds, priced end to end (blend, stack, A5 bound) with the
   January-July cross-fit; the best total wins.
3. Serve with `predict_v67` from the v97 full-year members, stack, gate against v97, upload.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3 import stack_pushback_v90 as sp
from src_v3.auto_release import gates, is_running, keep_awake, upload
from src_v3.fit_blend_v77 import blend
from src_v3.measure_adsb_base_v97 import out_path
from src_v3.measure_catboost_blend import holdout_rows
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.tune_stack_v93 import frame as stack_frame

SUB = C.ROOT / "submission"
BLEND_OUT = C.MODELS / "blend_v98_holdout.parquet"
log = logging.getLogger(__name__)


def holdout_blend(cat: float, anchor: float, per: float) -> np.ndarray:
    m = {k: pd.read_parquet(out_path(k))["pred"].values for k in ("lgb", "lgb_anchor", "lgb_per_airport", "cat", "cat_anchor")}
    lgb = (1 - per) * m["lgb"] + per * m["lgb_per_airport"]
    return blend({"lgb": lgb, "lgb_anchor": m["lgb_anchor"], "cat": m["cat"], "cat_anchor": m["cat_anchor"]},
                 cat, anchor, anchor)


def pick_blend() -> tuple[tuple, dict]:
    """The grid point with the lowest MSE among those better than the v97 shares in both months."""
    y = pd.read_parquet(PRED_OUT)["y"].values
    jan = holdout_rows()[0]["month"].values == 1  # same row order as PRED_OUT (checked in measure_member_v77)
    base = holdout_blend(*C.V98_DEFAULT)
    scores = {}
    for point in product(C.V98_CAT_SHARES, C.V98_ANCHOR_SHARES, C.V98_PER_AIRPORT_SHARES):
        p = holdout_blend(*point)
        scores[point] = {"all": mse_delta(p, base, y), "jan": mse_delta(p[jan], base[jan], y[jan]),
                         "jul": mse_delta(p[~jan], base[~jan], y[~jan])}
    ok = {k: v for k, v in scores.items() if v["jan"] < 0 and v["jul"] < 0}
    best = min(ok, key=lambda k: ok[k]["all"]) if ok else C.V98_DEFAULT
    log.info("blend shares %s: %s", best, scores.get(best, "default"))
    return best, {str(k): v for k, v in scores.items()}


def pick_stack(point: tuple) -> tuple[tuple, dict]:
    """MLP weight and stack rounds by end-to-end hold-out MSE on the chosen blend."""
    ref = pd.read_parquet(PRED_OUT)
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "y": ref["y"].values,
                  "base": holdout_blend(*point)}).to_parquet(BLEND_OUT)
    totals = {}
    for mlp in C.V98_MLP_WEIGHTS:
        rows = stack_frame(mlp, False, blend_path=BLEND_OUT)
        for rounds in C.V98_STACK_ROUNDS:
            s, _ = sp.crossfit(rows, rounds=rounds)
            served = np.fmax(s["stack"].values, sp.stand_bound(s))
            totals[(mlp, rounds)] = float(((rows["base"] - rows["y"]) ** 2).sum() - ((s["base"] - s["y"]) ** 2).sum()
                                          + ((served - s["y"]) ** 2).sum()) / C.HOLDOUT_ROWS
    best = min(totals, key=totals.get)
    log.info("stack settings %s; totals %s", best, totals)
    return best, {str(k): v for k, v in totals.items()}


def serve(point: tuple, mlp: float, rounds: int, out: str) -> list[str]:
    cat, anchor, per = point
    rank = ["models/plan_nm_taxi_rank_v65.parquet", "models/nm_clocks_rank_v73.parquet",
            "models/nm_forward_rank_v85.parquet", "models/opdi_own_rank_v90.parquet",
            "external/adsblol/adsb_clock_rank.parquet"]
    base = "kind-mango_v98_base.parquet"
    per_args = ["--per-airport-tag", "v97", "--per-airport-weight", str(per)] if per else []
    subprocess.run([sys.executable, "-m", "src_v3.predict_v67", "--weight", str(cat), "--ms3", "--base-model", "lgbm_r_all_v97",
                    "--catboost-tag", "v97c", "--catboost-anchor-tag", "v97ca", "--anchor-model", "lgbm_r_all_v97a",
                    "--anchor-weight", str(anchor), *per_args,
                    "--extra", *rank, "--pre-ms", "kind-mango_v98_base_pre_ms.parquet", "--out", base], check=True, cwd=C.ROOT)
    stack = ["--sources", "both", "--stand-bound", "--mlp-weight", str(mlp), "--rounds", str(rounds),
             "--adsb-train", "external/adsblol/v97/adsb_clock_train.parquet",
             "--blend-holdout", BLEND_OUT.relative_to(C.ROOT).as_posix()]
    subprocess.run([sys.executable, "-m", "src_v3.stack_pushback_v90", "serve", *stack, "--base", base, "--out", out],
                   check=True, cwd=C.ROOT)
    return stack


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="kind-mango_v97.parquet")
    ap.add_argument("--v97-pid", type=int, required=True, help="Process of release_v97; v98 ends if it ends without a score.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(C.ROOT / "logs" / "release_v98.log"), logging.StreamHandler()])
    keep_awake()
    while not (SUB / "kind-mango_v97.result.json").exists():
        if not is_running(args.v97_pid):
            raise SystemExit("release_v97 ended without a score (bar, gate or error); v98 builds on v97, so it stops too.")
        time.sleep(C.AUTO_POLL_S)
    point, blend_scores = pick_blend()
    (mlp, rounds), stack_totals = pick_stack(point)
    if (point, mlp, rounds) == (C.V98_DEFAULT, C.AUTO_MLP_WEIGHT, C.STACK_V90_ROUNDS):
        raise SystemExit("Every tweak priced worse than the v97 settings; v98 would equal v97. No upload.")
    out = "kind-mango_v98.parquet"
    stack = serve(point, mlp, rounds, out)
    checks = gates(out, args.ref)
    result = upload(out)
    (SUB / "kind-mango_v98.result.json").write_text(json.dumps(result, indent=1))
    (SUB / "kind-mango_v98.recipe.json").write_text(json.dumps(
        {"blend_shares": point, "mlp_weight": mlp, "stack_rounds": rounds, "stack_args": stack, "blend_scores": blend_scores,
         "stack_totals": stack_totals, "gates": checks, "score": result["score"]}, indent=1))
    log.info("done: %s scored %.4f s", out, result["score"])


if __name__ == "__main__":
    main()
