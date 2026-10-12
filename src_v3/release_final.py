"""Build the file of the final phase: the v132 recipe on the organiser files in `submission/`.

Work in a copy of the repository that holds the trained artefacts of `models/v132.manifest.json`. Put the
organiser file with the movements at `submission/ranking.parquet` and the template at
`submission/submitting.parquet`. `tables` deletes each ranking-side table, then rebuilds it from the ranking
file and the open data. `serve` scores the members, the stack and the rules, and writes
`submission/<name>.parquet`. On the files of the ranking phase the two commands rebuild
`kind-mango_v132.parquet`. `--also` names the ranking file of the first phase for the timestamp-precision rule
(`mvt_precision --also`). The script does not upload.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.mlp_member import COLUMN_SETS, RANK_OVERRIDES, out_paths
from src_v3.predict_v67 import with_adsb

A = "external/adsblol"
STANDS = "external/xplane/stands_osm_xplane.parquet"
ADSB = ["-m", "src_v3.build_adsb_clock"]
OPDI_V90 = ["models/opdi_own_train_v90.parquet", "models/opdi_own_rank_v90.parquet"]
OPDI_V127 = ["models/opdi_own_train_v127.parquet", "models/opdi_own_rank_v127.parquet"]
TABLES = [   # command, ranking-side outputs
    (["-m", "src_v2.cli", "build"], ["models/v2/cache/frame_rank.parquet"]),
    (["src/build_tempo_p2575.py"], ["models/tempo_p2575_rank.parquet"]),
    (["-m", "src_v3.build_plan_taxi_res_v57"], ["models/plan_taxi_res_rank_v57.parquet"]),
    (["-m", "src_v3.build_plan_nm_taxi_v65"], ["models/plan_nm_taxi_rank_v65.parquet"]),
    (["-m", "src_v3.build_nm_clocks_v73"], ["models/nm_clocks_rank_v73.parquet"]),
    (["-m", "src_v3.build_nm_forward_v85"], ["models/nm_forward_rank_v85.parquet", "models/nm_forward_rank_v86.parquet"]),
    (["-m", "src_v3.build_opdi_own"], [OPDI_V90[1]]),
    (["-m", "src_v3.build_opdi_own", "--link-flight", "--keep", *OPDI_V90, "--out", *OPDI_V127], [OPDI_V127[1]]),
    (ADSB, [f"{A}/adsb_clock_rank.parquet"]),
    ([*ADSB, "--stands", STANDS, "--out-dir", f"{A}/v116"], [f"{A}/v116/adsb_clock_rank.parquet"]),
    ([*ADSB, "--stands", STANDS, "--out-dir", f"{A}/v127", "--link-flight", "--keep", f"{A}/v116"],
     [f"{A}/v127/adsb_clock_rank.parquet"]),
    ([*ADSB, "--stands", STANDS, "--out-dir", f"{A}/v130", "--link-flight", "--link-takeoff", "--keep", f"{A}/v127",
      "--keep-linked"],
     [f"{A}/v130/adsb_clock_rank.parquet"]),
    (["-m", "src_v3.build_arr_artefact"], ["models/arr_art_rank_g1.parquet"]),
    (["-m", "src_v3.build_arr_link_v83"], ["models/arr_link_rank_v83.parquet"]),
    (["-m", "src_v3.build_gate_g3"], ["models/arr_art_op_link_rank_g3.parquet"]),
    (["-m", "src_v3.build_atfm_slot"], ["models/atfm_slot_rank.parquet"]),
    (["-m", "src_v3.build_lobt_terms"], ["models/lobt_terms_rank.parquet"]),
    (["-m", "src_v3.build_struct_terms"], ["models/struct_terms_rank.parquet"]),
]
RANK = ["models/plan_nm_taxi_rank_v65.parquet", "models/nm_clocks_rank_v73.parquet", "models/nm_forward_rank_v85.parquet",
        OPDI_V90[1], f"{A}/v130/adsb_clock_rank.parquet", "models/arr_art_op_link_rank_g3.parquet"]
TABM = ("tabm_s_k8", "tabm_j_k8", "tabm_plr_j_k8", "tabm_v116_k8", "tabm_plr_k8")
DUMP_V130 = C.MODELS / "lgbm_r_all_v130_rank_features.parquet"
DUMP_V97 = COLUMN_SETS["v97"][2]
log = logging.getLogger(__name__)


def run(cmd: list[str], gpu: bool = False) -> None:
    env = {**os.environ, "PYTHONPATH": f"{C.GPU_SITE}{os.pathsep}{C.ROOT}"} if gpu else None
    log.info("run %s", " ".join(cmd))
    subprocess.run([sys.executable, *cmd], check=True, cwd=C.ROOT, env=env)


def tables(first: int) -> None:
    """Rebuild the ranking-side tables from step `first`; a deleted output cannot serve as a cache."""
    for i, (cmd, outs) in enumerate(TABLES):
        if i < first:
            continue
        for out in outs:
            (C.ROOT / out).unlink(missing_ok=True)
        run(cmd)
        for out in outs:
            log.info("step %d wrote %s: %d rows", i, out, len(pd.read_parquet(C.ROOT / out, columns=["MVT_ID_mvt"])))


def members(name: str) -> None:
    """Base file, then the MLP and the TabM families from their saved fits on the v97 feature dump."""
    run(["-m", "src_v3.predict_v67", "--weight", "0.5", "--ms3", "--base-model", "lgbm_r_all_v130", "--catboost-tag", "v130c",
         "--catboost-anchor-tag", "v97ca", "--anchor-model", "lgbm_r_all_v97a", "--anchor-weight", "0.5",
         "--per-airport-tag", "v130", "--per-airport-anchor-tag", "v97a", "--per-airport-weight", "0.5", "--pa-guard",
         "--anchor-adsb", f"{A}/adsb_clock_rank.parquet", "--gate-booster", "lgbm_p_fb_lirf_g3.txt",
         "--gate-iso", "lirf_regime_g3.isotonic.pkl", "--gate-features", "lirf_regime_g3.features.txt",
         "--copy-window", "--copy-linked", "--extra", *RANK,
         "--pre-ms", f"{name}_base_pre_ms.parquet", "--out", f"{name}_base.parquet"])
    with_adsb(pd.read_parquet(DUMP_V130), f"{A}/adsb_clock_rank.parquet").to_parquet(DUMP_V97)
    run(["-m", "src_v3.mlp_member", "predict", "--columns", "v97", "--dump", str(DUMP_V97), "--out", str(out_paths("v97")[1])])
    for tag in TABM:
        over = [str(p) for p in RANK_OVERRIDES[C.TABM_SPECS[tag]["columns"]]]
        run(["-m", "src_v3.tabm_member", "predict", "--tag", tag, "--dump", str(DUMP_V97), "--overrides", *over,
             "--out", str(out_paths(tag)[1])], gpu=True)


def stack(name: str, also: list[str], adsb: str, tabm: tuple) -> None:
    """Neural term, stack and the three rules; the stack reads the ADS-B tables in `adsb`."""
    run(["-m", "src_v3.tabm_member", "combine", "--tags", *[t for t in tabm for _ in (0, 1)], "--share", "0.65"], gpu=True)
    run(["-m", "src_v3.stack_pushback_v90", "serve", "--sources", "both", "--stand-bound", "--rounds", "400", "--leaves", "15",
         "--min-data", "100", "--l2", "50", "--coverage-weights", "--adsb-train", f"{adsb}/adsb_clock_train.parquet",
         "--adsb-rank", f"{adsb}/adsb_clock_rank.parquet", "--opdi-train", OPDI_V127[0], "--opdi-rank", OPDI_V127[1],
         "--blend-holdout", "models/v3/blend_v130_holdout.parquet", "--mlp-weight", "0.72", "--mlp-columns", "nn",
         "--base", f"{name}_base.parquet", "--out", f"{name}_stack.parquet"])
    run(["-m", "src_v3.step_a_date", f"{name}_stack.parquet", f"{name}_stepa.parquet"])
    run(["-m", "src_v3.lobt_window", f"{name}_stepa.parquet", f"{name}_window.parquet"])
    run(["-m", "src_v3.mvt_precision", "serve", f"{name}_window.parquet", f"{name}.parquet", *(["--also", *also] if also else [])])


def check(name: str) -> None:
    """Format checks of the scorer: template rows in template order, float64, no NaN, no negative value."""
    out, tmpl = pd.read_parquet(C.ROOT / "submission" / f"{name}.parquet"), pd.read_parquet(C.SUB_TMPL)
    p = out["TAXITIME_SEC_mvt"]
    fmt = {"template_ids": bool(np.array_equal(out["MVT_ID_mvt"].values, tmpl["MVT_ID_mvt"].values)),
           "float64": str(p.dtype) == "float64", "no_nan": not p.isna().any(), "non_negative": bool((p >= 0).all())}
    log.info("%s: %d rows, mean %.1f s, %d rows above 7,200 s, format %s", name, len(out), p.mean(), int((p > 7200).sum()), fmt)
    if not all(fmt.values()):
        raise SystemExit(f"Format checks failed: {[k for k, v in fmt.items() if not v]}. The scorer rejects the file. "
                         "Fix the cause, then rerun `serve`.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("tables", "serve"))
    ap.add_argument("--name", default="kind-mango_final", help="serve: stem of the output files in `submission/`.")
    ap.add_argument("--first", type=int, default=0, help="tables: index of the first step to run, after a stopped run.")
    ap.add_argument("--stack-only", action="store_true", help="serve: reuse the base file and the member files.")
    ap.add_argument("--also", nargs="+", default=[], help="serve: other organiser releases of the same movements (P1).")
    ap.add_argument("--stack-adsb", default=f"{A}/v130", help="serve: folder of the ADS-B tables that the stack reads.")
    ap.add_argument("--tabm", nargs="+", default=list(TABM), help="serve --stack-only: TabM families of the neural term.")
    args = ap.parse_args()
    if args.stage == "tables":
        tables(args.first)
        return
    if not args.stack_only:
        members(args.name)
    stack(args.name, args.also, args.stack_adsb, tuple(args.tabm))
    check(args.name)


if __name__ == "__main__":
    main()
