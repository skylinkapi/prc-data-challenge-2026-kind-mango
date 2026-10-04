"""Price a refit plain LightGBM hold-out member inside the v97 blend plus stack.

The candidate file holds hold-out predictions in the row order of
`PRED_OUT`, for example `measure_adsb_base_v97 fit lgb --tag k3 --extra ...`
(third-edition K3). The v97 blend is rebuilt with the candidate in place of
the plain member; the per-airport, anchored and CatBoost members stay. The
MLP at 0.15, the cross-fitted v95 stack and the A5 bound follow, as v97
serves them. The report gives the member and the blend-plus-stack change on
the hold-out scale.
"""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from src_v3 import config as C
from src_v3.fit_blend_v77 import blend
from src_v3.measure_adsb_base_v97 import MEMBERS, out_path
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_pseudo_n5 import with_stack
from src_v3.measure_tail_rules import mse_delta

log = logging.getLogger(__name__)


def v97_blend(lgb, m: dict) -> pd.Series:
    return blend({"lgb": 0.5 * lgb + 0.5 * m["lgb_per_airport"], "lgb_anchor": m["lgb_anchor"],
                  "cat": m["cat"], "cat_anchor": m["cat_anchor"]}, *C.BLEND_V77_SERVED)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate", help="Hold-out prediction file of the candidate plain member.")
    ap.add_argument("--tag", required=True, help="Name of the report, models/v3/swap_<tag>.holdout.json.")
    args = ap.parse_args()
    ref = pd.read_parquet(PRED_OUT)
    y = ref["y"].values
    m = {k: pd.read_parquet(out_path(k))["pred"].values for k in MEMBERS}
    new = pd.read_parquet(C.ROOT / args.candidate)["pred"].values
    paths = {}
    for name, lgb in (("v97", m["lgb"]), (args.tag, new)):
        paths[name] = C.MODELS / f"blend_{name}_swap_holdout.parquet"
        pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "y": y, "base": v97_blend(lgb, m)}).to_parquet(paths[name])
    old, cand = with_stack(paths["v97"]), with_stack(paths[args.tag])
    report = {"member_minus_v97": mse_delta(new, m["lgb"], y),
              "blend_and_stack_minus_v97": mse_delta(cand["base"].values, old.loc[cand.index, "base"].values, cand["y"].values)}
    (C.MODELS / f"swap_{args.tag}.holdout.json").write_text(json.dumps(report, indent=1))
    log.info("%s", json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
