"""Price stack variants for v93 on the 2025 hold-out: capacity, MLP share of the base, A4 columns.

Every arm reads the same frame and the same cross-fit as `stack_pushback_v90
price` (months once each month holds `STACK_V90_MIN_FOLD_DAYS` ADS-B days,
else day parity). The report adds the A5 stand bound on top of each arm.
"""
from __future__ import annotations

import json
import logging
from itertools import product

import pandas as pd

from src_v3 import config as C
from src_v3 import stack_pushback_v90 as sp
from src_v3.build_nm_clocks_v73 import OUT_TRAIN as NM_TRAIN, load_dep

REPORT = C.MODELS / "tune_stack_v93.holdout.json"
MLP_REPORT = C.MODELS / "mlp_member.holdout.json"
log = logging.getLogger(__name__)


def frame(mlp_weight: float, neighbours: bool, extra: tuple = (), adsb_train=sp.ADSB_TRAIN,
          blend_path=sp.BLEND_HOLDOUT, mlp_holdout=sp.MLP_HOLDOUT) -> pd.DataFrame:
    rows = pd.read_parquet(blend_path)
    if mlp_weight:
        rows["base"] = (1 - mlp_weight) * rows["base"] + mlp_weight * pd.read_parquet(mlp_holdout)["pred"].values
    paths = (NM_TRAIN, sp.OPDI_TRAIN, adsb_train, *((sp.NB_TRAIN,) if neighbours else ()), *extra)
    dep = load_dep(sorted(C.TRAIN_DIR.glob("training_2025-0[17]-*.parquet")), ["CALLSIGN_flt"])
    return sp.frame(rows.dropna(subset=["MVT_ID_mvt"]), paths, dep)


def main() -> None:
    """`total` = MLP gain of the base on every row outside LIRF + stack gain + stand-bound gain."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mlp_gain = {float(k): v for k, v in json.loads(MLP_REPORT.read_text())["weights"].items()} | {0.0: 0.0}
    out = []
    for mlp, nb in product(C.STACK_V93_MLP_WEIGHTS, (False, True)):
        f = frame(mlp, nb)
        for leaves, n in product(C.STACK_V93_LEAVES, C.STACK_V93_ROUNDS):
            r = sp.price(f, {**C.STACK_V90_PARAMS, "num_leaves": leaves}, n)
            arm = {"mlp_weight": mlp, "neighbours": nb, "leaves": leaves, "rounds": n, "fold": r["fold"],
                   "rows": r["rows"], "stack": r["all"], "stand_bound": r["stand_bound"]["mse"],
                   "total": mlp_gain[mlp] + r["all"] + r["stand_bound"]["mse"]}
            log.info("%s", arm)
            out.append(arm)
    REPORT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
