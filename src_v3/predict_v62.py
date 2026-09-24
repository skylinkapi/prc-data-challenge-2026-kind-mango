"""v62 = v57 + MS3 (WINNING_PLAN L13).

Where the three v57 base members differ by more than 3,600 s outside LIRF,
the base term is the member median, not the mean. The median-minus-mean
delta goes onto the v57 pre-MS output. MS1 and MS2 then apply as in v57.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict_v30 import load_training_categories
from train_lgbm_v21 import CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS

from src_v3 import config as C
from src_v3.postprocess import apply_ms3
from src_v3.predict_v57 import serve_v30, write_ms

BASE = "lgbm_r_all_v57"
log = logging.getLogger(__name__)


def score_members(dump: pd.DataFrame, base: str = BASE) -> np.ndarray:
    """Score each base member on the served frame, clipped at 0 as in predict_v30."""
    feat = (C.ROOT / "models" / f"{base}.features.txt").read_text().splitlines()
    cats = load_training_categories()
    x = dump[feat].copy()
    for c in CAT_COLS:
        x[c] = pd.Categorical(x[c], categories=cats[c])
    return np.array([
        np.clip(lgb.Booster(model_file=str(C.ROOT / "models" / f"{base}_s{s}.txt"))
                .predict(x), 0, None)
        for s in DEFAULT_SEEDS])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre-ms", default="kind-mango_v62_pre_ms.parquet")
    ap.add_argument("--out", default="kind-mango_v62.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    dump_path = C.MODELS / "v57_rank_features.parquet"
    serve_v30(args.pre_ms, BASE, dump_features=str(dump_path))
    dump = pd.read_parquet(dump_path)
    members = score_members(dump)
    adep = dump["ADEP_mvt"].astype(str).values
    pre = pd.read_parquet(C.ROOT / "submission" / args.pre_ms)
    served = dump[["MVT_ID_mvt"]].merge(pre, on="MVT_ID_mvt", how="left",
                                        validate="1:1")["TAXITIME_SEC_mvt"].values
    gap = np.abs(members.mean(axis=0) - served)[adep != "LIRF"].max()
    if gap > 1e-6:
        raise AssertionError(f"Member scoring does not match the served base: max gap {gap:.3f} s. "
                             "Check the categorical cast, then rerun.")
    ms3, moved = apply_ms3(members, adep)
    log.info("MS3 moved %s", moved)

    delta = pd.DataFrame({"MVT_ID_mvt": dump["MVT_ID_mvt"].values,
                          "delta": ms3 - members.mean(axis=0)})
    pre = pre.merge(delta, on="MVT_ID_mvt", how="left", validate="1:1")
    pre["TAXITIME_SEC_mvt"] += pre.pop("delta").fillna(0)
    write_ms(pre, args.out)
    (C.MODELS / "predict_v62.json").write_text(json.dumps(moved, indent=1))


if __name__ == "__main__":
    main()
