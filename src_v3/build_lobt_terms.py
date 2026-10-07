"""Unclipped NM window terms for the neural members (nineteenth pass, J family).

`build_nm_clocks_v73` clips `MVT - LOBT` at `NM_CLOCK_CLIP_S`, so no member sees where the NM join window lies on
rows whose `LOBT` is hours before the take-off. This module writes `lobt_ml` (`MVT - LOBT`) and `lobt_sl`
(`SCHED - LOBT`) in seconds, unclipped, for the 2025 training files and the ranking file. Rows without `LOBT`
stay NaN.
"""
from __future__ import annotations

import logging

import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep

OUT_TRAIN = C.ROOT / "models" / "lobt_terms_train.parquet"
OUT_RANK = C.ROOT / "models" / "lobt_terms_rank.parquet"
log = logging.getLogger(__name__)


def terms(paths: list) -> pd.DataFrame:
    """`lobt_ml` and `lobt_sl` in seconds of every departure in `paths`, keyed on `MVT_ID_mvt`."""
    d = load_dep(paths)
    return pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].values,
                         "lobt_ml": (d["MVT_TIME_UTC_mvt"] - d["LOBT_flt"]).dt.total_seconds().values,
                         "lobt_sl": (d["SCHED_TIME_UTC_mvt"] - d["LOBT_flt"]).dt.total_seconds().values})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for paths, out in ((sorted(C.TRAIN_DIR.glob("training_*.parquet")), OUT_TRAIN), ([C.RANK], OUT_RANK)):
        t = terms(paths)
        t.to_parquet(out)
        log.info("%s: %d rows, LOBT on %.4f, MVT - LOBT above 7,200 s on %d", out.name, len(t),
                 t["lobt_ml"].notna().mean(), int((t["lobt_ml"] > C.NM_CLOCK_CLIP_S).sum()))


if __name__ == "__main__":
    main()
