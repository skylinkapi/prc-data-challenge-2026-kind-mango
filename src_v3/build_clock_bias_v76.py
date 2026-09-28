"""Clock bias (v76): the median error of the NM clock per stand, runway and operator.

`y - (MVT - AOBT_3)` is the error of the scored flight's clock. Per airport
it has a median of -296 to +33 s, and the stand, the runway and the
operator shift it further: leave-one-month-out medians per (airport,
stand, runway) cut its MSE on 2025 clean rows from 113,086 to 103,432.
The map uses clean rows (label and clock in 30 to 7,200 s and 0 to 7,200 s)
and keys with at least `CLOCK_BIAS_MIN_COUNT` rows. Training rows read the
map of the other months (leave one month out), so that no row sees its
own label; ranking rows read the 12-month map. An in-sample map cost
+1,531 MSE on the hold-out.
"""
from __future__ import annotations

import logging

import pandas as pd

from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep

OUT_TRAIN = C.ROOT / "models" / "clock_bias_train_v76.parquet"
OUT_RANK = C.ROOT / "models" / "clock_bias_rank_v76.parquet"
KEYS = {"cbias_stand": ["ADEP_mvt", "STAND_mvt"],
        "cbias_stand_rwy": ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
        "cbias_op": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"]}
FEATURES = list(KEYS)
COLS = ["STAND_mvt", "AIRCRAFT_OPERATOR_flt", "TAXITIME_SEC_mvt"]
log = logging.getLogger(__name__)


def load_rows(paths: list) -> pd.DataFrame:
    """Departure rows with the key columns, the month and the clock error on clean rows."""
    dep = load_dep(paths, extra=COLS)
    clock = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    y = dep["TAXITIME_SEC_mvt"]
    dep["err"] = (y - clock).where(y.between(30, C.Y_CLEAN_MAX) & clock.between(0, C.NM_CLOCK_CLIP_S))
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.month
    return dep


def bias_columns(rows: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    """Key medians of the clock error fitted on `fit`, mapped onto `rows`, keyed on `MVT_ID_mvt`."""
    out = pd.DataFrame({"MVT_ID_mvt": rows["MVT_ID_mvt"].values})
    for name, key in KEYS.items():
        g = fit.dropna(subset=["err"]).groupby(key)["err"].agg(["median", "count"])
        g = g.loc[g["count"] >= C.CLOCK_BIAS_MIN_COUNT, "median"].clip(-C.CLOCK_BIAS_CLIP_S, C.CLOCK_BIAS_CLIP_S)
        out[name] = rows[key].merge(g.rename(name), left_on=key, right_index=True, how="left")[name].values
        log.info("%s: %d keys, coverage %.3f", name, len(g), out[name].notna().mean())
    return out


def bias_oof(rows: pd.DataFrame) -> pd.DataFrame:
    """Bias columns of each month from the map of the other months in `rows`."""
    return pd.concat([bias_columns(rows[rows["month"] == m], rows[rows["month"] != m])
                      for m in sorted(rows["month"].unique())], ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train = load_rows(sorted(C.TRAIN_DIR.glob("training_*.parquet")))
    bias_oof(train).to_parquet(OUT_TRAIN)
    bias_columns(load_rows([C.RANK]), train).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
