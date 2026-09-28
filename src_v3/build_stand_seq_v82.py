"""Stand sequence (v81): the first arrival that blocks in at the departure's stand.

An arrival can block in at a stand only after the departure there has
pushed back, so the in-block time of the next arrival at the same
(airport, stand) bounds the off-block from above. On 2025 departures the
bound `MVT - next in-block` is positive on 10 % of rows and exceeds the
label by more than 60 s on 0.09 % of them. The next arrival is the first
with in-block at or after `AOBT_3 - 600 s` (or `MVT - 1,800 s` without
`AOBT_3`). Arrival in-block times are present in the 2025 training files
and in the 2026 ranking file; both go through the same code.

v82 adds the taxi-in and the landing time of that next arrival (a long
taxi-in means it waited for the stand), and the previous arrival at the
stand, whose in-block bounds the off-block from below (`MVT - previous
in-block` bounds the taxi-out from above).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src_v3 import config as C

OUT_TRAIN = C.ROOT / "models" / "stand_seq_train_v82.parquet"
OUT_RANK = C.ROOT / "models" / "stand_seq_rank_v82.parquet"
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"]
FEATURES = ["nxt_arr_lb", "nxt_arr_after_aobt3", "nxt_arr_taxiin", "nxt_arr_land_lb", "prv_arr_ub"]
AOBT_MARGIN_S, NO_AOBT_LOOKBACK_S = 600, 1800
log = logging.getLogger(__name__)


def seconds(ts: pd.Series) -> pd.Series:
    return ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype(float)


def stand_seq(paths: list) -> pd.DataFrame:
    """The two stand-sequence columns of every departure in `paths`, keyed on `MVT_ID_mvt`."""
    m = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    is_dep = m["PHASE_mvt"] == "DEP"
    m["key"] = np.where(is_dep, m["ADEP_mvt"], m["ADES_mvt"]) + "|" + m["STAND_mvt"].astype(str)
    arr = m[~is_dep & m["STAND_mvt"].notna() & m["BLOCK_TIME_UTC_mvt"].notna()]
    arr = pd.DataFrame({"key": arr["key"], "tn": seconds(arr["BLOCK_TIME_UTC_mvt"]),
                        "land": seconds(arr["MVT_TIME_UTC_mvt"]), "taxiin": arr["TAXITIME_SEC_mvt"]}).sort_values("tn")
    dep = m[is_dep].copy()
    dep["tm"], dep["ta"] = seconds(dep["MVT_TIME_UTC_mvt"]), seconds(dep["AOBT_3_flt"])
    dep["q"] = dep["ta"].sub(AOBT_MARGIN_S).fillna(dep["tm"] - NO_AOBT_LOOKBACK_S)
    has_stand = dep["STAND_mvt"].notna() & dep["q"].notna()
    j = pd.merge_asof(dep[has_stand].sort_values("q"), arr, left_on="q", right_on="tn", by="key", direction="forward")
    dep["q_prev"] = dep["ta"].add(AOBT_MARGIN_S).fillna(dep["tm"])
    prev = pd.merge_asof(dep[has_stand].sort_values("q_prev"), arr[["key", "tn"]].rename(columns={"tn": "tp"}),
                         left_on="q_prev", right_on="tp", by="key", direction="backward")
    clip = lambda x: x.clip(-C.NM_CLOCK_CLIP_S, C.NM_CLOCK_CLIP_S).values
    out = pd.DataFrame({"MVT_ID_mvt": j["MVT_ID_mvt"].values,
                        "nxt_arr_lb": clip(j["tm"] - j["tn"]),
                        "nxt_arr_after_aobt3": clip(j["tn"] - j["ta"]),
                        "nxt_arr_taxiin": j["taxiin"].clip(0, C.NM_CLOCK_CLIP_S).values,
                        "nxt_arr_land_lb": clip(j["tm"] - j["land"])})
    out = out.merge(pd.DataFrame({"MVT_ID_mvt": prev["MVT_ID_mvt"].values,
                                  "prv_arr_ub": (prev["tm"] - prev["tp"]).clip(-C.NM_CLOCK_CLIP_S, 86_400).values}),
                    on="MVT_ID_mvt", how="left", validate="1:1")
    out = dep[["MVT_ID_mvt"]].merge(out, on="MVT_ID_mvt", how="left", validate="1:1")
    log.info("%d departures; nxt_arr_lb coverage %.3f, share > 0 %.3f", len(out),
             out["nxt_arr_lb"].notna().mean(), (out["nxt_arr_lb"] > 0).mean())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stand_seq(sorted(C.TRAIN_DIR.glob("training_*.parquet"))).to_parquet(OUT_TRAIN)
    stand_seq([C.RANK]).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
