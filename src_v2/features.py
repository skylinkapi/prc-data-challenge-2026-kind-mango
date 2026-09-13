"""Feature list assembly. One function returns the columns the base sees;
another returns the columns the regime and tail heads see. Everything is
computed in `src_v2.frame` and `src_v2.encoders`; this module only lists.
"""
from __future__ import annotations

from src_v2 import encoders as E

BASE_ANCHOR = [
    "sd", "sd_mod_86400", "mvt_eobt1", "mvt_iobt", "mvt_lobt",
    "eobt1_minus_iobt", "sd_minus_mvt_eobt1",
    "eobt1_equals_sched", "flt_null",
    "plan_block", "plan_taxi_res",
]

BASE_TEMPO = [
    "nb_eobt_med_apt30", "nb_eobt_mean_apt30", "nb_eobt_cnt_apt30",
    "nb_eobt_med_apt60", "nb_eobt_mean_apt60", "nb_eobt_cnt_apt60",
    "nb_eobt_med_rwy30", "nb_eobt_mean_rwy30", "nb_eobt_cnt_rwy30",
    "order_later_eobt_30", "order_total_30",
    "stand_gap", "queue_eobt_mvt",
]

BASE_TIME = ["hour", "dow", "month"]

BASE_CAT = [
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
    "flt_prefix", "stand_prefix",
]

BASE_WEATHER = ["tmpc", "vis_km", "wind_kt", "gust_kt"]


def base_feature_list() -> list[str]:
    enc = [f"enc_{k}_{s}" for k in E.ENCODE_KEYS
           for s in ("med", "iqr", "cnt", "fbrate")]
    return BASE_ANCHOR + BASE_TEMPO + BASE_TIME + BASE_CAT + BASE_WEATHER + enc


def regime_feature_list() -> list[str]:
    """C1. Regime head reads anchor deltas, categoricals, hour, fb-rate encoders."""
    enc = [f"enc_{k}_fbrate" for k in E.ENCODE_KEYS]
    return (BASE_ANCHOR + BASE_TIME + ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt",
                                       "flt_prefix", "stand_prefix",
                                       "AIRCRAFT_TYPE_mvt"] + enc)


def tail_feature_list() -> list[str]:
    """D1. Null-flight tail head."""
    return ["sd", "sd_mod_86400", "hour", "ADEP_mvt", "stand_prefix", "flt_prefix"]


CAT_COLS = tuple(BASE_CAT)
