"""Planned-time features from the Network Manager flight-plan trajectory (M1).

ARVT_1_flt - EOBT_1_flt holds the planned taxi-out plus the planned flight time. The
residual against a route median isolates the planned taxi-out. All inputs are planned
values; none reads AOBT_3_flt, ARVT_3_flt or LOBT_flt (docs/MODEL_ANALYSIS.md F18).
"""
import pandas as pd

ROUTE_KEY = ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"]
PLAN_NUM_COLS = ["plan_block", "arvt1_mvt", "plan_taxi_res"]


def _route(dep: pd.DataFrame) -> pd.Series:
    return dep[ROUTE_KEY].astype(str).agg("|".join, axis=1)


def add_plan_times(dep: pd.DataFrame) -> pd.DataFrame:
    """Add plan_block and arvt1_mvt; dep needs mvt_ts and eobt1_ts."""
    arvt1 = pd.to_datetime(dep["ARVT_1_flt"], errors="coerce")
    dep["plan_block"] = (arvt1 - dep["eobt1_ts"]).dt.total_seconds()
    dep["arvt1_mvt"] = (arvt1 - dep["mvt_ts"]).dt.total_seconds()
    return dep


def fit_route_medians(dep: pd.DataFrame) -> pd.Series:
    """Median plan_block per route key on the fit rows."""
    return dep["plan_block"].groupby(_route(dep)).median()


def add_plan_residual(dep: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Add plan_taxi_res = plan_block minus the fitted route median."""
    dep["plan_taxi_res"] = dep["plan_block"] - _route(dep).map(medians).astype(float)
    return dep
