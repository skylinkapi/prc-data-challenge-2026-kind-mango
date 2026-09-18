"""MB4: cyclic day-of-year, local hour with DST, and public-holiday flag.

Reads a frame with `MVT_ID_mvt`, `mvt_ts` and `ADEP_mvt`, returns the four
new columns: `doy_sin`, `doy_cos`, `hour_local`, `is_public_hol`.

Uses the standard IANA timezone per airport (via zoneinfo, stdlib) so DST
is handled at the row's mvt_ts date. Public holidays come from the
`holidays` package used already in `build_calendar.py`, keyed on the
ADEP country.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

import holidays as _holidays

from src_v3 import config as C

# Airport -> IANA timezone. Same country map as build_calendar.py; add
# the tz next to it for DST-aware local hour.
ICAO_TZ = {
    "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",  "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}


def _cyclic_doy(ts: pd.Series) -> tuple[pd.Series, pd.Series]:
    doy = ts.dt.dayofyear.astype("float64")
    ang = 2 * np.pi * doy / 365.25
    return np.sin(ang), np.cos(ang)


def _local_hour_by_airport(ts: pd.Series, adep: pd.Series) -> pd.Series:
    """DST-aware local hour of the row's mvt_ts at the ADEP airport."""
    out = np.full(len(ts), np.nan, dtype="float64")
    for icao, tzname in ICAO_TZ.items():
        m = (adep.astype(str) == icao).values
        if not m.any():
            continue
        local_ts = ts[m].dt.tz_convert(ZoneInfo(tzname))
        out[m] = local_ts.dt.hour.astype("float64").values
    return pd.Series(out, index=ts.index)


def _holiday_flag(ts: pd.Series, adep: pd.Series) -> pd.Series:
    """Public-holiday flag at the ADEP country for the local date."""
    years = sorted({int(y) for y in ts.dt.year.dropna().astype(int).unique()})
    country_of = C.ICAO_COUNTRY
    hmap = {c: set(_holidays.country_holidays(c, years=years).keys())
            for c in set(country_of.values())}
    apt = adep.astype(str)
    country = apt.map(country_of)
    date = ts.dt.date
    flag = pd.Series([d in hmap.get(c, set()) if pd.notna(d) and c else False
                      for d, c in zip(date, country)], index=ts.index)
    return flag.astype("Int8")


def add_calendar_extra(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds doy_sin, doy_cos, hour_local, is_public_hol to `frame`. Does
    not remove the numeric `month` column; the caller decides whether to
    drop it from the feature list.
    """
    ts = pd.to_datetime(frame["mvt_ts"], utc=True, errors="coerce")
    out = frame.copy()
    s, c = _cyclic_doy(ts)
    out["doy_sin"] = s
    out["doy_cos"] = c
    out["hour_local"] = _local_hour_by_airport(ts, frame["ADEP_mvt"])
    out["is_public_hol"] = _holiday_flag(ts, frame["ADEP_mvt"])
    return out
