"""MF2: runway-queue structure before each take-off, backward windows only.

For each departure, from the other movements at the same airport with a
take-off or landing time in the window before its own take-off:

- `rwy_dep_{5,10}`: departures on the same runway in the last 5 and 10 min;
- `rwy_heavy_{5,10}`, `rwy_heavy_share_10`: of those, wake category H or J;
- `dir_dep_10`: departures in the last 10 min, any runway, to a destination
  within 30 degrees of the row's initial bearing;
- `rwy_set_age_min`: minutes since the set of runways used by departures and
  arrivals in the last 30 min last changed.

The pool is every movement of the file, with or without a label (MD8). The
2025 training files and the 2026 ranking file go through the same code.
"""
from __future__ import annotations

import glob
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C

AIRPORTS = C.ROOT / "external" / "airports" / "airports.csv"
OUT_TRAIN = C.ROOT / "models" / "runway_queue_train_v71.parquet"
OUT_RANK = C.ROOT / "models" / "runway_queue_rank_v71.parquet"
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
        "RUNWAY_mvt", "WK_TBL_CAT_flt"]
HEAVY = ("H", "J")
WINDOWS_S = (300, 600)
DIR_WINDOW_S, DIR_TOL_DEG = 600, 30.0
SET_WINDOW_S = 1800
FEATURES = ["rwy_dep_5", "rwy_dep_10", "rwy_heavy_5", "rwy_heavy_10",
            "rwy_heavy_share_10", "dir_dep_10", "rwy_set_age_min"]
log = logging.getLogger(__name__)


def load_movements(paths: list) -> pd.DataFrame:
    """Movements at the ten airports with airport, time in seconds, runway and wake class."""
    m = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    m["apt"] = np.where(m["PHASE_mvt"] == "DEP", m["ADEP_mvt"], m["ADES_mvt"])
    m = m[m["apt"].isin(C.TARGETS)].copy()
    ts = pd.to_datetime(m["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    m["t"] = (ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)  # same unit for us and ns files
    m["heavy"] = m["WK_TBL_CAT_flt"].isin(HEAVY).astype(np.int32)
    return m.dropna(subset=["RUNWAY_mvt"]).sort_values(["apt", "t"], kind="stable")


def bearings(dep: pd.DataFrame) -> np.ndarray:
    """Initial great-circle bearing in degrees from the airport to the destination."""
    a = pd.read_csv(AIRPORTS, usecols=["ident", "latitude_deg", "longitude_deg"]).set_index("ident")
    lat1, lon1 = (np.radians(dep["apt"].map(a[c]).values) for c in ("latitude_deg", "longitude_deg"))
    lat2, lon2 = (np.radians(dep["ADES_mvt"].map(a[c]).values) for c in ("latitude_deg", "longitude_deg"))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(x, y)) % 360


def runway_counts(dep: pd.DataFrame) -> pd.DataFrame:
    """Same-runway departure and heavy counts in `[t - W, t)`."""
    out = pd.DataFrame(index=dep.index)
    for _, g in dep.groupby(["apt", "RUNWAY_mvt"], sort=False):
        t, heavy = g["t"].values, np.concatenate([[0], np.cumsum(g["heavy"].values)])
        hi = np.searchsorted(t, t, side="left")
        for w in WINDOWS_S:
            lo = np.searchsorted(t, t - w, side="left")
            out.loc[g.index, f"rwy_dep_{w // 60}"] = hi - lo
            out.loc[g.index, f"rwy_heavy_{w // 60}"] = heavy[hi] - heavy[lo]
    out["rwy_heavy_share_10"] = out["rwy_heavy_10"] / out["rwy_dep_10"].where(out["rwy_dep_10"] > 0)
    return out


def direction_counts(dep: pd.DataFrame) -> pd.Series:
    """Departures in `[t - 10 min, t)` at the airport with a bearing within 30 degrees."""
    out = pd.Series(0, index=dep.index, dtype=np.int32)
    for _, g in dep.groupby("apt", sort=False):
        t, b = g["t"].values, g["bearing"].values
        n = np.zeros(len(g), dtype=np.int32)
        k = 1
        while k < len(g) and (t[k:] - t[:-k] < DIR_WINDOW_S).any():
            dt = t[k:] - t[:-k]
            diff = np.abs((b[k:] - b[:-k] + 180) % 360 - 180)
            n[k:] += ((dt > 0) & (dt < DIR_WINDOW_S) & (diff <= DIR_TOL_DEG)).astype(np.int32)
            k += 1
        out.loc[g.index] = n
    return out.where(dep["bearing"].notna())


def runway_set_age(moves: pd.DataFrame) -> pd.Series:
    """Minutes since the set of runways used in `[t - 30 min, t)` last changed; NaN in the first 30 min of the file."""
    age = pd.Series(np.nan, index=moves.index)
    for _, g in moves.groupby("apt", sort=False):
        t, rwy = g["t"].values, g["RUNWAY_mvt"].astype(str).values
        counts: dict[str, int] = {}
        lo, changed_at, dirty, ages = 0, t[0], False, np.empty(len(g))
        for i in range(len(g)):
            while t[lo] < t[i] - SET_WINDOW_S:
                counts[rwy[lo]] -= 1
                if counts[rwy[lo]] == 0:
                    del counts[rwy[lo]]
                    dirty = True
                lo += 1
            if dirty:
                changed_at, dirty = t[i], False
            ages[i] = (t[i] - changed_at) / 60 if t[i] - t[0] >= SET_WINDOW_S else np.nan
            if counts.get(rwy[i], 0) == 0:
                dirty = True
            counts[rwy[i]] = counts.get(rwy[i], 0) + 1
        age.loc[g.index] = ages
    return age


def build(paths: list) -> pd.DataFrame:
    """The seven MF2 columns keyed on the departure movement id."""
    moves = load_movements(paths)
    dep = moves[moves["PHASE_mvt"] == "DEP"].copy()
    dep["bearing"] = bearings(dep)
    out = runway_counts(dep)
    out["dir_dep_10"] = direction_counts(dep)
    out["rwy_set_age_min"] = runway_set_age(moves).loc[dep.index]
    out.insert(0, "MVT_ID_mvt", dep["MVT_ID_mvt"].values)
    log.info("%d departures; non-null share %s", len(out), out[FEATURES].notna().mean().round(3).to_dict())
    return out.dropna(subset=["MVT_ID_mvt"]).drop_duplicates("MVT_ID_mvt")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build(sorted(glob.glob(str(C.TRAIN_DIR / "*.parquet")))).to_parquet(OUT_TRAIN)
    build([C.RANK]).to_parquet(OUT_RANK)


if __name__ == "__main__":
    main()
