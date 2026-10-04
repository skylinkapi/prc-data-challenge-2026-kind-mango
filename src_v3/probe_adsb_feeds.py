"""Third-edition K1 probe: does a second ADSB.lol feed add ground coverage on one ranking day?

The clock builder runs twice on the departures of `--day`: on the `prod-0`
points alone, and on the union of `prod-0` and the other feeds that have a
points file for the day (deduplicated on `icao` and `ts`). The evening
before comes from `prod-0` in both runs. The report gives, per airport,
the share of departures with a ground segment and with a clock at the
assigned stand in each run.
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd

from src_v3 import build_adsb_clock as bc
from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep
from src_v3.fetch_adsb_days import POINTS, points_dir

REPORT = C.MODELS / "adsb_feeds_probe.json"
log = logging.getLogger(__name__)


def shares(clocks: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    """Per airport: share of departures with a ground segment and with a stand clock."""
    f = dep[["MVT_ID_mvt", "ADEP_mvt"]].merge(clocks, on="MVT_ID_mvt", how="left", validate="1:1")
    f = f.assign(segment=f["adsb_seg"].notna(), own=f[["adsb_own_move", "adsb_own_last"]].notna().any(axis=1))
    return f.groupby("ADEP_mvt")[["segment", "own"]].mean()


def build_on(points: Path, dep: pd.DataFrame) -> pd.DataFrame:
    """Clock columns of `dep` with the builder reading its points from `points`."""
    saved, bc.POINTS = bc.POINTS, points
    try:
        return bc.build(dep)
    finally:
        bc.POINTS = saved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2026-07-15")
    day = pd.Timestamp(ap.parse_args().day)
    name, before = f"{day:%Y.%m.%d}.parquet", f"{day - pd.Timedelta(days=1):%Y.%m.%d}.parquet"
    feeds = [f for f in C.ADSB_FEEDS[1:] if (points_dir(f) / name).exists()]
    union = pd.concat([pd.read_parquet(POINTS / name), *[pd.read_parquet(points_dir(f) / name) for f in feeds]])
    union = union.drop_duplicates(["icao", "ts"])
    dep = load_dep([C.RANK], ["CALLSIGN_flt", "STAND_mvt"])
    dep = dep[dep["MVT_TIME_UTC_mvt"].dt.date == day.date()].reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        union.to_parquet(tmp / name)
        if (POINTS / before).exists():
            pd.read_parquet(POINTS / before).to_parquet(tmp / before)
        runs = {"prod": shares(build_on(POINTS, dep), dep), "union": shares(build_on(tmp, dep), dep)}
    t = pd.concat(runs, axis=1).round(3)
    log.info("feeds %s, points prod %d, union %d\n%s", feeds, len(pd.read_parquet(POINTS / name)), len(union), t)
    REPORT.write_text(json.dumps({"day": f"{day:%Y-%m-%d}", "feeds": feeds, "points_union": len(union),
                                  "by_airport": {a: {f"{r}_{k}": float(t.loc[a, (r, k)]) for r in runs for k in ("segment", "own")}
                                                 for a in t.index}}, indent=1))


if __name__ == "__main__":
    main()
