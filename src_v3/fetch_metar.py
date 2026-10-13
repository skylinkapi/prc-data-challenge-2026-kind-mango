"""Download the METAR CSVs that `src/features_weather.py` reads (MODEL_ANALYSIS MC6).

Source: Iowa State University ASOS network archive (IEM), routine and
special reports, one CSV per ICAO in `external/metar/`. The column set
and the date range are the ones of the files that trained every served
model.
"""
from __future__ import annotations

import argparse
import logging
import time
import urllib.parse
import urllib.request

import pandas as pd

from src_v3 import config as C

log = logging.getLogger(__name__)


def url(icao: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Request URL of the IEM archive for one station and the days `[start, end)`."""
    query = [("station", icao), *[("data", c) for c in C.METAR_COLUMNS],
             ("year1", start.year), ("month1", start.month), ("day1", start.day),
             ("year2", end.year), ("month2", end.month), ("day2", end.day),
             ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
             ("missing", "M"), ("trace", "T"), ("direct", "no"),
             *[("report_type", r) for r in C.METAR_REPORT_TYPES]]
    return C.METAR_URL + "?" + urllib.parse.urlencode(query)


def fetch(icao: str, start: pd.Timestamp, end: pd.Timestamp) -> bytes:
    """CSV bytes for `icao` in `[start, end)`; retries because the archive rate-limits bursts."""
    for attempt in range(1, C.METAR_RETRIES + 1):
        try:
            with urllib.request.urlopen(url(icao, start, end), timeout=C.METAR_TIMEOUT_S) as r:
                return r.read()
        except OSError as e:
            log.warning("METAR %s attempt %d failed: %s", icao, attempt, e)
            time.sleep(C.METAR_BACKOFF_S * attempt)
    raise RuntimeError(f"The METAR download for {icao} failed {C.METAR_RETRIES} times. "
                       "Check the network and the IEM service status, then rerun.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=C.METAR_START)
    ap.add_argument("--end", default=C.METAR_END, help="First day that is not downloaded.")
    ap.add_argument("--out", default=str(C.ROOT / "external" / "metar"))
    args = ap.parse_args()
    out = C.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for icao in C.METAR_STATIONS:
        body = fetch(icao, pd.Timestamp(args.start), pd.Timestamp(args.end))
        (out / f"{icao}.csv").write_bytes(body)
        log.info("%s: %d lines", icao, body.count(b"\n"))


if __name__ == "__main__":
    main()
