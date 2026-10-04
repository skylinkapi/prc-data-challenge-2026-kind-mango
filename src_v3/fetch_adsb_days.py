"""WINNING_PLAN A2: download ADSB.lol days, keep the airport points, delete the raw archive.

`--feed` picks one of `C.ADSB_FEEDS`; points of a feed other than `prod-0` go to `points_<feed>/`
(third-edition lever K1).

The next day downloads while the current day extracts. A day with a points
file already on disk is skipped, so a stopped run resumes where it ended.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import urllib3

from src_v3 import config as C
from src_v3.build_adsb_points import extract_day

RAW = C.ADSB_DIR / "raw"
POINTS = C.ADSB_DIR / "points"
HTTP = urllib3.PoolManager(retries=urllib3.Retry(total=C.ADSB_RETRIES, backoff_factor=C.ADSB_BACKOFF_S,
                                                 status_forcelist=(429, 500, 502, 503, 504)),
                           timeout=urllib3.Timeout(connect=C.ADSB_CONNECT_TIMEOUT_S, read=C.ADSB_READ_TIMEOUT_S))
log = logging.getLogger(__name__)


def download(day: str, feed: str) -> list[Path]:
    """`download_once` with whole-day retries: a stalled stream (2025-04-10 hung 12 h) raises a read timeout."""
    for attempt in range(1, C.ADSB_RETRIES + 1):
        try:
            return download_once(day, feed)
        except (urllib3.exceptions.HTTPError, OSError) as e:
            log.warning("%s: attempt %d failed: %s", day, attempt, e)
            time.sleep(C.ADSB_BACKOFF_S * attempt)
    raise RuntimeError(f"The download of {day} failed {C.ADSB_RETRIES} times. Check the network, then rerun.")


def download_once(day: str, feed: str) -> list[Path]:
    """Download the parts of one day; return their paths. Days below 2 GB ship as one unsplit `.tar`."""
    parts = []
    for part in C.ADSB_PARTS:
        url = C.ADSB_URL.format(year=day[:4], day=day, part=part, feed=feed)
        dest = RAW / f"v{day}-{feed}.tar.{part}"
        r = HTTP.request("GET", url, preload_content=False)
        if r.status == 404 and not parts:
            r.release_conn()
            url, dest = url.removesuffix(f".{part}"), RAW / f"v{day}-{feed}.tar"
            r = HTTP.request("GET", url, preload_content=False)
        elif r.status == 404:
            r.release_conn()
            break
        if r.status != 200:
            r.release_conn()
            raise RuntimeError(f"The download of {url} failed with HTTP {r.status}. "
                               "Check the release tag of the day, then rerun.")
        with open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
        r.release_conn()
        parts.append(dest)
    log.info("%s: downloaded %d parts, %.2f GB", day, len(parts), sum(p.stat().st_size for p in parts) / 1e9)
    return parts


def released(days: list[str], feed: str) -> list[str]:
    """The days that have a release of `feed`; the tag list comes from the repository's git ref advertisement."""
    tags = set()
    for year in sorted({d[:4] for d in days}):
        body = HTTP.request("GET", C.ADSB_REFS_URL.format(year=year)).data.decode("latin-1")
        tags |= set(re.findall(rf"refs/tags/v(\d{{4}}\.\d{{2}}\.\d{{2}})-planes-readsb-{feed}\b(?!\^)", body))
    missing = [d for d in days if d not in tags]
    if missing:
        log.warning("%d days have no %s release and are skipped: %s", len(missing), feed, ", ".join(missing))
    return [d for d in days if d in tags]


def points_dir(feed: str) -> Path:
    return POINTS if feed == C.ADSB_FEEDS[0] else C.ADSB_DIR / f"points_{feed}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="First day, YYYY-MM-DD.")
    ap.add_argument("--end", required=True, help="Last day, YYYY-MM-DD, inclusive.")
    ap.add_argument("--feed", default=C.ADSB_FEEDS[0], choices=C.ADSB_FEEDS)
    args = ap.parse_args()
    points = points_dir(args.feed)
    RAW.mkdir(parents=True, exist_ok=True)
    points.mkdir(exist_ok=True)
    days = [d.strftime("%Y.%m.%d") for d in pd.date_range(args.start, args.end)]
    days = released([d for d in days if not (points / f"{d}.parquet").exists()], args.feed)
    with ThreadPoolExecutor(1) as ex:
        nxt = ex.submit(download, days[0], args.feed) if days else None
        for i, day in enumerate(days):
            parts = nxt.result()
            nxt = ex.submit(download, days[i + 1], args.feed) if i + 1 < len(days) else None
            extract_day(parts, points / f"{day}.parquet")
            for p in parts:
                p.unlink()


if __name__ == "__main__":
    main()
