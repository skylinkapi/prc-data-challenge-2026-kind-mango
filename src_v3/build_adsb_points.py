"""WINNING_PLAN A2: low points near the 10 airports from one ADSB.lol globe_history day.

The day archive is a tar split into 2 GB parts. The builder streams the
parts as one tar and never unpacks it to disk. A worker pool decodes each
`trace_full` file and keeps the points inside the box around an airport
that are on the ground or below `ADSB_MAX_ALT_FT`. Data (c) ADSB.lol
contributors, ODbL 1.0. The output is a derived database: keep it out of git.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import tarfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import orjson
import pandas as pd

from src_v3 import config as C

AIRPORTS = C.ROOT / "external" / "airports" / "airports.csv"
COLUMNS = ["icao", "reg", "type", "apt", "ts", "lat", "lon", "alt", "gs", "flight"]
log = logging.getLogger(__name__)


class Chain:
    """Read-only file object over the parts of a split archive, in order."""

    def __init__(self, paths: list[Path]):
        self.files = [open(p, "rb") for p in paths]

    def read(self, n: int = -1) -> bytes:
        out = b""
        while self.files and (n < 0 or len(out) < n):
            chunk = self.files[0].read(-1 if n < 0 else n - len(out))
            if not chunk:
                self.files.pop(0).close()
            out += chunk
        return out


def load_boxes() -> list[tuple]:
    """One box per target airport: the OurAirports reference point plus and minus `ADSB_BOX_DEG`."""
    a = pd.read_csv(AIRPORTS, usecols=["ident", "latitude_deg", "longitude_deg"])
    a = a[a["ident"].isin(C.TARGETS)]
    dlat, dlon = C.ADSB_BOX_DEG
    return [(r.ident, r.latitude_deg - dlat, r.latitude_deg + dlat, r.longitude_deg - dlon,
             r.longitude_deg + dlon) for r in a.itertuples()]


def init_worker(boxes: list[tuple]) -> None:
    global BOXES
    BOXES = boxes


def parse_trace(raw: bytes) -> list[tuple] | None:
    """Kept points of one aircraft-day, one tuple per point, in `COLUMNS` order; None for a corrupt trace.

    The 2025-10-15 release holds a trace with a broken gzip stream; one bad aircraft-day must not drop the day.
    """
    try:
        d = orjson.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    except (zlib.error, EOFError, gzip.BadGzipFile, orjson.JSONDecodeError):
        return None
    base, icao, reg, typ = d["timestamp"], d.get("icao"), d.get("r"), d.get("t")
    out, flight = [], None
    for p in d["trace"]:
        if p[8] and p[8].get("flight"):
            flight = p[8]["flight"].strip()
        lat, lon, alt = p[1], p[2], p[3]
        if alt != "ground" and (alt is None or alt > C.ADSB_MAX_ALT_FT):
            continue
        for apt, la0, la1, lo0, lo1 in BOXES:
            if la0 <= lat <= la1 and lo0 <= lon <= lo1:
                out.append((icao, reg, typ, apt, base + p[0], lat, lon,
                            C.ADSB_GROUND if alt == "ground" else alt, p[4], flight))
                break
    return out


def members(parts: list[Path]):
    """Yield the bytes of each `trace_full` member of a day archive; the archive streams across its parts."""
    with tarfile.open(fileobj=Chain(parts), mode="r|") as tar:
        for m in tar:
            if m.isfile() and "trace_full" in m.name:
                yield tar.extractfile(m).read()


def extract_day(parts: list[Path], out: Path) -> None:
    """Write the kept points of one day to `out`."""
    rows, traces, bad = [], 0, 0
    with Pool(C.ADSB_WORKERS, initializer=init_worker, initargs=(load_boxes(),)) as pool:
        for pts in pool.imap_unordered(parse_trace, members(parts), chunksize=64):
            traces += 1
            if pts is None:
                bad += 1
            else:
                rows.extend(pts)
    if bad > C.ADSB_MAX_BAD_TRACE_SHARE * traces:
        raise RuntimeError(f"{out.stem}: {bad} of {traces} traces are corrupt, above the "
                           f"{C.ADSB_MAX_BAD_TRACE_SHARE:.0%} limit. Delete the raw parts, download the day again, then rerun.")
    if bad:
        log.warning("%s: skipped %d corrupt traces of %d", out.stem, bad, traces)
    f = pd.DataFrame(rows, columns=COLUMNS)
    f["ts"] = pd.to_datetime(f["ts"], unit="s", utc=True).astype("datetime64[us, UTC]")
    tmp = out.with_suffix(".tmp")
    f.to_parquet(tmp)
    tmp.replace(out)  # readers of the points folder never see a half-written day
    log.info("%s: %d points, %d aircraft", out.name, len(f), f["icao"].nunique())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("day", help="Day as YYYY.MM.DD, as in the ADSB.lol release tag.")
    args = ap.parse_args()
    parts = sorted((C.ADSB_DIR / "raw").glob(f"v{args.day}.tar.*"))
    if not parts:
        raise FileNotFoundError(f"No archive parts for {args.day} in {C.ADSB_DIR / 'raw'}. "
                                "Download the day first, then rerun.")
    (C.ADSB_DIR / "points").mkdir(exist_ok=True)
    extract_day(parts, C.ADSB_DIR / "points" / f"{args.day}.parquet")


if __name__ == "__main__":
    main()
