"""Download every open dataset of the pipeline into `external/` with one command.

`python -m src_v3.fetch_external` gets each source of the README data-source table except the ADS-B days.
`python -m src_v3.fetch_external adsb` gets the ADSB.lol days of the final recipe (`ADSB_FINAL_RANGES`): each
released day is a download of about 4 GB, of which about 15 MB stay on disk. `--list` shows the steps of each
source and which outputs are on disk.

A step runs only if one of its outputs is missing, so a stopped run resumes and a file on disk is never replaced.
Delete an output to fetch it again. OSM and X-Plane serve their current state, not the state of the first download.
The organiser files are not open data: `REPRODUCE.md` section 2 gets them with an OSN account. `xplane` stores the
`apt.dat` files only; `python -m src_v3.build_xplane_stands build` needs the organiser files too.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import urllib.request
from functools import partial
from pathlib import Path

import pandas as pd

from src_v3 import config as C

EC = "external/eurocontrol"
OPDI = C.ROOT / "external" / "opdi"
EVENTS = "external/opdi/events_all.parquet"
LONG = ("adsb",)
log = logging.getLogger(__name__)


def get(url: str, path: Path) -> None:
    """Download `url` to `path`; a stopped download leaves no file under the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(url, part)
    part.replace(path)


def download(url: str, out: str) -> tuple:
    return partial(get, url, C.ROOT / out), [out]


def merge_opdi() -> None:
    """Join the filtered OPDI chunks from `OPDI_FIRST_CHUNK` on into the event table that the builders read."""
    files = [f for f in sorted((OPDI / "filtered").glob("*.parquet")) if f.stem >= C.OPDI_FIRST_CHUNK]
    ev = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True, errors="coerce").astype("datetime64[us, UTC]")
    for col in ("osn_flight_id", "osm_airport"):
        ev[col] = ev[col].astype(str).str.strip().str.upper()
    ev = ev.dropna(subset=["event_time"]).sort_values(["osm_airport", "event_time"]).reset_index(drop=True)
    ev.to_parquet(C.ROOT / EVENTS)


# source -> steps; a step is a command of this Python or a function, with the outputs that make it skip
SOURCES = {
    "airports": [download(C.OURAIRPORTS_URL.format(n), f"external/airports/{n}.csv") for n in C.OURAIRPORTS_FILES],
    "metar": [(["-m", "src_v3.fetch_metar"], [f"external/metar/{s}.csv" for s in C.METAR_STATIONS])],
    "eurocontrol": [
        *[download(C.EC_XLS_URL.format(n), f"{EC}/{n}.xlsx") for n in C.EC_XLS_FILES],
        *[download(C.EC_CSV_URL.format(n, y), f"{EC}/{n}_{y}.csv") for n in C.EC_CSV_FILES for y in C.ATFM_SLOT_YEARS],
        (["src/build_eurocontrol_daily.py"], [f"{EC}/daily_features.parquet"])],
    "osm": [(["src/build_osm_stands.py"], ["external/osm/stands.parquet"]),
            (["src/build_osm_taxi_paths.py"], ["external/osm/taxi_paths.parquet"])],
    "vrs": [(["src/build_vrs_aircraft.py"], ["external/vrs/aircraft_types.parquet"])],
    "xplane": [(["-m", "src_v3.build_xplane_stands", "fetch"], [f"external/xplane/{a}.dat" for a in C.XPLANE_SCENERY])],
    "opdi": [(["src/fetch_opdi_events.py"], [EVENTS]), (merge_opdi, [EVENTS]),
             (["src/build_opdi_taxi_v2.py"], ["external/opdi/taxi_out_v2.parquet"])],
    # no output list: `fetch_adsb_days` skips the days that are on disk
    "adsb": [(["-m", "src_v3.fetch_adsb_days", "--start", a, "--end", b], []) for a, b in C.ADSB_FINAL_RANGES],
}


def label(step) -> str:
    return " ".join(["python", *step]) if isinstance(step, list) else getattr(step, "__name__", "download")


def is_done(outs: list[str]) -> bool:
    return bool(outs) and all((C.ROOT / o).exists() for o in outs)


def fetch(source: str) -> None:
    """Run the steps of `source` that have a missing output."""
    for step, outs in SOURCES[source]:
        if is_done(outs):
            log.info("%s: on disk, skipped: %s", source, ", ".join(outs))
            continue
        log.info("%s: %s -> %s", source, label(step), ", ".join(outs) or "external/adsblol/points/")
        subprocess.run([sys.executable, *step], check=True, cwd=C.ROOT) if isinstance(step, list) else step()


def show() -> None:
    for source, steps in SOURCES.items():
        for step, outs in steps:
            state = "on disk" if is_done(outs) else "to do"
            log.info("%-11s %-7s %s -> %s", source, state, label(step), ", ".join(outs) or "external/adsblol/points/")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", choices=list(SOURCES), metavar="SOURCE",
                    help=f"One or more of: {', '.join(SOURCES)}. Default: every source except {', '.join(LONG)}.")
    ap.add_argument("--list", action="store_true", help="Show the steps and their state; download nothing.")
    args = ap.parse_args()
    if args.list:
        return show()
    for source in args.sources or [s for s in SOURCES if s not in LONG]:
        fetch(source)
    if not args.sources:
        log.info("The ADS-B days did not run. `python -m src_v3.fetch_external adsb` downloads them (about 4 GB a day).")


if __name__ == "__main__":
    main()
