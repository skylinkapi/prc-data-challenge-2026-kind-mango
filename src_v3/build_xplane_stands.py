"""Stand positions from the X-Plane Scenery Gateway where OSM has no position for the assigned stand (EDDM, EHAM).

OSM has no node for the assigned stand on 42 % of EDDM and 32 % of EHAM departures in 2026, so no ADS-B
clock at the stand can exist there. The Gateway airport packs (GPL v2 or later) list every stand as an
`apt.dat` row 1300: latitude, longitude, heading, type, name. `fetch` stores each `apt.dat` unchanged,
copyright header included. `build` keeps the OSM stands and adds one X-Plane position for each stand value
of the movement files that OSM lacks. A name matches on letters and digits with leading zeros dropped
(EHAM `C06` is X-Plane `C6`); a base stand without its own row takes the mean of its lettered sub-positions
(EDDM `215` is the mean of `215A` and `215B`).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import urllib.request
import zipfile

import pandas as pd

from src_v3 import config as C
from src_v3.build_adsb_clock import STANDS
from src_v3.build_nm_clocks_v73 import load_dep

log = logging.getLogger(__name__)


def key(s: str) -> str:
    """The key `build_adsb_clock.own_stand` matches on: upper-case letters and digits."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def canon(s: str) -> str:
    """`key` with the leading zeros of each digit group dropped."""
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", key(s))


def fetch() -> None:
    C.XPLANE_DIR.mkdir(parents=True, exist_ok=True)
    for apt, scenery in C.XPLANE_SCENERY.items():
        pack = json.load(urllib.request.urlopen(C.XPLANE_API.format(scenery), timeout=120))["scenery"]
        z = zipfile.ZipFile(io.BytesIO(base64.b64decode(pack["masterZipBlob"])))
        (C.XPLANE_DIR / f"{apt}.dat").write_bytes(z.read(f"{apt}.dat"))
        log.info("%s: scenery %d, accepted %s", apt, scenery, pack.get("dateAccepted"))


def xplane_positions(apt: str) -> pd.DataFrame:
    """Mean position per canonical stand name of the `apt.dat` rows 1300."""
    rows = [line.split(None, 6) for line in (C.XPLANE_DIR / f"{apt}.dat").read_text(encoding="utf8", errors="ignore").splitlines()
            if line.startswith("1300 ")]
    x = pd.DataFrame([(canon(r[6]), float(r[1]), float(r[2])) for r in rows if len(r) == 7], columns=["c", "lat", "lon"])
    return x.groupby("c")[["lat", "lon"]].mean()


def fill(apt: str, values: pd.Series, osm_keys: set) -> pd.DataFrame:
    """One X-Plane position per stand value that OSM lacks; exact canonical name first, else its lettered sub-positions."""
    xp = xplane_positions(apt)
    out = []
    for v in values[~values.map(key).isin(osm_keys)].unique():
        c = canon(v)
        sub = xp.loc[[c]] if c in xp.index else xp[xp.index.str.fullmatch(re.escape(c) + "[A-Z]")]
        if len(sub):
            out.append((apt, v, sub["lat"].mean(), sub["lon"].mean()))
    return pd.DataFrame(out, columns=["ADEP_mvt", "stand_ref", "lat", "lon"])


def build() -> None:
    osm = pd.read_parquet(STANDS)
    dep = load_dep([*sorted(C.TRAIN_DIR.glob("training_*.parquet")), C.RANK], ["STAND_mvt"])
    parts = [osm]
    for apt in C.XPLANE_SCENERY:
        values = dep.loc[dep["ADEP_mvt"] == apt, "STAND_mvt"].dropna().astype(str)
        f = fill(apt, values, set(osm.loc[osm["ADEP_mvt"] == apt, "stand_ref"].map(key)))
        covered = values.map(key).isin(set(f["stand_ref"].map(key)))
        log.info("%s: %d stand values filled; %.3f of departures gain a position", apt, len(f), covered.mean())
        parts.append(f)
    pd.concat(parts, ignore_index=True).to_parquet(C.STANDS_OSM_XPLANE)
    log.info("wrote %s", C.STANDS_OSM_XPLANE)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("fetch", "build"))
    fetch() if ap.parse_args().mode == "fetch" else build()


if __name__ == "__main__":
    main()
