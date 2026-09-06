"""Download OPDI flight-event archive for 2025-01 through 2026-08 and
stream-filter each 10-day chunk to keep only:
  - events at our 10 target airports (osm_airport in TARGET_ICAOS)
  - taxi-relevant event types (parking, apron, taxiway, runway, threshold)

Raw chunks (~245 MB each) are downloaded then filtered then discarded.
Filtered per-chunk parquets accumulate under external/opdi/filtered/.

Handles resume: skips any chunk whose filtered output already exists.
"""
import json
import os
import time
import urllib.request
import urllib.error
from datetime import date, timedelta

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "external", "opdi", "filtered")
RAW_TMP = os.path.join(ROOT, "external", "opdi", "_raw_tmp")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RAW_TMP, exist_ok=True)

BASE = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events"

TARGET_ICAOS = {"EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
                "LFPG", "LIRF", "LTFM", "LSZH"}

# Ground-taxi event types worth keeping. Airborne events discarded.
KEEP_TYPES = {
    "entry-parking_position", "exit-parking_position",
    "entry-apron",            "exit-apron",
    "entry-taxiway",          "exit-taxiway",
    "entry-runway",           "exit-runway",
    "entry-threshold",        "exit-threshold",
}


def chunks_covering(start: date, end: date) -> list[tuple[date, date]]:
    """OPDI 10-day chunks continuing from 2022-01-01. Return chunks that
    overlap [start, end]."""
    origin = date(2022, 1, 1)
    delta = timedelta(days=10)
    # find first chunk covering >= start
    first_idx = max(0, (start - origin).days // 10)
    result = []
    d = origin + first_idx * delta
    while d < end:
        result.append((d, d + delta))
        d += delta
    return result


def chunk_filename(a: date, b: date) -> str:
    return f"flight_events_{a:%Y%m%d}_{b:%Y%m%d}.parquet"


def parse_info_batch(info_series: pd.Series) -> pd.DataFrame:
    """Extract osm_airport, osn_flight_id, osm_ref from info JSON."""
    def _p(s):
        if not isinstance(s, str) or not s or s == "":
            return (None, None, None)
        try:
            d = json.loads(s)
            return (d.get("osm_airport"), d.get("osn_flight_id"),
                    d.get("osm_ref"))
        except Exception:
            return (None, None, None)
    parsed = info_series.map(_p)
    return pd.DataFrame(list(parsed.values), index=info_series.index,
                        columns=["osm_airport", "osn_flight_id", "osm_ref"])


def fetch_and_filter(a: date, b: date, retries: int = 3) -> None:
    fname = chunk_filename(a, b)
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
        print(f"[skip] {fname}: filtered exists ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)
        return

    url = f"{BASE}/{fname}"
    raw_path = os.path.join(RAW_TMP, fname)

    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            print(f"[dl]   {fname}: attempt {attempt} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "prc-dc-2026/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(raw_path, "wb") as f:
                # stream to avoid holding full body in RAM
                while True:
                    buf = r.read(1 << 20)
                    if not buf: break
                    f.write(buf)
            dl_mb = os.path.getsize(raw_path) / 1e6
            print(f"[dl]   {fname}: {dl_mb:.1f} MB in {time.time()-t0:.1f}s", flush=True)
            break
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[warn] {fname} attempt {attempt}: {e}", flush=True)
            time.sleep(10 * attempt)
    else:
        print(f"[FAIL] {fname}: gave up", flush=True)
        return

    t0 = time.time()
    df = pd.read_parquet(raw_path)
    n0 = len(df)
    # Filter first by type (cheap), then parse info only for surviving rows
    df = df[df["type"].isin(KEEP_TYPES)]
    n1 = len(df)
    info_cols = parse_info_batch(df["info"])
    df = pd.concat([df.drop(columns=["info", "source", "version"]).reset_index(drop=True),
                    info_cols.reset_index(drop=True)], axis=1)
    df = df[df["osm_airport"].isin(TARGET_ICAOS)]
    df["osn_flight_id"] = df["osn_flight_id"].astype(str).str.strip()
    df.to_parquet(out_path)
    n2 = len(df)
    print(f"[fil]  {fname}: {n0:,} -> {n1:,} (type) -> {n2:,} (airport)   "
          f"{os.path.getsize(out_path)/1e6:.1f} MB  ({time.time()-t0:.1f}s)", flush=True)

    try:
        os.remove(raw_path)
    except OSError:
        pass


def main():
    start = date(2025, 1, 1)
    end = date(2026, 8, 1)
    chunks = chunks_covering(start, end)
    print(f"Fetching {len(chunks)} OPDI event chunks from {start} to {end}", flush=True)
    for a, b in chunks:
        fetch_and_filter(a, b)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
