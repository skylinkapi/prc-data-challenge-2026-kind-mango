"""Pull aeroway=parking_position (stands) from OSM Overpass API for the
10 target airports. Save as external/osm/stands.parquet with columns
ADEP_mvt, stand_ref, lat, lon.

Uses a ~5 km bbox around each airport's OurAirports reference point.
"""
import json
import os
import time
import urllib.request
import urllib.parse
import pandas as pd

from features_weather import TARGET_ICAOS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM_DIR = os.path.join(ROOT, "external", "osm")
os.makedirs(OSM_DIR, exist_ok=True)

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]


def airport_coords() -> pd.DataFrame:
    ap = pd.read_csv(os.path.join(ROOT, "external", "airports", "airports.csv"),
                     low_memory=False)
    ap = ap[ap["ident"].isin(TARGET_ICAOS)][["ident", "latitude_deg", "longitude_deg"]]
    return ap.rename(columns={"ident": "ADEP_mvt"})


def bbox(lat: float, lon: float, half_km: float = 3.5) -> tuple:
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * abs(pd.np.cos(pd.np.deg2rad(lat))) if False
                      else 111.0 * 0.7)  # rough at 45N
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def fetch_stands(icao: str, lat: float, lon: float) -> list:
    import math
    dlat = 3.5 / 111.0
    dlon = 3.5 / (111.0 * max(math.cos(math.radians(lat)), 0.3))
    s, w, n, e = lat - dlat, lon - dlon, lat + dlat, lon + dlon
    query = f"""
[out:json][timeout:120];
(
  node["aeroway"="parking_position"]({s},{w},{n},{e});
  way["aeroway"="parking_position"]({s},{w},{n},{e});
);
out center tags;
"""
    data = urllib.parse.urlencode({"data": query}).encode()
    last_err = None
    for mirror in OVERPASS_MIRRORS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(mirror, data=data,
                                             headers={"User-Agent": "prc-dc-2026/1.0"})
                with urllib.request.urlopen(req, timeout=240) as r:
                    j = json.load(r)
                break
            except Exception as e:
                last_err = e
                time.sleep(10 * (attempt + 1))
        else:
            continue
        break
    else:
        raise last_err
    rows = []
    for el in j.get("elements", []):
        ref = el.get("tags", {}).get("ref") or el.get("tags", {}).get("name")
        if not ref:
            continue
        if el["type"] == "node":
            plat, plon = el["lat"], el["lon"]
        else:
            c = el.get("center", {})
            plat, plon = c.get("lat"), c.get("lon")
        if plat is None:
            continue
        rows.append({"ADEP_mvt": icao, "stand_ref": str(ref).strip().upper(),
                     "lat": plat, "lon": plon})
    return rows


def main():
    coords = airport_coords()
    all_rows = []
    for _, r in coords.iterrows():
        icao = r["ADEP_mvt"]
        cache = os.path.join(OSM_DIR, f"stands_{icao}.json")
        if os.path.exists(cache):
            rows = json.load(open(cache))
            print(f"[cache] {icao}: {len(rows)} stands")
        else:
            print(f"[fetch] {icao} at ({r['latitude_deg']:.3f}, {r['longitude_deg']:.3f}) ...")
            try:
                rows = fetch_stands(icao, r["latitude_deg"], r["longitude_deg"])
                json.dump(rows, open(cache, "w"))
                print(f"  -> {len(rows)} stands")
            except Exception as e:
                print(f"  FAIL: {e}")
                rows = []
            time.sleep(3)   # polite delay for Overpass
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = os.path.join(OSM_DIR, "stands.parquet")
    df.to_parquet(out)
    print(f"\nSaved {len(df)} stands across {df['ADEP_mvt'].nunique()} airports -> {out}")
    print("\nStands per airport:")
    print(df.groupby("ADEP_mvt").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
