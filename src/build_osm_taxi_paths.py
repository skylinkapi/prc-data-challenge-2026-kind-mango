"""Fetch full OSM taxi network per airport, build a graph, compute shortest
paths from every stand to every runway threshold end. Save a lookup table:

external/osm/taxi_paths.parquet   (ADEP_mvt, stand_ref, rwy_ident,
                                   path_length_m, n_turns)

Uses networkx for graph + shortest_path. Segments come from OSM ways with
aeroway in {taxiway, taxilane, runway}. Nodes = way vertices; edges =
consecutive vertex pairs of the same way, weighted by great-circle distance.

Stand centroids (already in external/osm/stands.parquet) and runway threshold
coords (from OurAirports runways.csv) are snapped to the nearest graph node.
"""
import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM_DIR = os.path.join(ROOT, "external", "osm")
RAW_DIR = os.path.join(OSM_DIR, "raw_taxi")
os.makedirs(RAW_DIR, exist_ok=True)

TARGET_ICAOS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
                "LFPG", "LIRF", "LTFM", "LSZH"]

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * r * math.asin(math.sqrt(a))


def bbox_for(lat, lon, half_km=4.0):
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(math.cos(math.radians(lat)), 0.3))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def fetch_taxi_ways(icao: str, lat: float, lon: float) -> list:
    """Return the parsed OSM ways for taxiways+runways at this airport."""
    cache = os.path.join(RAW_DIR, f"{icao}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    s, w, n, e = bbox_for(lat, lon)
    query = f"""
[out:json][timeout:180];
(
  way["aeroway"="taxiway"]({s},{w},{n},{e});
  way["aeroway"="taxilane"]({s},{w},{n},{e});
  way["aeroway"="runway"]({s},{w},{n},{e});
);
out body geom tags;
"""
    data = urllib.parse.urlencode({"data": query}).encode()
    for mirror in OVERPASS_MIRRORS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(mirror, data=data,
                                             headers={"User-Agent": "prc-dc-2026/1.0"})
                with urllib.request.urlopen(req, timeout=240) as r:
                    j = json.load(r)
                ways = [e for e in j.get("elements", []) if e["type"] == "way"]
                json.dump(ways, open(cache, "w"))
                print(f"  [{icao}] fetched {len(ways)} ways from {mirror}", flush=True)
                return ways
            except Exception as exc:
                print(f"  [{icao}] {mirror} attempt {attempt}: {exc}", flush=True)
                time.sleep(8 * (attempt + 1))
    return []


def build_graph(ways: list) -> nx.Graph:
    """Nodes keyed by rounded (lat, lon) so shared endpoints merge."""
    G = nx.Graph()
    def key(p):
        return (round(p["lat"], 6), round(p["lon"], 6))
    for w in ways:
        geom = w.get("geometry", [])
        tags = w.get("tags", {})
        if len(geom) < 2:
            continue
        for a, b in zip(geom, geom[1:]):
            ka, kb = key(a), key(b)
            if ka == kb:
                continue
            d = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
            if G.has_edge(ka, kb):
                # keep the shortest, or take a runway edge if it exists
                pass
            G.add_edge(ka, kb, weight=d,
                       aeroway=tags.get("aeroway"),
                       ref=tags.get("ref"))
    return G


def snap_to_graph(G: nx.Graph, lat: float, lon: float) -> tuple | None:
    """Return the nearest graph node key. O(n) — fine at airport scale."""
    if lat is None or lon is None or (isinstance(lat, float) and math.isnan(lat)):
        return None
    best = None
    best_d = math.inf
    for k in G.nodes:
        nlat, nlon = k
        d = haversine_m(lat, lon, nlat, nlon)
        if d < best_d:
            best_d = d
            best = k
    return best if best_d < 400 else None    # 400 m snap radius


def path_length_and_turns(G: nx.Graph, src, dst) -> tuple:
    try:
        nodes = nx.shortest_path(G, src, dst, weight="weight")
    except nx.NetworkXNoPath:
        return (None, None)
    if len(nodes) < 2:
        return (0.0, 0)
    length = sum(G[u][v]["weight"] for u, v in zip(nodes, nodes[1:]))
    # turn count: angle change > 30 deg between consecutive segments
    turns = 0
    for i in range(1, len(nodes) - 1):
        (y1, x1), (y2, x2), (y3, x3) = nodes[i-1], nodes[i], nodes[i+1]
        a1 = math.atan2(y2 - y1, x2 - x1)
        a2 = math.atan2(y3 - y2, x3 - x2)
        diff = math.degrees((a2 - a1 + math.pi) % (2*math.pi) - math.pi)
        if abs(diff) > 30:
            turns += 1
    return (length, turns)


def _pad_rwy(s: str) -> str:
    s = str(s).strip().upper()
    for i, ch in enumerate(s):
        if not ch.isdigit():
            num, suf = s[:i], s[i:]
            break
    else:
        num, suf = s, ""
    return ("0" + num + suf) if num and len(num) == 1 else s


def main():
    ap = pd.read_csv(os.path.join(ROOT, "external", "airports", "airports.csv"), low_memory=False)
    coords = ap[ap["ident"].isin(TARGET_ICAOS)][["ident", "latitude_deg", "longitude_deg"]] \
        .rename(columns={"ident": "ADEP_mvt"})
    stands = pd.read_parquet(os.path.join(OSM_DIR, "stands.parquet"))
    stands["stand_ref"] = stands["stand_ref"].astype(str).str.upper().str.strip()
    stands = stands.groupby(["ADEP_mvt", "stand_ref"], as_index=False)[["lat", "lon"]].mean()

    rwys = pd.read_csv(os.path.join(ROOT, "external", "airports", "runways.csv"),
                       low_memory=False)
    rwys = rwys[rwys["airport_ident"].isin(TARGET_ICAOS) & (rwys["closed"] == 0)]
    rwy_rows = []
    for _, row in rwys.iterrows():
        for end, lat_c, lon_c in [
            ("le_ident", "le_latitude_deg", "le_longitude_deg"),
            ("he_ident", "he_latitude_deg", "he_longitude_deg"),
        ]:
            ident = str(row[end]).strip().upper()
            if not ident or ident == "NAN":
                continue
            rwy_rows.append({"ADEP_mvt": row["airport_ident"],
                             "rwy_ident": _pad_rwy(ident),
                             "rwy_lat": row[lat_c],
                             "rwy_lon": row[lon_c]})
    rwy_df = pd.DataFrame(rwy_rows).drop_duplicates(subset=["ADEP_mvt", "rwy_ident"])

    out_rows = []
    for _, row in coords.iterrows():
        icao = row["ADEP_mvt"]
        print(f"\n=== {icao} ===", flush=True)
        ways = fetch_taxi_ways(icao, row["latitude_deg"], row["longitude_deg"])
        if not ways:
            print(f"  [{icao}] no ways; skipping", flush=True)
            continue
        G = build_graph(ways)
        # Restrict to largest connected component
        if G.number_of_nodes() == 0:
            continue
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        print(f"  [{icao}] graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

        # Snap stands and runway thresholds
        apt_stands = stands[stands["ADEP_mvt"] == icao].copy()
        apt_stands["node"] = [snap_to_graph(G, la, lo) for la, lo in
                              zip(apt_stands["lat"], apt_stands["lon"])]
        apt_rwys = rwy_df[rwy_df["ADEP_mvt"] == icao].copy()
        apt_rwys["node"] = [snap_to_graph(G, la, lo) for la, lo in
                            zip(apt_rwys["rwy_lat"], apt_rwys["rwy_lon"])]

        n_stand_snap = apt_stands["node"].notna().sum()
        n_rwy_snap = apt_rwys["node"].notna().sum()
        print(f"  [{icao}] snapped {n_stand_snap}/{len(apt_stands)} stands, "
              f"{n_rwy_snap}/{len(apt_rwys)} runway ends", flush=True)

        # All-pairs shortest paths from stand nodes -> runway nodes
        t0 = time.time()
        pairs_done = 0
        for _, sr in apt_stands.dropna(subset=["node"]).iterrows():
            for _, rr in apt_rwys.dropna(subset=["node"]).iterrows():
                length, turns = path_length_and_turns(G, sr["node"], rr["node"])
                if length is None:
                    continue
                out_rows.append({
                    "ADEP_mvt": icao, "stand_ref": sr["stand_ref"],
                    "rwy_ident": rr["rwy_ident"],
                    "path_length_m": round(length, 1),
                    "n_turns": turns,
                })
                pairs_done += 1
        print(f"  [{icao}] computed {pairs_done} paths in {time.time()-t0:.1f}s", flush=True)

    out = pd.DataFrame(out_rows)
    out_path = os.path.join(OSM_DIR, "taxi_paths.parquet")
    out.to_parquet(out_path)
    print(f"\nSaved {len(out):,} rows -> {out_path}   ({os.path.getsize(out_path)/1e6:.2f} MB)")
    if len(out):
        print("\nPer-airport summary:")
        print(out.groupby("ADEP_mvt").agg(
            n_pairs=("path_length_m", "count"),
            mean_length_m=("path_length_m", "mean"),
            mean_turns=("n_turns", "mean"),
        ).round(1).to_string())


if __name__ == "__main__":
    main()
