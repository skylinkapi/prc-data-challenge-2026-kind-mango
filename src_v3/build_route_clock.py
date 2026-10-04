"""Third-edition K3: route distance from the assigned stand to the first ADS-B ground point.

On 89.8 % of the segment-only rows the aircraft is first seen while
taxiing, and the error grows with the distance of that first point from
the stand (`budget_v97.holdout.json`, `segment_rows`). The served columns
hold straight-line distances only. This builder snaps the assigned stand
and the first segment point to the OSM taxi graph (`src/build_osm_taxi_paths.py`
caches) and writes the shortest-path distance between them, and a clock
that adds the time to cover it at `ADSB_TAXI_SPEED_MS`. The 2025 training
days and the 2026 ranking days go through the same code.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_osm_taxi_paths import RAW_DIR, build_graph
from src_v3 import build_adsb_clock as bc
from src_v3 import config as C
from src_v3.build_nm_clocks_v73 import load_dep

OUT_TRAIN = C.ADSB_DIR / "route_clock_train.parquet"
OUT_RANK = C.ADSB_DIR / "route_clock_rank.parquet"
FEATURES = ["adsb_route_d0", "adsb_route_t"]
log = logging.getLogger(__name__)


def airport_graph(icao: str) -> tuple[nx.Graph, np.ndarray, cKDTree] | None:
    """Largest connected taxi graph of the airport, its node keys and a KD tree in metres."""
    path = Path(RAW_DIR) / f"{icao}.json"
    if not path.exists():
        return None
    g = build_graph(json.loads(path.read_text()))
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    nodes = np.array(list(g.nodes))
    k = np.cos(np.radians(nodes[:, 0].mean()))
    return g, nodes, cKDTree(np.c_[nodes[:, 0], nodes[:, 1] * k] * bc.M_PER_DEG)


def snap(tree: cKDTree, nodes: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Index of the nearest graph node within `ROUTE_SNAP_M`, or -1."""
    k = np.cos(np.radians(nodes[:, 0].mean()))
    ok = ~(np.isnan(lat) | np.isnan(lon))
    out = np.full(len(lat), -1)
    if ok.any():
        d, i = tree.query(np.c_[lat[ok], lon[ok] * k] * bc.M_PER_DEG)
        out[ok] = np.where(d <= C.ROUTE_SNAP_M, i, -1)
    return out


def route_d0(f: pd.DataFrame, graph: tuple) -> np.ndarray:
    """Shortest-path metres from the snapped stand node to the snapped first-point node of each row."""
    g, nodes, tree = graph
    s, p = snap(tree, nodes, f["own_lat"].values, f["own_lon"].values), snap(tree, nodes, f["lat"].values, f["lon"].values)
    out = np.full(len(f), np.nan)
    for node in np.unique(s[(s >= 0) & (p >= 0)]):
        dist = nx.single_source_dijkstra_path_length(g, tuple(nodes[node]), cutoff=C.ROUTE_CUTOFF_M, weight="weight")
        rows = np.flatnonzero((s == node) & (p >= 0))
        out[rows] = [dist.get(tuple(nodes[j]), np.nan) for j in p[rows]]
    return out


def first_points(dep: pd.DataFrame) -> pd.DataFrame:
    """Per departure: the first point of its last ground segment, with `adsb_seg`, and the assigned stand position."""
    dep = bc.own_stand(dep.assign(mvt=dep["MVT_TIME_UTC_mvt"].astype("datetime64[us, UTC]"),
                                  CALLSIGN_flt=dep["CALLSIGN_flt"].str.strip()))
    parts = []
    for d, rows in dep.groupby(dep["mvt"].dt.floor("D")):
        if (bc.POINTS / f"{d:%Y.%m.%d}.parquet").exists():
            g = bc.segment(rows, bc.load_points(d))
            parts.append(g.groupby("MVT_ID_mvt", sort=False)[["lat", "lon", "dt"]].first().reset_index())
    pts = pd.concat(parts, ignore_index=True).rename(columns={"dt": "adsb_seg"})
    return dep[["MVT_ID_mvt", "ADEP_mvt", "own_lat", "own_lon"]].merge(pts, on="MVT_ID_mvt", how="left", validate="1:1")


def build(dep: pd.DataFrame) -> pd.DataFrame:
    """K3 columns for every departure in `dep`, keyed on `MVT_ID_mvt`."""
    f = first_points(dep)
    f["adsb_route_d0"] = np.nan
    for icao, rows in f.groupby("ADEP_mvt").groups.items():
        graph = airport_graph(icao)
        if graph is not None:
            f.loc[rows, "adsb_route_d0"] = route_d0(f.loc[rows], graph)
    f["adsb_route_t"] = f["adsb_seg"] + f["adsb_route_d0"] / C.ADSB_TAXI_SPEED_MS
    log.info("%d departures; route distance on %.3f of the rows with a segment", len(f),
             f.loc[f["adsb_seg"].notna(), "adsb_route_d0"].notna().mean())
    return f[["MVT_ID_mvt", *FEATURES]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    extra = ["CALLSIGN_flt", "STAND_mvt"]
    build(load_dep([C.RANK], extra)).to_parquet(OUT_RANK)
    build(load_dep(sorted(C.TRAIN_DIR.glob("training_*.parquet")), extra)).to_parquet(OUT_TRAIN)


if __name__ == "__main__":
    main()
