"""Consolidate 57 filtered OPDI event chunks into two tables:

  external/opdi/events_all.parquet
    All ground-taxi events at the 10 target airports.
    Columns: flight_id, type, event_time, osm_airport, osn_flight_id, osm_ref

  external/opdi/taxi_out.parquet
    Per-OPDI-flight actual taxi-out at each target airport.
    Columns: flight_id, osm_airport, osn_flight_id,
             off_block_ts (exit-parking_position),
             runway_entry_ts (first entry-runway),
             actual_taxi_sec, n_taxiway_events, crossed_other_runway

Uses OSN-derived data. Called only after user confirmed OPDI is allowed.
"""
import glob
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(ROOT, "external", "opdi", "filtered")
OUT_EVENTS = os.path.join(ROOT, "external", "opdi", "events_all.parquet")
OUT_TAXI = os.path.join(ROOT, "external", "opdi", "taxi_out.parquet")

KEEP_TYPES = {
    "exit-parking_position",
    "entry-runway",
    "entry-taxiway",
    "exit-runway",
}


def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.parquet")))
    print(f"Loading {len(files)} filtered chunks...")
    t0 = time.time()
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df = df[df["type"].isin(KEEP_TYPES)]
        frames.append(df[["flight_id", "type", "event_time",
                          "osm_airport", "osn_flight_id", "osm_ref"]])
    events = pd.concat(frames, ignore_index=True)
    print(f"  {len(events):,} events in {time.time()-t0:.1f}s")

    events["event_time"] = pd.to_datetime(events["event_time"], utc=True, errors="coerce") \
        .astype("datetime64[us, UTC]")
    events["osn_flight_id"] = events["osn_flight_id"].astype(str).str.strip().str.upper()
    events["osm_airport"] = events["osm_airport"].astype(str).str.upper().str.strip()
    events = events.dropna(subset=["event_time"])
    events = events.sort_values(["osm_airport", "event_time"]).reset_index(drop=True)
    print(f"  {len(events):,} events after cleaning")

    events.to_parquet(OUT_EVENTS)
    print(f"  saved -> {OUT_EVENTS}  ({os.path.getsize(OUT_EVENTS)/1e6:.1f} MB)")

    # Build per-flight taxi records
    print("\nBuilding per-flight taxi table...")
    t0 = time.time()
    # For each (osm_airport, flight_id): first exit-parking + first entry-runway after it
    exit_park = events[events["type"] == "exit-parking_position"].copy()
    entry_rwy = events[events["type"] == "entry-runway"].copy()
    entry_tax = events[events["type"] == "entry-taxiway"].copy()

    # First off-block per (airport, flight_id)
    off_block = exit_park.sort_values("event_time").drop_duplicates(
        subset=["osm_airport", "flight_id"], keep="first"
    )[["osm_airport", "flight_id", "osn_flight_id", "event_time"]] \
        .rename(columns={"event_time": "off_block_ts"})

    # Get first runway-entry AFTER off_block, per (airport, flight_id)
    er = entry_rwy[["osm_airport", "flight_id", "event_time", "osm_ref"]] \
        .rename(columns={"event_time": "runway_entry_ts", "osm_ref": "runway_entry_ref"})

    merged = off_block.merge(er, on=["osm_airport", "flight_id"], how="left")
    merged = merged[merged["runway_entry_ts"] > merged["off_block_ts"]]
    # keep first entry-runway per flight
    merged = merged.sort_values("runway_entry_ts").drop_duplicates(
        subset=["osm_airport", "flight_id"], keep="first")

    merged["actual_taxi_sec"] = (merged["runway_entry_ts"] - merged["off_block_ts"]) \
        .dt.total_seconds()

    # Filter unreasonable values
    n0 = len(merged)
    merged = merged[merged["actual_taxi_sec"].between(30, 7200)]
    print(f"  {n0:,} -> {len(merged):,} after 30-7200s filter")

    # n_taxiway_events per (airport, flight_id) — hop count on the taxiway network
    tax_count = entry_tax.groupby(["osm_airport", "flight_id"]).size().rename("n_taxiway_events")
    merged = merged.merge(tax_count.reset_index(), on=["osm_airport", "flight_id"], how="left")
    merged["n_taxiway_events"] = merged["n_taxiway_events"].fillna(0).astype(int)

    # crossed_other_runway: any entry-runway on a DIFFERENT runway between off_block and runway_entry
    # (i.e. runway crossing on the way to the departure runway)
    er_all = entry_rwy[["osm_airport", "flight_id", "event_time", "osm_ref"]]
    def _cross_flag(row):
        # inefficient row-by-row; only applied to per-flight table
        return None   # implemented per-airport below
    # Batch approach: for each airport, group events by flight
    def crossed(sub):
        # sub is DataFrame with cols event_time, osm_ref (only entry-runway rows for this flight)
        if len(sub) <= 1:
            return 0
        return int(sub["osm_ref"].nunique() > 1)
    er_grouped = er_all.groupby(["osm_airport", "flight_id"]) \
        .apply(crossed, include_groups=False).rename("crossed_other_runway").reset_index()
    merged = merged.merge(er_grouped, on=["osm_airport", "flight_id"], how="left")
    merged["crossed_other_runway"] = merged["crossed_other_runway"].fillna(0).astype(int)

    print(f"  built in {time.time()-t0:.1f}s")
    print(f"  {len(merged):,} taxi records")
    print("\nSummary per airport:")
    print(merged.groupby("osm_airport").agg(
        n=("actual_taxi_sec", "size"),
        mean_taxi=("actual_taxi_sec", "mean"),
        median_taxi=("actual_taxi_sec", "median"),
        crossings_rate=("crossed_other_runway", "mean"),
    ).round(2).to_string())

    merged.to_parquet(OUT_TAXI)
    print(f"\nSaved -> {OUT_TAXI}  ({os.path.getsize(OUT_TAXI)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
