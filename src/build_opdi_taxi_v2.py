"""Improved OPDI per-flight taxi table (v2).

Uses first `entry-taxiway` as pushback proxy when `exit-parking_position` is
missing. Aircraft enter the taxi network from the gate → the first taxiway
event is a good pushback timestamp. Coverage extends from LSZH+EDDF only
(exit-parking sparse elsewhere) to essentially all 9 non-LTFM airports.

Output: external/opdi/taxi_out_v2.parquet
  columns: osm_airport, flight_id, osn_flight_id,
           pushback_ts, runway_entry_ts, actual_taxi_sec,
           pushback_source ('exit_park' or 'entry_tax')
"""
import os
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = os.path.join(ROOT, "external", "opdi", "events_all.parquet")
OUT = os.path.join(ROOT, "external", "opdi", "taxi_out_v2.parquet")


def main():
    t0 = time.time()
    print("Loading OPDI events...")
    ev = pd.read_parquet(EVENTS)
    print(f"  {len(ev):,} events")

    # First exit-parking_position per (airport, flight)
    ep = ev[ev["type"] == "exit-parking_position"].sort_values("event_time") \
        .drop_duplicates(["osm_airport", "flight_id"], keep="first") \
        [["osm_airport", "flight_id", "osn_flight_id", "event_time"]] \
        .rename(columns={"event_time": "exit_park_ts"})

    # First entry-taxiway per (airport, flight)
    et = ev[ev["type"] == "entry-taxiway"].sort_values("event_time") \
        .drop_duplicates(["osm_airport", "flight_id"], keep="first") \
        [["osm_airport", "flight_id", "event_time"]] \
        .rename(columns={"event_time": "entry_tax_ts"})

    # First entry-runway per (airport, flight)
    er = ev[ev["type"] == "entry-runway"].sort_values("event_time") \
        .drop_duplicates(["osm_airport", "flight_id"], keep="first") \
        [["osm_airport", "flight_id", "event_time"]] \
        .rename(columns={"event_time": "runway_entry_ts"})

    # Merge all three
    m = er.merge(ep, on=["osm_airport", "flight_id"], how="left") \
          .merge(et, on=["osm_airport", "flight_id"], how="left")

    # Pushback = exit_park if present, else entry_tax
    m["pushback_ts"] = m["exit_park_ts"]
    m["pushback_source"] = "exit_park"
    fill_mask = m["pushback_ts"].isna() & m["entry_tax_ts"].notna()
    m.loc[fill_mask, "pushback_ts"] = m.loc[fill_mask, "entry_tax_ts"]
    m.loc[fill_mask, "pushback_source"] = "entry_tax"

    # Compute taxi
    m = m.dropna(subset=["pushback_ts", "runway_entry_ts"])
    m = m[m["runway_entry_ts"] > m["pushback_ts"]]
    m["actual_taxi_sec"] = (m["runway_entry_ts"] - m["pushback_ts"]).dt.total_seconds()
    n0 = len(m)
    m = m[m["actual_taxi_sec"].between(30, 7200)]
    print(f"  taxi records: {n0:,} raw -> {len(m):,} after 30-7200s filter")

    keep = ["osm_airport", "flight_id", "osn_flight_id",
            "pushback_ts", "runway_entry_ts", "actual_taxi_sec", "pushback_source"]
    m[keep].to_parquet(OUT)

    print("\nPer-airport coverage (records):")
    src_split = m.groupby("osm_airport")["pushback_source"].value_counts().unstack(fill_value=0)
    print(src_split.to_string())
    print("\nPer-airport summary:")
    print(m.groupby("osm_airport").agg(
        n=("actual_taxi_sec", "size"),
        mean=("actual_taxi_sec", "mean"),
        median=("actual_taxi_sec", "median"),
    ).round(1).to_string())
    print(f"\nBuilt in {time.time()-t0:.1f}s. Saved -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
