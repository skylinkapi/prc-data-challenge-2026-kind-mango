"""Historical OPDI taxi climatology per (airport, month, day-of-week, hour-bin).

Uses 2022-2024 chunks ONLY to avoid leakage into 2025 training or 2026 test.
For each per-flight taxi record (pushback -> runway entry), aggregates to
per-bucket median and count.

Output: external/opdi/climatology.parquet
  columns: ADEP_mvt, month, dow, hour_bin,
           clim_median_taxi_sec, clim_count, clim_mean_sched_delay
"""
import glob
import os
import time
import gc
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(ROOT, "external", "opdi", "filtered")
OUT = os.path.join(ROOT, "external", "opdi", "climatology.parquet")


def _hour_bin(h: pd.Series) -> pd.Series:
    return pd.cut(h, bins=[-1, 5, 9, 13, 17, 21, 24],
                  labels=["night", "earlyam", "midam", "midpm", "evening", "latenight"])


def main():
    t0 = time.time()
    print("Loading 2022-2024 chunks for climatology...")
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.parquet")))
    # Filter to 2022-2024
    hist_files = [f for f in files
                  if any(f"{y}" in os.path.basename(f) for y in ["2022", "2023", "2024"])
                  and "2025" not in os.path.basename(f).split("_")[3]]
    print(f"  {len(hist_files)} historical chunks")

    frames = [pd.read_parquet(f) for f in hist_files]
    ev = pd.concat(frames, ignore_index=True)
    print(f"  {len(ev):,} events")
    del frames; gc.collect()

    # Build per-flight taxi (first exit-parking or first entry-taxiway -> first entry-runway)
    ep = ev[ev["type"] == "exit-parking_position"].sort_values("event_time") \
        .drop_duplicates(["osm_airport", "flight_id"], keep="first") \
        [["osm_airport", "flight_id", "event_time"]] \
        .rename(columns={"event_time": "push_ts"})
    et = ev[ev["type"] == "entry-taxiway"].sort_values("event_time") \
        .drop_duplicates(["osm_airport", "flight_id"], keep="first") \
        [["osm_airport", "flight_id", "event_time"]] \
        .rename(columns={"event_time": "tax_ts"})
    er = ev[ev["type"] == "entry-runway"].sort_values("event_time") \
        .drop_duplicates(["osm_airport", "flight_id"], keep="first") \
        [["osm_airport", "flight_id", "event_time"]] \
        .rename(columns={"event_time": "rwy_ts"})
    del ev; gc.collect()

    m = er.merge(ep, on=["osm_airport", "flight_id"], how="left") \
          .merge(et, on=["osm_airport", "flight_id"], how="left")
    m["pushback_ts"] = m["push_ts"].fillna(m["tax_ts"])
    m = m.dropna(subset=["pushback_ts", "rwy_ts"])
    m = m[m["rwy_ts"] > m["pushback_ts"]]
    m["actual_taxi_sec"] = (m["rwy_ts"] - m["pushback_ts"]).dt.total_seconds()
    m = m[m["actual_taxi_sec"].between(30, 7200)].copy()

    m["month"] = pd.to_datetime(m["rwy_ts"]).dt.month
    m["dow"] = pd.to_datetime(m["rwy_ts"]).dt.dayofweek
    m["hour"] = pd.to_datetime(m["rwy_ts"]).dt.hour
    m["hour_bin"] = _hour_bin(m["hour"])

    print(f"\n  built {len(m):,} per-flight taxi records")

    # Aggregate: per (airport, month, dow, hour_bin) median + count
    agg = m.groupby(["osm_airport", "month", "dow", "hour_bin"], observed=True) \
        .agg(clim_median_taxi=("actual_taxi_sec", "median"),
             clim_mean_taxi=("actual_taxi_sec", "mean"),
             clim_count=("actual_taxi_sec", "size")) \
        .reset_index()

    # Also coarser: (airport, month, hour_bin) as fallback
    agg2 = m.groupby(["osm_airport", "month", "hour_bin"], observed=True) \
        .agg(clim_median_taxi_2=("actual_taxi_sec", "median"),
             clim_count_2=("actual_taxi_sec", "size")).reset_index()

    print(f"\nFine buckets: {len(agg):,}")
    print(f"Coarse buckets: {len(agg2):,}")

    agg = agg.rename(columns={"osm_airport": "ADEP_mvt"})
    agg2 = agg2.rename(columns={"osm_airport": "ADEP_mvt"})
    combined = agg.merge(agg2, on=["ADEP_mvt", "month", "hour_bin"], how="left")

    combined.to_parquet(OUT)
    print(f"\nSaved -> {OUT}   ({os.path.getsize(OUT)/1024:.1f} KB)")
    print(f"\nPer-airport row counts:")
    print(combined.groupby("ADEP_mvt").size().to_string())
    print(f"\nTotal build time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
