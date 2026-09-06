"""Download VRS StandingData aircraft-type lookup, consolidate to a single
parquet, keyed by ICAO type code with one active row per code.

Source: github.com/vradarserver/standing-data/tree/main/model-type/schema-01
"""
import io
import os
import time
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "external", "vrs")
os.makedirs(OUT_DIR, exist_ok=True)

BASE = "https://raw.githubusercontent.com/vradarserver/standing-data/main/model-type/schema-01"
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def fetch_csv(letter: str) -> pd.DataFrame:
    url = f"{BASE}/{letter}.csv"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prc-2026/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
            return df
        except Exception as e:
            print(f"  {letter}: attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return pd.DataFrame()


def main():
    frames = []
    for L in LETTERS:
        df = fetch_csv(L)
        if len(df):
            print(f"  {L}: {len(df):,} rows")
            frames.append(df)
    all_ = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows: {len(all_):,}")
    print(f"Unique ICAO codes: {all_['ICAO'].nunique():,}")

    # For each ICAO, take the first ACTIVE row. Fall back to first row if no active.
    active = all_[all_["IsActive"] == 1].copy()
    dedup_active = active.drop_duplicates(subset="ICAO", keep="first")
    still_missing = set(all_["ICAO"]) - set(dedup_active["ICAO"])
    dedup_inactive = all_[all_["ICAO"].isin(still_missing)].drop_duplicates(subset="ICAO", keep="first")
    dedup = pd.concat([dedup_active, dedup_inactive], ignore_index=True)
    print(f"After dedupe: {len(dedup):,} ICAO codes")

    keep = ["ICAO", "Manufacturer", "Model", "Engines",
            "EngineTypeCode", "SpeciesCode", "WakeTurbulenceCode"]
    dedup = dedup[keep].rename(columns={"ICAO": "AIRCRAFT_TYPE_mvt"})
    dedup["Engines"] = pd.to_numeric(dedup["Engines"], errors="coerce")

    out_path = os.path.join(OUT_DIR, "aircraft_types.parquet")
    dedup.to_parquet(out_path)
    print(f"\nSaved -> {out_path}   ({os.path.getsize(out_path)/1024:.1f} KB)")

    # Sample
    print("\nSample rows:")
    print(dedup.head(15).to_string(index=False))
    print("\nSpecies distribution:")
    print(dedup["SpeciesCode"].value_counts().to_string())
    print("\nEngine count distribution:")
    print(dedup["Engines"].value_counts().to_string())
    print("\nEngine type distribution:")
    print(dedup["EngineTypeCode"].value_counts().to_string())
    print("\nTop 15 manufacturers:")
    print(dedup["Manufacturer"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
