"""VRS StandingData aircraft-type features.

Joins on AIRCRAFT_TYPE_mvt (ICAO type code). Adds:
  vrs_manufacturer  — categorical (Airbus/Boeing/Embraer/ATR/...)
  vrs_engines       — numeric engine count (1/2/3/4/...)
  vrs_engine_type   — categorical (J=jet, T=turboprop, P=piston, E=electric)
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOKUP = os.path.join(ROOT, "external", "vrs", "aircraft_types.parquet")

VRS_CAT_COLS = ["vrs_manufacturer", "vrs_engine_type"]
VRS_NUM_COLS = ["vrs_engines"]


def add_vrs(dep: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(LOOKUP):
        out = dep.copy()
        for c in VRS_CAT_COLS + VRS_NUM_COLS:
            out[c] = None if c in VRS_CAT_COLS else float("nan")
        return out
    v = pd.read_parquet(LOOKUP)
    v = v.rename(columns={
        "Manufacturer": "vrs_manufacturer",
        "Engines": "vrs_engines",
        "EngineTypeCode": "vrs_engine_type",
    })
    keep = ["AIRCRAFT_TYPE_mvt", "vrs_manufacturer", "vrs_engines", "vrs_engine_type"]
    v = v[keep].drop_duplicates(subset="AIRCRAFT_TYPE_mvt", keep="first")
    return dep.merge(v, on="AIRCRAFT_TYPE_mvt", how="left")


if __name__ == "__main__":
    import glob
    from features_weather import TARGET_ICAOS
    p = os.path.join(ROOT, "training", "training_2025-06-01_2025-07-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].head(30000)
    out = add_vrs(df)
    print(f"Rows: {len(out):,}")
    for c in VRS_CAT_COLS + VRS_NUM_COLS:
        cov = out[c].notna().mean() * 100
        print(f"  {c:20s} coverage {cov:5.1f}%")
    print("\nManufacturer counts (top 15) among our DEPs:")
    print(out["vrs_manufacturer"].value_counts().head(15).to_string())
    print("\nEngine counts:")
    print(out["vrs_engines"].value_counts().to_string())
    print("\nEngine type:")
    print(out["vrs_engine_type"].value_counts().to_string())
