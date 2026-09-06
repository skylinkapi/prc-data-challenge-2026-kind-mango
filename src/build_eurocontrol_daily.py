"""Consolidate 5 Eurocontrol daily-per-airport Excel files into one clean
parquet: external/eurocontrol/daily_features.parquet.

Columns per (date, ICAO): daily traffic, ATC/all pre-departure delay minutes
and delayed-flight counts, ATFM slot adherence, arrival ATFM delay minutes.
"""
import os
import pandas as pd
from features_weather import TARGET_ICAOS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external", "eurocontrol")
OUT = os.path.join(SRC, "daily_features.parquet")

FILES = {
    "ATC_Pre-Departure_Delay.xlsx":      ["FLT_DEP_IFR_2", "DLY_ATC_PRE_2"],
    "All_Pre-Departure_Delay.xlsx":      ["FLT_DEP_IFR_2", "DLY_ALL_PRE_2"],
    "ATFM_Slot_Adherence.xlsx":          ["FLT_DEP_1", "FLT_DEP_REG_1",
                                          "FLT_DEP_OUT_EARLY_1", "FLT_DEP_IN_1",
                                          "FLT_DEP_OUT_LATE_1"],
    "Airport_Arrival_ATFM_Delay.xlsx":   ["FLT_ARR_1", "DLY_APT_ARR_1",
                                          "DLY_APT_ARR_C_1", "DLY_APT_ARR_W_1",
                                          "DLY_APT_ARR_D_1", "FLT_ARR_1_DLY",
                                          "FLT_ARR_1_DLY_15"],
    "Airport_Traffic.xlsx":              ["FLT_DEP_IFR_2", "FLT_ARR_IFR_2", "FLT_TOT_IFR_2"],
}


def load_one(fname: str, keep_cols: list) -> pd.DataFrame:
    p = os.path.join(SRC, fname)
    df = pd.read_excel(p, sheet_name="DATA")
    df["FLT_DATE"] = pd.to_datetime(df["FLT_DATE"], errors="coerce").dt.date
    df = df[df["APT_ICAO"].isin(TARGET_ICAOS)]
    df = df[(df["YEAR"].between(2025, 2026))]
    keep = ["FLT_DATE", "APT_ICAO"] + [c for c in keep_cols if c in df.columns]
    df = df[keep].copy()
    # rename metric cols with file prefix to disambiguate
    prefix = fname.replace(".xlsx", "").replace("-", "_").lower()
    ren = {c: f"{prefix}__{c}" for c in keep_cols if c in df.columns}
    df = df.rename(columns=ren)
    return df


def main():
    merged = None
    for fname, cols in FILES.items():
        print(f"reading {fname} ...")
        d = load_one(fname, cols)
        print(f"  {len(d):,} rows, cols: {[c for c in d.columns if c not in ('FLT_DATE','APT_ICAO')]}")
        if merged is None:
            merged = d
        else:
            # drop duplicate metric cols across files (FLT_DEP_IFR_2 repeats)
            dupes = [c for c in d.columns if c in merged.columns and c not in ("FLT_DATE", "APT_ICAO")]
            d = d.drop(columns=dupes)
            merged = merged.merge(d, on=["FLT_DATE", "APT_ICAO"], how="outer")

    merged = merged.sort_values(["APT_ICAO", "FLT_DATE"]).reset_index(drop=True)
    print(f"\nFinal shape: {merged.shape}")
    print(f"Airports: {sorted(merged['APT_ICAO'].unique())}")
    print(f"Date range: {merged['FLT_DATE'].min()} -> {merged['FLT_DATE'].max()}")
    print(f"Missing per col:")
    print((merged.isna().mean()*100).round(2).to_string())
    merged.to_parquet(OUT)
    print(f"\nSaved -> {OUT}   ({os.path.getsize(OUT)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
