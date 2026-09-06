"""Stand-to-runway straight-line distance (meters), a physical prior for taxi
time. Uses OSM stand centroids and OurAirports runway threshold coordinates.

Straight-line distance is a rough proxy for the real taxi path (~0.7 correlation
in aviation studies); still the strongest single-feature physical signal.
"""
import math
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDS_PARQUET = os.path.join(ROOT, "external", "osm", "stands.parquet")
RUNWAYS_CSV = os.path.join(ROOT, "external", "airports", "runways.csv")

TARGET_ICAOS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
                "LFPG", "LIRF", "LTFM", "LSZH"]


def _pad(s: str) -> str:
    """Zero-pad single-digit runway prefix: '6R' -> '06R'."""
    s = str(s).strip().upper()
    for i, ch in enumerate(s):
        if not ch.isdigit():
            num, suf = s[:i], s[i:]
            break
    else:
        num, suf = s, ""
    if num and len(num) == 1:
        return "0" + num + suf
    return s


def load_stand_coords() -> pd.DataFrame:
    st = pd.read_parquet(STANDS_PARQUET)
    st["stand_ref"] = st["stand_ref"].astype(str).str.upper().str.strip()
    # Average across duplicate OSM refs for the same stand at an airport
    return st.groupby(["ADEP_mvt", "stand_ref"], as_index=False)[["lat", "lon"]].mean()


def load_runway_thresholds() -> pd.DataFrame:
    r = pd.read_csv(RUNWAYS_CSV, low_memory=False)
    r = r[r["airport_ident"].isin(TARGET_ICAOS) & (r["closed"] == 0)]
    rows = []
    for _, row in r.iterrows():
        for end, hdg, lat, lon in [
            ("le_ident", "le_heading_degT", "le_latitude_deg", "le_longitude_deg"),
            ("he_ident", "he_heading_degT", "he_latitude_deg", "he_longitude_deg"),
        ]:
            ident = str(row[end]).strip().upper()
            if not ident or ident == "NAN":
                continue
            rows.append({
                "ADEP_mvt": row["airport_ident"],
                "rwy_ident": _pad(ident),
                "rwy_thr_lat": row[lat],
                "rwy_thr_lon": row[lon],
                "rwy_heading": row[hdg],
            })
    return pd.DataFrame(rows).drop_duplicates(subset=["ADEP_mvt", "rwy_ident"])


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_taxi_distance(dep: pd.DataFrame) -> pd.DataFrame:
    stands = load_stand_coords()
    rwys = load_runway_thresholds()
    out = dep.copy()
    out["_stand_norm"] = out["STAND_mvt"].astype(str).str.upper().str.strip()
    out["_rwy_norm"] = out["RUNWAY_mvt"].astype(str).map(_pad)

    out = out.merge(stands.rename(columns={"stand_ref": "_stand_norm",
                                           "lat": "stand_lat",
                                           "lon": "stand_lon"}),
                    on=["ADEP_mvt", "_stand_norm"], how="left")
    out = out.merge(rwys.rename(columns={"rwy_ident": "_rwy_norm"}),
                    on=["ADEP_mvt", "_rwy_norm"], how="left")

    out["taxi_dist_m"] = haversine_m(
        out["stand_lat"].values, out["stand_lon"].values,
        out["rwy_thr_lat"].values, out["rwy_thr_lon"].values,
    )
    out["taxi_dist_known"] = (~out["taxi_dist_m"].isna()).astype(np.int8)
    return out.drop(columns=["_stand_norm", "_rwy_norm"])


TAXI_NUM_COLS = ["taxi_dist_m", "taxi_dist_known",
                 "stand_lat", "stand_lon", "rwy_thr_lat", "rwy_thr_lon"]


if __name__ == "__main__":
    import glob
    from features_weather import TARGET_ICAOS as TARGETS
    p = os.path.join(ROOT, "training", "training_2025-06-01_2025-07-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGETS)].copy()
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    df["TAXITIME_SEC_mvt"] = df["TAXITIME_SEC_mvt"].astype(float)
    out = add_taxi_distance(df)
    print(f"Rows: {len(out):,}   distance known: {out['taxi_dist_known'].mean()*100:.1f}%")
    print("\nSample per airport (distance vs taxi time):")
    for apt in TARGETS[:10]:
        sub = out[(out["ADEP_mvt"] == apt) & out["taxi_dist_known"].eq(1)]
        if len(sub) < 100:
            print(f"  {apt}: too few points ({len(sub)})")
            continue
        corr = sub["taxi_dist_m"].corr(sub["TAXITIME_SEC_mvt"])
        print(f"  {apt}: n={len(sub):,}  corr(dist,taxi)={corr:+.3f}  "
              f"median dist={sub['taxi_dist_m'].median():.0f} m")
