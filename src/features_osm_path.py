"""OSM-derived real taxi-path features:
   taxi_path_length_m    graph shortest-path from stand to runway threshold
   n_turns               turns > 30 deg along that path
   path_vs_haversine     ratio of graph path to straight-line distance

Replaces / augments the haversine `taxi_dist_m` from features_taxi_distance.py.
Lookup table produced by build_osm_taxi_paths.py.
"""
import math
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOKUP = os.path.join(ROOT, "external", "osm", "taxi_paths.parquet")


def _pad_rwy(s: str) -> str:
    s = str(s).strip().upper()
    for i, ch in enumerate(s):
        if not ch.isdigit():
            num, suf = s[:i], s[i:]
            break
    else:
        num, suf = s, ""
    return ("0" + num + suf) if num and len(num) == 1 else s


OSM_PATH_NUM_COLS = ["taxi_path_length_m", "n_turns", "path_vs_haversine"]


def add_osm_path(dep: pd.DataFrame) -> pd.DataFrame:
    """dep needs: ADEP_mvt, STAND_mvt, RUNWAY_mvt, taxi_dist_m."""
    if not os.path.exists(LOOKUP):
        # Lookup not built yet; return NaN placeholders so training scripts still work
        out = dep.copy()
        for c in OSM_PATH_NUM_COLS:
            out[c] = np.nan
        return out

    lk = pd.read_parquet(LOOKUP)
    lk["stand_ref"] = lk["stand_ref"].astype(str).str.upper().str.strip()
    lk["rwy_ident"] = lk["rwy_ident"].astype(str).str.upper().str.strip()

    out = dep.copy()
    out["_stand_norm"] = out["STAND_mvt"].astype(str).str.upper().str.strip()
    out["_rwy_norm"] = out["RUNWAY_mvt"].astype(str).map(_pad_rwy)

    out = out.merge(lk.rename(columns={"stand_ref": "_stand_norm",
                                       "rwy_ident": "_rwy_norm"}),
                    on=["ADEP_mvt", "_stand_norm", "_rwy_norm"], how="left")

    # Derived ratio: how much longer is real path than straight-line?
    if "taxi_dist_m" in out.columns:
        out["path_vs_haversine"] = out["path_length_m"] / out["taxi_dist_m"].replace(0, np.nan)
    else:
        out["path_vs_haversine"] = np.nan

    out = out.rename(columns={"path_length_m": "taxi_path_length_m"})
    return out.drop(columns=["_stand_norm", "_rwy_norm"])


if __name__ == "__main__":
    # smoke test on training subset
    from features_weather import TARGET_ICAOS
    from features_taxi_distance import add_taxi_distance
    import glob
    p = os.path.join(ROOT, "training", "training_2025-06-01_2025-07-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].head(30000)
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    df = add_taxi_distance(df)
    out = add_osm_path(df)
    print(f"Rows: {len(out):,}")
    for c in OSM_PATH_NUM_COLS:
        cov = out[c].notna().mean() * 100
        print(f"  {c:30s} coverage {cov:5.1f}%")
    if out["taxi_path_length_m"].notna().any():
        print("\nSample per airport (path vs haversine ratio):")
        good = out.dropna(subset=["taxi_path_length_m", "taxi_dist_m"])
        for a, g in good.groupby("ADEP_mvt"):
            if len(g) < 100: continue
            print(f"  {a}: n={len(g):,}  mean_ratio={g['path_vs_haversine'].mean():.2f}  "
                  f"corr(path,taxi)={g['taxi_path_length_m'].corr(g['TAXITIME_SEC_mvt']):+.3f}  "
                  f"corr(haversine,taxi)={g['taxi_dist_m'].corr(g['TAXITIME_SEC_mvt']):+.3f}")
