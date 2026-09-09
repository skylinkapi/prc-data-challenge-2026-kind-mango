"""Aircraft performance features from openap (physical model of ~37 major types).

Provides real MTOW, empty weight, wing dimensions, engine count/type per ICAO
type. Covers most commercial jets in our data; unknown types get NaN
(LightGBM handles natively).

Features added:
  op_mtow           max takeoff weight (kg)
  op_oew            operating empty weight (kg)
  op_wingspan       wing span (m)
  op_fuselage_len   fuselage length (m)
  op_n_engines      number of engines
  op_max_pax        max passenger count
"""
import os
import numpy as np
import pandas as pd

try:
    from openap import prop as openap_prop
    OPENAP_AVAILABLE = True
except ImportError:
    OPENAP_AVAILABLE = False

OPENAP_NUM_COLS = ["op_mtow", "op_oew", "op_wingspan", "op_fuselage_len",
                   "op_n_engines", "op_max_pax"]


def _build_lookup() -> pd.DataFrame:
    """Return a DataFrame keyed by upper-case ICAO type with the feature cols."""
    if not OPENAP_AVAILABLE:
        return pd.DataFrame(columns=["AIRCRAFT_TYPE_mvt"] + OPENAP_NUM_COLS)
    rows = []
    for icao in openap_prop.available_aircraft():
        try:
            a = openap_prop.aircraft(icao)
            rows.append({
                "AIRCRAFT_TYPE_mvt": icao.upper(),
                "op_mtow":          a.get("mtow"),
                "op_oew":           a.get("oew"),
                "op_wingspan":      (a.get("wing") or {}).get("span"),
                "op_fuselage_len":  (a.get("fuselage") or {}).get("length"),
                "op_n_engines":     (a.get("engine") or {}).get("number"),
                "op_max_pax":       (a.get("pax") or {}).get("max"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


_lookup_cache = None


def add_openap(dep: pd.DataFrame) -> pd.DataFrame:
    global _lookup_cache
    if _lookup_cache is None:
        _lookup_cache = _build_lookup()
    return dep.merge(_lookup_cache, on="AIRCRAFT_TYPE_mvt", how="left")


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import glob
    from features_weather import TARGET_ICAOS
    lookup = _build_lookup()
    print(f"openap lookup covers {len(lookup)} types")
    print(lookup.head(10).to_string())
    print()
    p = os.path.join(ROOT, "training", "training_2025-06-01_2025-07-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].head(50000)
    out = add_openap(df)
    print(f"\nCoverage on training sample: {out['op_mtow'].notna().mean()*100:.1f}%")
    print("\nMTOW distribution (kg) among covered rows:")
    print(out["op_mtow"].describe(percentiles=[.1, .5, .9]).round(0).to_string())
    print("\nTop covered types:")
    print(out.dropna(subset=["op_mtow"])["AIRCRAFT_TYPE_mvt"].value_counts().head(10).to_string())
    print("\nTop uncovered types:")
    print(out[out["op_mtow"].isna()]["AIRCRAFT_TYPE_mvt"].value_counts().head(10).to_string())
