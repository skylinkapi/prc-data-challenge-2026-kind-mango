"""Weather features: as-of joins DEP movements to METAR, computes crosswind on
the active runway, low-visibility flag, precipitation/snow/thunder flags.

METAR source: Iowa State ASOS archive (external/metar/{ICAO}.csv).
Runway headings: OurAirports runways.csv (external/airports/runways.csv).
"""
import glob
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METAR_DIR = os.path.join(ROOT, "external", "metar")
RWY_CSV = os.path.join(ROOT, "external", "airports", "runways.csv")

TARGET_ICAOS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
                "LFPG", "LIRF", "LTFM", "LSZH"]


def load_metar_all() -> pd.DataFrame:
    frames = []
    for icao in TARGET_ICAOS:
        p = os.path.join(METAR_DIR, f"{icao}.csv")
        m = pd.read_csv(p, low_memory=False, na_values=["M"])
        m["valid"] = pd.to_datetime(m["valid"], utc=True, errors="coerce").astype("datetime64[us, UTC]")
        # numeric
        for c in ["drct", "sknt", "gust", "vsby", "skyl1", "p01i"]:
            m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.dropna(subset=["valid"]).sort_values("valid")
        m["ADEP_mvt"] = icao
        frames.append(m[["ADEP_mvt", "valid", "drct", "sknt", "gust",
                         "vsby", "skyc1", "skyl1", "wxcodes", "p01i"]])
    return pd.concat(frames, ignore_index=True)


def load_runway_headings() -> dict:
    """Return {(ICAO, runway_end_id): heading_deg}. Runway ids stripped."""
    r = pd.read_csv(RWY_CSV, low_memory=False)
    r = r[r["airport_ident"].isin(TARGET_ICAOS) & (r["closed"] == 0)]
    out = {}
    for _, row in r.iterrows():
        a = row["airport_ident"]
        for end, hdg in [("le_ident", "le_heading_degT"),
                         ("he_ident", "he_heading_degT")]:
            ident = str(row[end]).strip().upper()
            deg = row[hdg]
            if pd.notna(deg) and ident and ident != "NAN":
                out[(a, ident)] = float(deg)
    return out


def norm_runway(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip().str.replace(r"^0+", "", regex=True)


def lookup_heading(apt: pd.Series, rwy: pd.Series, table: dict) -> np.ndarray:
    """Match on both zero-padded and un-padded runway id (e.g. '6R' vs '06R')."""
    def _try(a: str, r: str):
        if (a, r) in table:
            return table[(a, r)]
        # try zero-padded ("6R" -> "06R")
        num, suf = "", r
        for i, ch in enumerate(r):
            if not ch.isdigit():
                num, suf = r[:i], r[i:]
                break
        else:
            num, suf = r, ""
        if num and len(num) == 1:
            padded = "0" + num + suf
            return table.get((a, padded), np.nan)
        return np.nan
    return np.array([_try(a, r) for a, r in zip(apt.values, rwy.values)])


def parse_wx_flags(wx: pd.Series) -> pd.DataFrame:
    s = wx.fillna("").astype(str).str.upper()
    return pd.DataFrame({
        "wx_precip": s.str.contains(r"RA|DZ|SN|SG|GR|GS|PL|IC", regex=True).astype(np.int8),
        "wx_snow":   s.str.contains(r"SN|FZ|SG",              regex=True).astype(np.int8),
        "wx_thunder":s.str.contains(r"TS",                    regex=False).astype(np.int8),
        "wx_freezing":s.str.contains(r"FZ",                   regex=False).astype(np.int8),
    })


def add_weather(dep: pd.DataFrame) -> pd.DataFrame:
    """dep must contain: ADEP_mvt, mvt_ts (UTC), RUNWAY_mvt.
    Returns dep with added weather cols.
    """
    metar = load_metar_all()
    headings = load_runway_headings()

    dep = dep.copy()
    dep["_rwy_norm"] = norm_runway(dep["RUNWAY_mvt"].fillna(""))
    dep["rwy_hdg"] = lookup_heading(dep["ADEP_mvt"], dep["_rwy_norm"], headings)

    # as-of merge per airport (nearest METAR ≤ 45 min before mvt_ts)
    out = []
    for icao in TARGET_ICAOS:
        d = dep[dep["ADEP_mvt"] == icao].sort_values("mvt_ts")
        if d.empty:
            continue
        m = metar[metar["ADEP_mvt"] == icao].drop(columns=["ADEP_mvt"])
        joined = pd.merge_asof(
            d, m, left_on="mvt_ts", right_on="valid",
            direction="backward", tolerance=pd.Timedelta("45min"),
        )
        out.append(joined)
    df = pd.concat(out).sort_index()

    # crosswind / headwind on active runway (kt)
    delta = np.deg2rad(df["drct"] - df["rwy_hdg"])
    df["wind_cross_kt"] = df["sknt"].fillna(0) * np.abs(np.sin(delta))
    df["wind_head_kt"]  = df["sknt"].fillna(0) * np.cos(delta)  # neg = tailwind
    df["gust_kt"] = df["gust"].fillna(df["sknt"]).fillna(0)

    # visibility: statute miles → km
    df["vis_km"] = df["vsby"] * 1.60934
    df["low_vis"] = (df["vis_km"] < 5.0).astype(np.int8)   # LVP likely
    df["very_low_vis"] = (df["vis_km"] < 1.5).astype(np.int8)

    # ceiling
    df["ceiling_ft"] = np.where(df["skyc1"].isin(["BKN", "OVC", "VV"]),
                                df["skyl1"], 25000)
    df["low_ceiling"] = (df["ceiling_ft"] < 500).astype(np.int8)

    df = pd.concat([df, parse_wx_flags(df["wxcodes"])], axis=1)
    return df.drop(columns=["_rwy_norm", "valid"])


if __name__ == "__main__":
    # quick smoke test
    p = os.path.join(ROOT, "training", "training_2025-01-01_2025-02-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    w = add_weather(df)
    print(f"Rows: {len(w):,}")
    cols = ["ADEP_mvt", "RUNWAY_mvt", "rwy_hdg", "drct", "sknt",
            "wind_cross_kt", "wind_head_kt", "vis_km", "low_vis",
            "ceiling_ft", "wx_precip", "wx_snow", "wx_thunder"]
    print(w[cols].head(10).to_string())
    print("\nCoverage:")
    print((~w[["sknt", "vis_km", "rwy_hdg"]].isna()).mean().round(3).to_string())
