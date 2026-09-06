"""Join Eurocontrol daily-per-airport features to DEP rows.

Daily table: external/eurocontrol/daily_features.parquet (built by
build_eurocontrol_daily.py). Joined on (mvt_ts.date, ADEP_mvt).
Also adds day-before values as lag features (yesterday's delay predicts today's
carry-over congestion).
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet")


def load_daily() -> pd.DataFrame:
    d = pd.read_parquet(DAILY)
    d["FLT_DATE"] = pd.to_datetime(d["FLT_DATE"])
    return d


def add_eurocontrol(dep: pd.DataFrame) -> pd.DataFrame:
    """dep needs: mvt_ts, ADEP_mvt. Adds daily + 1-day-lag features."""
    d = load_daily()
    metric_cols = [c for c in d.columns if c not in ("FLT_DATE", "APT_ICAO")]

    same = d.rename(columns={"APT_ICAO": "ADEP_mvt"})
    same_r = same.rename(columns={c: f"ec_{c}" for c in metric_cols})

    prev = d.copy()
    prev["FLT_DATE"] = prev["FLT_DATE"] + pd.Timedelta(days=1)
    prev = prev.rename(columns={"APT_ICAO": "ADEP_mvt",
                                **{c: f"ec_lag1_{c}" for c in metric_cols}})

    out = dep.copy()
    out["_date"] = pd.to_datetime(out["mvt_ts"].dt.date)

    out = out.merge(same_r, left_on=["_date", "ADEP_mvt"],
                    right_on=["FLT_DATE", "ADEP_mvt"], how="left") \
             .drop(columns=["FLT_DATE"])
    out = out.merge(prev, left_on=["_date", "ADEP_mvt"],
                    right_on=["FLT_DATE", "ADEP_mvt"], how="left") \
             .drop(columns=["FLT_DATE", "_date"])
    return out


EC_NUM_COLS = None  # populated on first import for reuse


def eurocontrol_numeric_cols() -> list:
    global EC_NUM_COLS
    if EC_NUM_COLS is None:
        d = load_daily()
        metrics = [c for c in d.columns if c not in ("FLT_DATE", "APT_ICAO")]
        EC_NUM_COLS = [f"ec_{c}" for c in metrics] + [f"ec_lag1_{c}" for c in metrics]
    return EC_NUM_COLS


if __name__ == "__main__":
    import glob
    from features_weather import TARGET_ICAOS
    p = os.path.join(ROOT, "training", "training_2025-07-01_2025-08-01.parquet")
    df = pd.read_parquet(p)
    df = df[(df["PHASE_mvt"] == "DEP") & df["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], errors="coerce")
    df = df.head(20000)
    out = add_eurocontrol(df)
    ec_cols = eurocontrol_numeric_cols()
    print(f"Added {len(ec_cols)} EC cols. Sample (LIRF, 2025-07-13):")
    sample = out[(out["ADEP_mvt"] == "LIRF") &
                 (out["mvt_ts"].dt.date == pd.Timestamp("2025-07-13").date())]
    show = ["mvt_ts", "TAXITIME_SEC_mvt",
            "ec_atfm_slot_adherence__FLT_DEP_OUT_LATE_1",
            "ec_all_pre_departure_delay__DLY_ALL_PRE_2",
            "ec_lag1_all_pre_departure_delay__DLY_ALL_PRE_2"]
    print(sample[show].head(10).to_string())
    print(f"\nCoverage of EC cols on this sample:")
    print((~out[ec_cols].isna()).mean().sort_values().head(10).round(3).to_string())
