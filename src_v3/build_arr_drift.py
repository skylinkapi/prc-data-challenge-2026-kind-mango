"""MF4: multi-day arrival taxi-in drift features.

For every DEP row at airport A with mvt_ts date d, compute:
  arr_txi_1d_med: median of validated arrival TAXITIME_SEC_mvt at A on the
    single previous day (d - 1). NaN when no arrival data.
  arr_txi_7d_med: median at A over the 7 days [d - 8, d - 1].
  arr_txi_1d_drift: 1d - 2025 fit-month median at A.
  arr_txi_7d_drift: 7d - 2025 fit-month median at A.

Uses arrival rows from both training (2025) and ranking (2026) parquets
so that 2026 DEP rows see the current 2026 arrival trend.

Coverage: rows with insufficient arrivals in the window get NaN. Base
LightGBM handles missing.

Writes:
  models/arr_drift_train.parquet
  models/arr_drift_rank.parquet
  models/arr_drift_medians.parquet   # 2025 fit-month reference
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src_v3 import config as C

ROOT = C.ROOT
TRAIN_DIR = C.TRAIN_DIR
RANK = C.RANK
FRAME_TRAIN = ROOT / "models" / "v2" / "cache" / "frame_train.parquet"
FRAME_RANK = ROOT / "models" / "v2" / "cache" / "frame_rank.parquet"
OUT_TRAIN = ROOT / "models" / "arr_drift_train.parquet"
OUT_RANK = ROOT / "models" / "arr_drift_rank.parquet"
MED_OUT = ROOT / "models" / "arr_drift_medians.parquet"

FIT_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
MIN_ARR_COUNT = 5  # a day-airport slice needs at least this many arrivals

log = logging.getLogger(__name__)


def load_arrivals(paths: list[str]) -> pd.DataFrame:
    parts = []
    for f in paths:
        t = pd.read_parquet(f, columns=["ADES_mvt", "PHASE_mvt",
                                        "MVT_TIME_UTC_mvt",
                                        "TAXITIME_SEC_mvt"])
        t = t[(t["PHASE_mvt"] == "ARR") & t["ADES_mvt"].isin(C.TARGETS)]
        t = t[t["TAXITIME_SEC_mvt"].astype(float).between(30, 7200)]
        parts.append(t[["ADES_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]])
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    df["date"] = df["ts"].dt.floor("D")
    df["month"] = df["ts"].dt.month
    return df


def daily_medians(arr: pd.DataFrame) -> pd.DataFrame:
    """Per (airport, date) median arrival taxi-in with row count."""
    g = arr.groupby(["ADES_mvt", "date"]).agg(
        median=("TAXITIME_SEC_mvt", "median"),
        count=("TAXITIME_SEC_mvt", "size"),
    ).reset_index()
    g = g[g["count"] >= MIN_ARR_COUNT]
    return g


def rolling_medians(daily: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Per airport, rolling median over the previous `window_days` days.

    Uses `.rolling` on a sorted date index. Because the daily-medians frame
    has one entry per (airport, date), we compute a weighted rolling median
    approximation as the mean of the daily medians (fine given the
    per-day counts are similar in scale).
    """
    frames = []
    for a, g in daily.groupby("ADES_mvt"):
        g = g.sort_values("date").reset_index(drop=True)
        g["value"] = g["median"].rolling(window_days, min_periods=1).mean()
        frames.append(g[["ADES_mvt", "date", "value"]])
    return pd.concat(frames, ignore_index=True)


def per_airport_fit_median(arr: pd.DataFrame) -> pd.Series:
    """Per-airport median arrival taxi over the 2025 fit months."""
    m = arr[arr["month"].isin(FIT_MONTHS)]
    med = m.groupby("ADES_mvt")["TAXITIME_SEC_mvt"].median()
    return med


def apply_drift(frame: pd.DataFrame, roll_1d: pd.DataFrame,
                roll_7d: pd.DataFrame, per_apt: pd.Series) -> pd.DataFrame:
    """For each DEP row, look up the previous day's 1d and 7d values at
    its airport, then subtract the 2025 fit-month per-airport median."""
    f = frame[["MVT_ID_mvt", "ADEP_mvt", "mvt_ts"]].copy()
    f["mvt_ts"] = pd.to_datetime(f["mvt_ts"], utc=True, errors="coerce")
    # Previous day the median refers to = row date - 1 day
    f["prev_date"] = (f["mvt_ts"].dt.floor("D") - pd.Timedelta(days=1))
    m1 = f.merge(roll_1d.rename(columns={"ADES_mvt": "ADEP_mvt",
                                          "date": "prev_date",
                                          "value": "arr_txi_1d_med"}),
                 on=["ADEP_mvt", "prev_date"], how="left")
    m2 = m1.merge(roll_7d.rename(columns={"ADES_mvt": "ADEP_mvt",
                                           "date": "prev_date",
                                           "value": "arr_txi_7d_med"}),
                  on=["ADEP_mvt", "prev_date"], how="left")
    apt_med = m2["ADEP_mvt"].map(per_apt)
    m2["arr_txi_1d_drift"] = m2["arr_txi_1d_med"] - apt_med
    m2["arr_txi_7d_drift"] = m2["arr_txi_7d_med"] - apt_med
    return m2[["MVT_ID_mvt", "arr_txi_1d_med", "arr_txi_7d_med",
               "arr_txi_1d_drift", "arr_txi_7d_drift"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    log.info("loading arrival rows from training + ranking")
    train_paths = sorted(glob.glob(str(TRAIN_DIR / "*.parquet")))
    arr = load_arrivals(train_paths + [str(RANK)])
    log.info("arrivals: %d rows", len(arr))

    per_apt = per_airport_fit_median(
        arr[arr["ts"].dt.year == 2025]
    )
    per_apt.to_frame("median").to_parquet(MED_OUT)
    log.info("2025 fit-month per-airport medians: %s",
             {a: round(v) for a, v in per_apt.items()})

    daily = daily_medians(arr)
    log.info("daily entries: %d", len(daily))

    roll_1d = rolling_medians(daily, window_days=1)   # just the day itself
    roll_7d = rolling_medians(daily, window_days=7)

    train_frame = pd.read_parquet(FRAME_TRAIN, columns=["MVT_ID_mvt",
                                                        "ADEP_mvt", "mvt_ts"])
    rank_frame = pd.read_parquet(FRAME_RANK, columns=["MVT_ID_mvt",
                                                       "ADEP_mvt", "mvt_ts"])

    log.info("apply on training rows")
    res_train = apply_drift(train_frame, roll_1d, roll_7d, per_apt)
    for c in ("arr_txi_1d_med", "arr_txi_7d_med",
              "arr_txi_1d_drift", "arr_txi_7d_drift"):
        cov = res_train[c].notna().mean()
        log.info("  %s coverage %.3f", c, cov)
    res_train.to_parquet(OUT_TRAIN)
    log.info("wrote %s (%d rows)", OUT_TRAIN, len(res_train))

    log.info("apply on ranking rows")
    res_rank = apply_drift(rank_frame, roll_1d, roll_7d, per_apt)
    for c in ("arr_txi_1d_med", "arr_txi_7d_med",
              "arr_txi_1d_drift", "arr_txi_7d_drift"):
        cov = res_rank[c].notna().mean()
        log.info("  %s coverage %.3f", c, cov)
    res_rank.to_parquet(OUT_RANK)
    log.info("wrote %s (%d rows)", OUT_RANK, len(res_rank))


if __name__ == "__main__":
    main()
