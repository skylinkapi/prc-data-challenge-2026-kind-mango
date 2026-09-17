"""MP6: arrival mirror.

Trains a LightGBM regressor on 2025 arrival taxi-in and scores it on the
2026 arrival taxi-in labels the organiser did include on ARR rows of the
ranking file (fourteenth-pass fact 7). Reports FULL RMSE, per-airport and
per-month, so any departure-side lever can be re-measured with real 2026
labels before it goes to the leaderboard.

The feature set is deliberately small in this first version: airport,
runway, stand and stand-prefix, aircraft type, operator, hour, dow and
month. It is not a full mirror of the departure recipe (that would need
per-arrival weather, congestion and anchor deltas). The pass calls for a
mirror; this file is the minimum viable one and the runner other measures
build on.

Writes models/v3/arrival_mirror.json with per-airport and per-month RMSE.
Deterministic and seeded (MC4).
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src_v3 import config as C

RAW_COLS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
            "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
            "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "FLIGHT_mvt"]

FEATURES = ("ADES_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "stand_prefix",
            "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "flt_prefix",
            "hour", "dow", "month", "sched_delay_arr")
CAT_FOR_ARR = ("ADES_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt",
               "stand_prefix", "AIRCRAFT_TYPE_mvt",
               "AIRCRAFT_OPERATOR_flt", "flt_prefix",
               "hour", "dow", "month")
log = logging.getLogger(__name__)


def _stand_prefix(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.extract(r"^([A-Za-z]+)")[0]
    return x.fillna("UNK")


def _flt_prefix(s: pd.Series) -> pd.Series:
    return s.astype("string").str[:3].fillna("UNK")


def load_arrivals(paths: list[str], targets: tuple[str, ...] = C.TARGETS
                  ) -> pd.DataFrame:
    parts = []
    for p in paths:
        t = pd.read_parquet(p, columns=RAW_COLS)
        t = t[(t["PHASE_mvt"] == "ARR") & t["ADES_mvt"].isin(targets)]
        t = t[t["TAXITIME_SEC_mvt"].astype(float).between(30, 7200)]
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    df["mvt_ts"] = pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    df["sched_ts"] = pd.to_datetime(df["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
    df["hour"] = df["mvt_ts"].dt.hour.astype("Int16")
    df["dow"] = df["mvt_ts"].dt.dayofweek.astype("Int16")
    df["month"] = df["mvt_ts"].dt.month.astype("Int16")
    df["sched_delay_arr"] = (df["mvt_ts"] - df["sched_ts"]).dt.total_seconds()
    df["stand_prefix"] = _stand_prefix(df["STAND_mvt"])
    df["flt_prefix"] = _flt_prefix(df["FLIGHT_mvt"])
    return df


def _to_cat(df: pd.DataFrame, cats: dict[str, pd.Index]) -> pd.DataFrame:
    df = df.copy()
    for c in CAT_FOR_ARR:
        v = df[c].astype("string")
        df[c] = pd.Categorical(v, categories=cats[c])
    return df


def train_and_score(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    cats = {c: pd.Index(train[c].astype("string").dropna().unique())
            for c in CAT_FOR_ARR}
    train = _to_cat(train, cats)
    test = _to_cat(test, cats)
    y_train = train["TAXITIME_SEC_mvt"].astype(float).values
    y_test = test["TAXITIME_SEC_mvt"].astype(float).values
    dt = lgb.Dataset(train[list(FEATURES)], label=y_train,
                     categorical_feature=list(CAT_FOR_ARR),
                     params={"linear_tree": True, "feature_pre_filter": False})
    params = {**C.LGB_LINEAR_BASE,
              "num_leaves": 127, "linear_lambda": 1.0}
    log.info("train %d, test %d, features %d", len(train), len(test), len(FEATURES))
    t0 = time.time()
    b = lgb.train(params, dt, num_boost_round=1000)
    pred = np.clip(b.predict(test[list(FEATURES)]), 0, None)
    e2 = (pred - y_test) ** 2
    full = float(np.sqrt(e2.mean()))
    apt = test["ADES_mvt"].astype(str).values
    mon = test["month"].astype(int).values
    per_apt = {a: float(np.sqrt(e2[apt == a].mean())) for a in sorted(set(apt))}
    per_month = {int(m): float(np.sqrt(e2[mon == m].mean())) for m in sorted(set(mon))}
    per_cell = {a: {int(m): float(np.sqrt(e2[(apt == a) & (mon == m)].mean()))
                    for m in sorted(set(mon[apt == a]))}
                for a in sorted(set(apt))}
    log.info("full RMSE %.2f in %.0fs", full, time.time() - t0)
    return {"full_rmse": full, "per_airport_rmse": per_apt,
            "per_month_rmse": per_month, "per_airport_month_rmse": per_cell,
            "n_train": int(len(train)), "n_test": int(len(test))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(C.MODELS / "arrival_mirror.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    train_paths = sorted(glob.glob(str(C.TRAIN_DIR / "*.parquet")))
    rank_path = str(C.RANK)
    log.info("loading 2025 arrivals")
    train_arr = load_arrivals(train_paths)
    log.info("loading 2026 arrivals")
    test_arr = load_arrivals([rank_path])
    if train_arr.empty or test_arr.empty:
        raise RuntimeError("arrival load returned empty")
    log.info("2025 arrivals %d, 2026 arrivals %d", len(train_arr), len(test_arr))

    report = train_and_score(train_arr, test_arr)
    Path(args.out).write_text(json.dumps(report, indent=1))
    log.info("wrote %s", args.out)
    log.info("full RMSE %.2f", report["full_rmse"])
    for a, r in report["per_airport_rmse"].items():
        log.info("  %s: %.2f", a, r)


if __name__ == "__main__":
    main()
