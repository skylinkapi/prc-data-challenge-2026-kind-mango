"""L3.e: arrival taxi-in residual — drift-corrected live tempo.

For each DEP row at airport X, the median observed arrival taxi-in time at X
in the previous 30 minutes (from ARR rows with ADES = X), minus the airport's
2025 fit-month median arrival taxi-in. Positive = the taxiway is running
slower than usual right now. Emits arr_txi_res_30 and arr_txi_cnt_30 keyed on
MVT_ID_mvt. Writes models/arr_txi_res_{train,rank}.parquet.
"""
import glob
import logging
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_train.parquet")
RANK_FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_rank.parquet")
OUT_TRAIN = os.path.join(ROOT, "models", "arr_txi_res_train.parquet")
OUT_RANK = os.path.join(ROOT, "models", "arr_txi_res_rank.parquet")
MED_OUT = os.path.join(ROOT, "models", "arr_txi_airport_medians.parquet")
_NS_MIN = np.int64(60) * 1_000_000_000
WINDOW_MIN = 30
FIT_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
TARGETS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
log = logging.getLogger(__name__)


def load_arr_rows(paths: list[str]) -> pd.DataFrame:
    """ARR rows at target airports with valid taxi-in time."""
    parts = []
    cols = ["ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    for f in paths:
        p = pd.read_parquet(f, columns=cols)
        p = p[(p["PHASE_mvt"] == "ARR") & p["ADES_mvt"].isin(TARGETS)]
        p = p[p["TAXITIME_SEC_mvt"].astype(float) > 0]
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def airport_medians(arr: pd.DataFrame) -> pd.Series:
    """Per-airport 2025 fit-month median taxi-in."""
    arr = arr.copy()
    arr["ts"] = pd.to_datetime(arr["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
    arr["month"] = arr["ts"].dt.month
    fit = arr[arr["month"].isin(FIT_MONTHS)]
    med = fit.groupby("ADES_mvt")["TAXITIME_SEC_mvt"].median().dropna()
    log.info("airport medians: %s", {k: round(v) for k, v in med.items()})
    return med


def _window_median(dep: pd.DataFrame, arr_ns: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """For each DEP row, median arr taxi-in in prev 30 min at same airport."""
    win_ns = np.int64(WINDOW_MIN) * _NS_MIN
    med = np.full(len(dep), np.nan)
    cnt = np.full(len(dep), np.nan)
    left = dep[["MVT_ID_mvt", "ADEP_mvt", "mvt_ts"]].copy()
    left["mvt_ts"] = pd.to_datetime(left["mvt_ts"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    left["mvt_ns"] = left["mvt_ts"].astype("int64")
    left["_orig_idx"] = np.arange(len(left))
    for a, g in left.groupby("ADEP_mvt", sort=False, dropna=True):
        if a not in arr_ns:
            continue
        r_ts, r_val = arr_ns[a]
        left_ts = g["mvt_ns"].values
        orig_idx = g["_orig_idx"].values
        hi = np.searchsorted(r_ts, left_ts, side="left")
        lo = np.searchsorted(r_ts, left_ts - win_ns, side="left")
        n = hi - lo
        for i in np.where(n > 0)[0]:
            med[orig_idx[i]] = np.median(r_val[lo[i]:hi[i]])
        cnt[orig_idx] = n.astype(float)
    return pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].values,
                         "arr_txi_med_30": med, "arr_txi_cnt_30": cnt})


def build_arr_index(arr: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arr = arr.copy()
    arr["mvt_ts"] = pd.to_datetime(arr["MVT_TIME_UTC_mvt"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    arr = arr.dropna(subset=["mvt_ts"]).sort_values(["ADES_mvt", "mvt_ts"])
    idx = {}
    for a, g in arr.groupby("ADES_mvt", sort=False):
        idx[a] = (g["mvt_ts"].astype("int64").values, g["TAXITIME_SEC_mvt"].astype(float).values)
    return idx


def apply_residual(win: pd.DataFrame, medians: pd.Series, dep: pd.DataFrame) -> pd.DataFrame:
    apt = dep["ADEP_mvt"].astype(str)
    med = apt.map(medians)
    win = win.merge(dep[["MVT_ID_mvt"]].assign(_apt_med=med.values),
                    on="MVT_ID_mvt", how="left")
    win["arr_txi_res_30"] = win["arr_txi_med_30"] - win["_apt_med"]
    return win[["MVT_ID_mvt", "arr_txi_res_30", "arr_txi_cnt_30"]]


def main() -> None:
    log.info("loading ARR rows from training + ranking")
    t0 = time.time()
    train_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))
    arr_train = load_arr_rows(train_paths)
    arr_rank = load_arr_rows([RANK])
    log.info("arr train %d, rank %d (%.0fs)", len(arr_train), len(arr_rank), time.time() - t0)

    medians = airport_medians(arr_train)
    medians.to_frame("arr_txi_median").to_parquet(MED_OUT)

    # Training set: window over train ARR only
    train_dep = pd.read_parquet(FRAME, columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_ts"])
    idx_train = build_arr_index(arr_train)
    log.info("windowed medians on %d train rows", len(train_dep))
    t0 = time.time()
    win_train = _window_median(train_dep, idx_train)
    res_train = apply_residual(win_train, medians, train_dep)
    log.info("train residual computed in %.0fs, coverage %.3f", time.time() - t0,
             res_train["arr_txi_res_30"].notna().mean())
    res_train.to_parquet(OUT_TRAIN)

    # Ranking set: window over rank ARR (they exist in 2026)
    rank_dep = pd.read_parquet(RANK_FRAME, columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_ts"])
    idx_rank = build_arr_index(arr_rank)
    log.info("windowed medians on %d rank rows", len(rank_dep))
    t0 = time.time()
    win_rank = _window_median(rank_dep, idx_rank)
    res_rank = apply_residual(win_rank, medians, rank_dep)
    log.info("rank residual computed in %.0fs, coverage %.3f", time.time() - t0,
             res_rank["arr_txi_res_30"].notna().mean())
    res_rank.to_parquet(OUT_RANK)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
