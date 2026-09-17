"""L3.c: neighbour mvt_eobt1 tempo grouped by (airport, stand_prefix), 30 min.

Terminal-level tempo. stand_prefix is the alphabetic prefix of STAND_mvt, so
LFPG stands G1..G14 share the prefix "G". Emits median, mean and count over
the previous 30 minutes of departures at the same terminal. Writes
models/tempo_term30_{train,rank}.parquet.
"""
import logging
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_train.parquet")
RANK_FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_rank.parquet")
OUT_TRAIN = os.path.join(ROOT, "models", "tempo_term30_train.parquet")
OUT_RANK = os.path.join(ROOT, "models", "tempo_term30_rank.parquet")
_NS_MIN = np.int64(60) * 1_000_000_000
WINDOW_MIN = 30
KEYS = ("ADEP_mvt", "stand_prefix")
log = logging.getLogger(__name__)


def _term_tempo(frame: pd.DataFrame) -> pd.DataFrame:
    win_ns = np.int64(WINDOW_MIN) * _NS_MIN
    right = frame.dropna(subset=["mvt_eobt1", *KEYS]).copy()
    right["mvt_ns"] = right["mvt_ts"].astype("int64")
    right = right.sort_values([*KEYS, "mvt_ns"]).reset_index(drop=True)

    med = np.full(len(frame), np.nan)
    mean = np.full(len(frame), np.nan)
    cnt = np.full(len(frame), np.nan)

    left = frame[["MVT_ID_mvt", "mvt_ts", *KEYS]].copy()
    left["mvt_ns"] = left["mvt_ts"].astype("int64")
    left["_orig_idx"] = np.arange(len(left))
    left = left.sort_values([*KEYS, "mvt_ns"]).reset_index(drop=True)

    for k, right_g in right.groupby(list(KEYS), sort=False, dropna=True):
        r_ts = right_g["mvt_ns"].values
        r_val = right_g["mvt_eobt1"].values
        mask = np.ones(len(left), dtype=bool)
        for kn, kv in zip(KEYS, k):
            mask &= (left[kn].values == kv)
        left_ts = left.loc[mask, "mvt_ns"].values
        orig_idx = left.loc[mask, "_orig_idx"].values
        if len(left_ts) == 0:
            continue
        hi = np.searchsorted(r_ts, left_ts, side="left")
        lo = np.searchsorted(r_ts, left_ts - win_ns, side="left")
        n = hi - lo
        cumsum = np.concatenate(([0.0], np.cumsum(r_val)))
        s = cumsum[hi] - cumsum[lo]
        mean_g = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        med_g = np.full(len(left_ts), np.nan)
        for i in np.where(n > 0)[0]:
            med_g[i] = np.median(r_val[lo[i]:hi[i]])
        mean[orig_idx] = mean_g
        med[orig_idx] = med_g
        cnt[orig_idx] = n.astype(float)
    return pd.DataFrame({
        "MVT_ID_mvt": frame["MVT_ID_mvt"].values,
        "nb_eobt_med_term30": med,
        "nb_eobt_mean_term30": mean,
        "nb_eobt_cnt_term30": cnt,
    })


def build(path: str, out: str) -> None:
    if os.path.exists(out):
        log.info("cache hit: %s", out)
        return
    log.info("read %s", path)
    frame = pd.read_parquet(path, columns=["MVT_ID_mvt", "mvt_ts",
                                           "ADEP_mvt", "stand_prefix",
                                           "mvt_eobt1"])
    t0 = time.time()
    out_df = _term_tempo(frame)
    cov = out_df["nb_eobt_med_term30"].notna().mean()
    log.info("term30 coverage %.3f in %.0fs", cov, time.time() - t0)
    out_df.to_parquet(out)
    log.info("wrote %s (%d rows, %d cols)", out, len(out_df), len(out_df.columns))


def main() -> None:
    build(FRAME, OUT_TRAIN)
    build(RANK_FRAME, OUT_RANK)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
