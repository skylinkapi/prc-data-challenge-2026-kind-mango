"""L3.a: 25th and 75th percentiles of neighbour mvt_eobt1.

Same three windows the deployed tempo uses (apt30, apt60, rwy30). One column
per (window, quantile), six columns total, keyed on MVT_ID_mvt. Writes
models/tempo_p2575.parquet.

Uses the src_v2 training frame as the source of neighbour mvt_eobt1 values
and per-row keys, so the definition matches the shipped tempo exactly.
"""
import logging
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_train.parquet")
RANK_FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_rank.parquet")
OUT_TRAIN = os.path.join(ROOT, "models", "tempo_p2575_train.parquet")
OUT_RANK = os.path.join(ROOT, "models", "tempo_p2575_rank.parquet")
_NS_MIN = np.int64(60) * 1_000_000_000
GROUPS = [(30, ("ADEP_mvt",), "_apt30"),
          (60, ("ADEP_mvt",), "_apt60"),
          (30, ("ADEP_mvt", "RUNWAY_mvt"), "_rwy30")]
log = logging.getLogger(__name__)


def _quantiles_group(frame: pd.DataFrame, keys: tuple[str, ...],
                     window_min: int, suffix: str) -> pd.DataFrame:
    """p25 and p75 of mvt_eobt1 in the previous window_min minutes,
    grouped by keys. NaN when no neighbour."""
    win_ns = np.int64(window_min) * _NS_MIN
    right = frame.dropna(subset=["mvt_eobt1"]).copy()
    right["mvt_ns"] = right["mvt_ts"].astype("int64")
    right = right.sort_values([*keys, "mvt_ns"]).reset_index(drop=True)

    p25 = np.full(len(frame), np.nan)
    p75 = np.full(len(frame), np.nan)

    left = frame[["MVT_ID_mvt", "mvt_ts", *keys]].copy()
    left["mvt_ns"] = left["mvt_ts"].astype("int64")
    left["_orig_idx"] = np.arange(len(left))
    left = left.sort_values([*keys, "mvt_ns"]).reset_index(drop=True)

    grouper = list(keys) if len(keys) > 1 else keys[0]
    for k, right_g in right.groupby(grouper, sort=False, dropna=True):
        r_ts = right_g["mvt_ns"].values
        r_val = right_g["mvt_eobt1"].values
        if len(keys) == 1:
            mask = (left[keys[0]].values == k)
        else:
            mask = np.ones(len(left), dtype=bool)
            for kn, kv in zip(keys, k):
                mask &= (left[kn].values == kv)
        left_ts = left.loc[mask, "mvt_ns"].values
        orig_idx = left.loc[mask, "_orig_idx"].values
        if len(left_ts) == 0:
            continue
        hi = np.searchsorted(r_ts, left_ts, side="left")
        lo = np.searchsorted(r_ts, left_ts - win_ns, side="left")
        n = hi - lo
        for i in np.where(n > 0)[0]:
            window = r_val[lo[i]:hi[i]]
            p25[orig_idx[i]], p75[orig_idx[i]] = np.percentile(window, [25, 75])
    return pd.DataFrame({
        "MVT_ID_mvt": frame["MVT_ID_mvt"].values,
        f"nb_eobt_p25{suffix}": p25,
        f"nb_eobt_p75{suffix}": p75,
    })


def build(path: str, out: str) -> None:
    if os.path.exists(out):
        log.info("cache hit: %s", out)
        return
    log.info("read %s", path)
    frame = pd.read_parquet(path, columns=["MVT_ID_mvt", "mvt_ts",
                                           "ADEP_mvt", "RUNWAY_mvt", "mvt_eobt1"])
    out_df = pd.DataFrame({"MVT_ID_mvt": frame["MVT_ID_mvt"].values})
    for win, keys, suf in GROUPS:
        t0 = time.time()
        g = _quantiles_group(frame, keys, win, suf)
        out_df = out_df.merge(g, on="MVT_ID_mvt", how="left")
        cov = out_df[f"nb_eobt_p25{suf}"].notna().mean()
        log.info("group %s: coverage %.3f in %.0fs", suf, cov, time.time() - t0)
    out_df.to_parquet(out)
    log.info("wrote %s (%d rows, %d cols)", out, len(out_df), len(out_df.columns))


def main() -> None:
    build(FRAME, OUT_TRAIN)
    build(RANK_FRAME, OUT_RANK)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
