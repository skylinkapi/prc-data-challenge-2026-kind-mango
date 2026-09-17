"""L3.b: 25th and 75th percentiles of neighbour mvt_iobt.

Same three windows the deployed tempo uses (apt30, apt60, rwy30) but keyed on
`mvt_iobt` instead of `mvt_eobt1`. IOBT is the initial-plan off-block and
carries a different tail from EOBT_1, which reads live updates. Writes
models/tempo_iobt_p2575_{train,rank}.parquet.
"""
import logging
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_train.parquet")
RANK_FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_rank.parquet")
OUT_TRAIN = os.path.join(ROOT, "models", "tempo_iobt_p2575_train.parquet")
OUT_RANK = os.path.join(ROOT, "models", "tempo_iobt_p2575_rank.parquet")
_NS_MIN = np.int64(60) * 1_000_000_000
GROUPS = [(30, ("ADEP_mvt",), "_apt30"),
          (60, ("ADEP_mvt",), "_apt60"),
          (30, ("ADEP_mvt", "RUNWAY_mvt"), "_rwy30")]
ANCHOR = "mvt_iobt"
PREFIX = "nb_iobt"
log = logging.getLogger(__name__)


def _quantiles_group(frame: pd.DataFrame, keys: tuple[str, ...],
                     window_min: int, suffix: str) -> pd.DataFrame:
    win_ns = np.int64(window_min) * _NS_MIN
    right = frame.dropna(subset=[ANCHOR]).copy()
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
        r_val = right_g[ANCHOR].values
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
        f"{PREFIX}_p25{suffix}": p25,
        f"{PREFIX}_p75{suffix}": p75,
    })


def build(path: str, out: str) -> None:
    if os.path.exists(out):
        log.info("cache hit: %s", out)
        return
    log.info("read %s", path)
    frame = pd.read_parquet(path, columns=["MVT_ID_mvt", "mvt_ts",
                                           "ADEP_mvt", "RUNWAY_mvt", ANCHOR])
    out_df = pd.DataFrame({"MVT_ID_mvt": frame["MVT_ID_mvt"].values})
    for win, keys, suf in GROUPS:
        t0 = time.time()
        g = _quantiles_group(frame, keys, win, suf)
        out_df = out_df.merge(g, on="MVT_ID_mvt", how="left")
        cov = out_df[f"{PREFIX}_p25{suf}"].notna().mean()
        log.info("group %s: coverage %.3f in %.0fs", suf, cov, time.time() - t0)
    out_df.to_parquet(out)
    log.info("wrote %s (%d rows, %d cols)", out, len(out_df), len(out_df.columns))


def main() -> None:
    build(FRAME, OUT_TRAIN)
    build(RANK_FRAME, OUT_RANK)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
