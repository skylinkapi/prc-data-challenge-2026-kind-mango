"""Forward take-off order: departures in the next 30 min that filed an earlier EOBT_1.

The backward count of B2 sits in the src_v2 frame as order_later_eobt_30. This
module adds its mirror, order_earlier_eobt_next30, for the training and the
ranking frame, keyed by movement id. Reads other movements' MVT_TIME and
EOBT_1_flt only, which the organiser permits (RECAP, Discord 2026-09-09).
"""
import logging
import os

import numpy as np
import pandas as pd

from train_r_all_v40 import V2_FRAME

WINDOW_NS = np.int64(30 * 60) * 1_000_000_000
OUT = os.path.join(os.path.dirname(V2_FRAME), "order_next.parquet")
log = logging.getLogger(__name__)


def order_next(frame: pd.DataFrame) -> pd.Series:
    """Per row, count of same-airport departures with MVT in (t, t + 30 min] and an earlier EOBT_1."""
    out = np.full(len(frame), np.nan)
    mvt = frame["mvt_ts"].astype("datetime64[ns, UTC]").astype("int64").values
    eobt = pd.to_datetime(frame["EOBT_1_flt"], utc=True).astype("datetime64[ns, UTC]").astype("int64").values
    eobt = eobt.astype(float)
    eobt[frame["EOBT_1_flt"].isna().values] = np.nan
    for a in frame["ADEP_mvt"].dropna().unique():
        idx = np.where(frame["ADEP_mvt"].values == a)[0]
        order = idx[np.argsort(mvt[idx], kind="stable")]
        t, e = mvt[order], eobt[order]
        hi = np.searchsorted(t, t + WINDOW_NS, side="right")
        res = np.full(len(order), np.nan)
        for i in range(len(order)):
            if np.isnan(e[i]):
                continue
            seg = e[i + 1:hi[i]]
            res[i] = np.sum(seg < e[i])
        out[order] = res
    return pd.Series(out, index=frame.index, name="order_earlier_eobt_next30")


def main() -> None:
    parts = []
    for src in ("train", "rank"):
        f = pd.read_parquet(os.path.join(os.path.dirname(V2_FRAME), f"frame_{src}.parquet"),
                            columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_ts", "EOBT_1_flt"])
        s = order_next(f)
        parts.append(pd.DataFrame({"MVT_ID_mvt": f["MVT_ID_mvt"].values, s.name: s.values}))
        log.info("%s: %d rows, coverage %.3f, mean %.2f", src, len(f), s.notna().mean(), np.nanmean(s.values))
    pd.concat(parts, ignore_index=True).to_parquet(OUT)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
