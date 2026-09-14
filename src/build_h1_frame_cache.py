"""H1: rebuild the cached frame with the movement id inside it.

The v36 cache (97 served columns plus the label) keeps its movement id in a
side file written by a composite-key join; 169 rows carry a null id and get
null tempo values in every merge since. This script recovers the 169 ids from
the src_v2 frame_train leftovers: first by a content key, then by per-airport
mvt_ts rank, verifying each assignment against the time order of the cache.
The assembled frame carries MVT_ID_mvt, the 110 served columns of the v41
stack, the label, sd, the month and the airport.

Writes models/h1_frame_cache.parquet and models/h1_frame_cache.feat.txt.
"""
import logging
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
OLD_CACHE = os.path.join(MODELS, "v36_tune_cache.parquet")
OLD_IDS = os.path.join(MODELS, "v36_tune_cache.ids.parquet")
V2_FRAME = os.path.join(MODELS, "v2", "cache", "frame_train.parquet")
OUT = os.path.join(MODELS, "h1_frame_cache.parquet")
TEMPO_COLS = ["nb_eobt_med_apt30", "nb_eobt_mean_apt30", "nb_eobt_cnt_apt30",
              "nb_eobt_med_apt60", "nb_eobt_mean_apt60", "nb_eobt_cnt_apt60",
              "nb_eobt_med_rwy30", "nb_eobt_mean_rwy30", "nb_eobt_cnt_rwy30",
              "order_later_eobt_30", "order_total_30", "stand_gap", "queue_eobt_mvt"]
KEY_NUM = ["month", "TAXITIME_SEC_mvt", "mvt_eobt1"]
log = logging.getLogger(__name__)


def content_key(df: pd.DataFrame, sd_col: str) -> pd.Series:
    """Join key from columns the two pipelines compute identically."""
    k = (df["ADEP_mvt"].astype(str) + "|" + df["RUNWAY_mvt"].astype(str) + "|"
         + df["STAND_mvt"].astype(str) + "|" + df[sd_col].round(0).astype("Int64").astype(str))
    for col in KEY_NUM:
        k = k + "|" + df[col].round(0).astype("Int64").astype(str)
    return k


def verify_content(ids: pd.Series, cache: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Fail unless every non-null id reproduces label, month and airport."""
    ok = ids.notna().values
    m = pd.DataFrame({"id": ids[ok].values,
                      "y": cache["TAXITIME_SEC_mvt"].values[ok],
                      "month": cache["month"].values[ok],
                      "adep": cache["ADEP_mvt"].astype(str).values[ok]})
    ref = frame[["MVT_ID_mvt", "TAXITIME_SEC_mvt", "month", "ADEP_mvt"]].rename(
        columns={"TAXITIME_SEC_mvt": "y_f", "month": "month_f", "ADEP_mvt": "adep_f"})
    m = m.merge(ref, left_on="id", right_on="MVT_ID_mvt", how="left", validate="1:1")
    bad = ((m["y"] != m["y_f"]) | (m["month"] != m["month_f"])
           | (m["adep"] != m["adep_f"].astype(str)) | m["MVT_ID_mvt"].isna())
    if bad.any():
        raise RuntimeError(f"id map failed verification: {int(bad.sum())} of {len(m)} rows wrong")


def recover_null_ids(cache: pd.DataFrame, frame: pd.DataFrame, ids: pd.Series) -> pd.Series:
    """Fill the null ids from the frame_train leftovers, 1:1 per airport."""
    ids = ids.copy()
    left = frame[~frame["MVT_ID_mvt"].isin(set(ids.dropna()))].copy()
    null_idx = np.where(ids.isna())[0]
    lk = content_key(left, "sd")
    nk = content_key(cache.iloc[null_idx], "sched_delay")
    uniq_l = set(lk.value_counts().pipe(lambda s: s[s == 1]).index)
    uniq_n = set(nk.value_counts().pipe(lambda s: s[s == 1]).index)
    both = lk.isin(uniq_n) & lk.isin(uniq_l)
    one = nk.isin(set(lk[both]))
    ids.iloc[null_idx[one.values]] = nk[one].map(dict(zip(lk, left["MVT_ID_mvt"]))).values
    left = frame[~frame["MVT_ID_mvt"].isin(set(ids.dropna()))].copy()
    log.info("content key recovered %d of %d null ids",
             len(null_idx) - int(ids.isna().sum()), len(null_idx))
    adep = cache["ADEP_mvt"].astype(str)
    for apt in sorted({adep.iloc[i] for i in np.where(ids.isna())[0]}):
        ids = recover_by_time_rank(ids, adep, frame, left, apt)
    if ids.isna().any():
        raise RuntimeError(f"{int(ids.isna().sum())} ids stayed unrecovered")
    return ids


def recover_by_time_rank(ids: pd.Series, adep: pd.Series, frame: pd.DataFrame,
                         left: pd.DataFrame, apt: str) -> pd.Series:
    """Assign the remaining airport rows by mvt_ts rank; check the time order."""
    pos = np.where(ids.isna() & (adep == apt))[0]
    cand = left[left["ADEP_mvt"].astype(str) == apt].sort_values("mvt_ts")
    if len(cand) != len(pos):
        raise RuntimeError(f"{apt}: {len(cand)} leftover ids for {len(pos)} null rows")
    ts = frame.set_index("MVT_ID_mvt")["mvt_ts"]
    seq = ids.map(ts)
    for rank, i in enumerate(pos):
        ids.iloc[i] = cand["MVT_ID_mvt"].iloc[rank]
        seq.iloc[i] = cand["mvt_ts"].iloc[rank]
    if not seq[adep == apt].dropna().is_monotonic_increasing:
        raise RuntimeError(f"{apt}: time-rank assignment broke the cache time order")
    log.info("%s: %d ids assigned by time rank", apt, len(pos))
    return ids


def main() -> None:
    with open(OLD_CACHE + ".feat.txt") as f:
        feat = f.read().split()
    cache = pd.read_parquet(OLD_CACHE)
    frame = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt",
                                               "TAXITIME_SEC_mvt", "month", "sd", "mvt_ts",
                                               "mvt_eobt1", *TEMPO_COLS])
    ids = pd.read_parquet(OLD_IDS)["MVT_ID_mvt"]
    log.info("old side file: %d null ids", ids.isna().sum())
    verify_content(ids, cache, frame)
    ids = recover_null_ids(cache, frame, ids)
    if not ids.is_unique:
        raise RuntimeError("recovered ids are not unique")
    verify_content(ids, cache, frame)
    out = cache.copy()
    out["MVT_ID_mvt"] = ids.values
    out = out.merge(frame[["MVT_ID_mvt", *TEMPO_COLS]], on="MVT_ID_mvt", how="left", validate="1:1")
    log.info("tempo coverage %.4f on %d rows", out["nb_eobt_med_apt30"].notna().mean(), len(out))
    out.to_parquet(OUT)
    with open(OUT + ".feat.txt", "w") as f:
        f.write("\n".join(feat + TEMPO_COLS))
    log.info("wrote %s: %d rows, %d columns", OUT, len(out), len(out.columns))


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    """The H1 frame and the 110 served columns of the v41 stack."""
    with open(OUT + ".feat.txt") as f:
        feat = f.read().splitlines()
    return pd.read_parquet(OUT), feat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
