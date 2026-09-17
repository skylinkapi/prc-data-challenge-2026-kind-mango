"""L8: calendar columns from the `holidays` package.

For each row, flag whether the ADEP country observes a public holiday on the
MVT_TIME date, and whether the day is a weekend. Six countries cover the ten
target airports. Writes calendar_{train,rank}.parquet keyed on MVT_ID_mvt.
"""
import logging
import os

import holidays
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_train.parquet")
RANK_FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_rank.parquet")
OUT_TRAIN = os.path.join(ROOT, "models", "calendar_train.parquet")
OUT_RANK = os.path.join(ROOT, "models", "calendar_rank.parquet")

ICAO_COUNTRY = {"EDDF": "DE", "EDDM": "DE", "EGLL": "GB", "EHAM": "NL",
                "LEBL": "ES", "LEMD": "ES", "LFPG": "FR", "LIRF": "IT",
                "LSZH": "CH", "LTFM": "TR"}
log = logging.getLogger(__name__)


def holiday_map(years: list[int]) -> dict[str, set]:
    return {c: set(holidays.country_holidays(c, years=years).keys())
            for c in set(ICAO_COUNTRY.values())}


def build(path: str, out: str) -> None:
    if os.path.exists(out):
        log.info("cache hit: %s", out)
        return
    log.info("read %s", path)
    frame = pd.read_parquet(path, columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_ts"])
    ts = pd.to_datetime(frame["mvt_ts"], utc=True, errors="coerce")
    date = ts.dt.date
    years = sorted(set(ts.dt.year.dropna().astype(int)))
    hmap = holiday_map(years)
    country = frame["ADEP_mvt"].astype(str).map(ICAO_COUNTRY)
    is_hol = pd.Series([d in hmap.get(c, set()) if pd.notna(d) and c else False
                        for d, c in zip(date, country)], index=frame.index).astype("Int8")
    dow = ts.dt.dayofweek
    is_wknd = (dow >= 5).astype("Int8")
    is_hol_dow = ((is_hol == 1) | (is_wknd == 1)).astype("Int8")
    out_df = pd.DataFrame({"MVT_ID_mvt": frame["MVT_ID_mvt"].values,
                           "is_public_hol": is_hol.values,
                           "is_weekend": is_wknd.values,
                           "is_hol_or_wknd": is_hol_dow.values})
    log.info("public hol coverage %.4f, weekend %.4f",
             (is_hol == 1).mean(), (is_wknd == 1).mean())
    out_df.to_parquet(out)
    log.info("wrote %s (%d rows)", out, len(out_df))


def main() -> None:
    build(FRAME, OUT_TRAIN)
    build(RANK_FRAME, OUT_RANK)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
