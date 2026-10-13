"""Month level: shift the served values of an airport and month to the mean taxi-out that EUROCONTROL publishes.

EUROCONTROL publishes per airport and month the reference and the additional taxi-out time of the departures
that have a reference (ansperformance.eu, `taxi_out_additional_time`). Their sum per departure is the mean
taxi-out of those departures, so the file is an aggregate of the labels. The organisers allow it (Discord,
2026-10-08). On the 2025 labels up to `Y_CLEAN_MAX`, the label mean minus the published mean is stable from April
to October at seven airports (EGLL -1 s, LEBL 7 s); in winter the file drops the long departures and the gap
grows to 164 s. `serve` moves a month of an airport only if the month is in `MONTH_LEVEL_MONTHS`, the gap of the
airport has a leave-one-month-out error of at most `MONTH_LEVEL_MAX_ERR_S`, and the file keeps a share of the
departures that the 2025 months also show. LIRF is skipped: its served values hold a hedge for the labels above
`Y_CLEAN_MAX`. With `--kept-fit` (v139) the gap is a line on the share of dropped departures where that share
varies in 2025; at LFPG the gap grows from 10 s at 8 % dropped to 45 s at 30 %.
`measure FILE` prices the rule on a hold-out file with the columns `MVT_ID_mvt`, `y`, `pred`, `ADEP_mvt` and
`MVT_TIME_UTC_mvt`; the airport gap leaves the priced month out. Data (c) EUROCONTROL, used with attribution.
"""
from __future__ import annotations

import argparse
import logging
import urllib.request

import numpy as np
import pandas as pd

from src_v3 import config as C

KEYS = ["apt", "year", "month"]
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]
log = logging.getLogger(__name__)


def published() -> pd.DataFrame:
    """Published mean taxi-out and kept share per airport, year and month, from the yearly CSVs (downloaded once)."""
    parts = []
    for year in C.ATFM_SLOT_YEARS:
        path = C.ATFM_SLOT_DIR / f"taxi_out_additional_time_{year}.csv"
        if not path.exists():
            urllib.request.urlretrieve(C.EC_CSV_URL.format("taxi_out_additional_time", year), path)
        parts.append(pd.read_csv(path))
    d = pd.concat(parts, ignore_index=True)
    d = d[d["APT_ICAO"].isin(C.TARGETS)]
    return pd.DataFrame({"apt": d["APT_ICAO"].values, "year": d["YEAR"].values, "month": d["MONTH_NUM"].astype(int).values,
                         "pub": ((d["TOTAL_REF_TIME_MIN"] + d["TOTAL_ADD_TIME_MIN"]) / d["TOTAL_REF_NB_FL"]).values * 60,
                         "kept": (d["TOTAL_REF_NB_FL"] / d["TF"]).values})


def keyed(rows: pd.DataFrame) -> pd.DataFrame:
    """Add the keys `apt`, `year` and `month` of each row."""
    t = rows["MVT_TIME_UTC_mvt"].dt
    return rows.assign(apt=rows["ADEP_mvt"].astype(str), year=t.year, month=t.month)


def means(rows: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean of `value` up to `Y_CLEAN_MAX` per airport, year and month."""
    r = keyed(rows[rows[value] <= C.Y_CLEAN_MAX])
    return r.groupby(KEYS)[value].mean().rename("mean").reset_index()


def gap_fit(g: pd.DataFrame, kept_fit: bool) -> tuple[float, float]:
    """Slope and intercept of the gap on the share of dropped departures; a constant gap if that share hardly varies."""
    if not kept_fit or g["kept"].max() - g["kept"].min() < C.MONTH_LEVEL_MIN_KEPT_RANGE:
        return 0.0, float(g["gap"].median())
    slope, icpt = np.polyfit(1 - g["kept"], g["gap"], 1)
    return float(slope), float(icpt)


def mapping(labels: pd.DataFrame, pub: pd.DataFrame, kept_fit: bool, skip_month: int | None = None) -> pd.DataFrame:
    """Per airport over the calibration months: the gap fit, its leave-one-month-out error, the range of the kept share."""
    g = means(labels, "TAXITIME_SEC_mvt").merge(pub, on=KEYS)
    g = g[g["month"].isin(C.MONTH_LEVEL_MONTHS) & (g["month"] != skip_month)]
    g = g.assign(gap=g["mean"] - g["pub"])
    rows = []
    for apt, d in g.groupby("apt"):
        loo = [r.gap - np.polyval(gap_fit(d[d["month"] != r.month], kept_fit), 1 - r.kept) for r in d.itertuples()]
        slope, icpt = gap_fit(d, kept_fit)
        rows.append({"apt": apt, "slope": slope, "icpt": icpt, "err": float(np.sqrt(np.mean(np.square(loo)))),
                     "kept_min": d["kept"].min(), "kept_max": d["kept"].max()})
    return pd.DataFrame(rows)


def shifts(served: pd.DataFrame, pub: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """Shift of each reliable airport month: the published mean plus the airport gap, minus the served mean."""
    c = means(served, "pred").merge(pub, on=KEYS).merge(gaps, on="apt")
    c = c.assign(gap=c["icpt"] + c["slope"] * (1 - c["kept"]))
    ok = (c["month"].isin(C.MONTH_LEVEL_MONTHS) & ~c["apt"].isin(C.MONTH_LEVEL_SKIP) & (c["err"] <= C.MONTH_LEVEL_MAX_ERR_S)
          & c["kept"].between(c["kept_min"] - C.MONTH_LEVEL_KEPT_SLACK, c["kept_max"] + C.MONTH_LEVEL_KEPT_SLACK))
    return c[ok].assign(shift=c["pub"] + c["gap"] - c["mean"])


def moved(served: pd.DataFrame, s: pd.DataFrame) -> np.ndarray:
    """`pred` plus the shift of its airport month; values above `Y_CLEAN_MAX` and other months do not move."""
    shift = keyed(served)[KEYS].merge(s[[*KEYS, "shift"]], on=KEYS, how="left")["shift"].fillna(0).values
    pred = served["pred"].values.astype(float)
    return np.where(pred <= C.Y_CLEAN_MAX, np.clip(pred + shift, 0, None), pred)


def departures(paths: list, label: bool = False) -> pd.DataFrame:
    """Departure rows of the given movement files."""
    d = pd.concat([pd.read_parquet(p, columns=COLS + ["TAXITIME_SEC_mvt"] * label) for p in paths], ignore_index=True)
    return d[d["PHASE_mvt"] == "DEP"]


def labels_2025() -> pd.DataFrame:
    return departures(sorted(C.TRAIN_DIR.glob("training_2025-*.parquet")), label=True)


def serve(src: str, out: str, kept_fit: bool) -> None:
    """Shift the reliable airport months of the file `src` and write `out`; log each shift."""
    sub = pd.read_parquet(C.ROOT / "submission" / src)
    served = sub.rename(columns={"TAXITIME_SEC_mvt": "pred"}).merge(
        departures([C.RANK]).drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="left", validate="1:1")
    pub = published()
    s = shifts(served, pub, mapping(labels_2025(), pub, kept_fit))
    new = moved(served, s)
    sub.assign(TAXITIME_SEC_mvt=new.astype("float64")).to_parquet(C.ROOT / "submission" / out, index=False)
    for r in s.itertuples():
        log.info("%s %d-%02d: published %.1f s, gap %.1f s, served mean %.1f s, shift %+.1f s", r.apt, r.year, r.month,
                 r.pub, r.gap, r.mean, r.shift)
    log.info("%d airport months and %d rows moved", len(s), int((new != served["pred"].values).sum()))


def measure(path: str, kept_fit: bool) -> None:
    """Price on a hold-out file; the gap of an airport comes from the calibration months other than the priced month."""
    h = pd.read_parquet(path)
    pub, labels = published(), labels_2025()
    new = h["pred"].values.astype(float)
    for m in sorted(set(h["MVT_TIME_UTC_mvt"].dt.month) & set(C.MONTH_LEVEL_MONTHS)):
        rows = (h["MVT_TIME_UTC_mvt"].dt.month == m).values
        s = shifts(h[rows], pub, mapping(labels, pub, kept_fit, skip_month=m))
        new[rows] = moved(h[rows], s)
        log.info("month %d shifts: %s", m, s.set_index("apt")["shift"].round(1).to_dict())
    d = pd.Series(((new - h["y"]) ** 2 - (h["pred"] - h["y"]) ** 2).values / C.HOLDOUT_ROWS)
    log.info("price %+.1f MSE; by airport %s", d.sum(), d.groupby(h["ADEP_mvt"].astype(str).values).sum().round(1).to_dict())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("measure", "serve"))
    ap.add_argument("src", help="serve: served file in submission/; measure: hold-out file.")
    ap.add_argument("out", nargs="?", help="serve: output file in submission/.")
    ap.add_argument("--kept-fit", action="store_true", help="Fit the gap on the share of dropped departures (v139).")
    args = ap.parse_args()
    measure(args.src, args.kept_fit) if args.mode == "measure" else serve(args.src, args.out, args.kept_fit)


if __name__ == "__main__":
    main()
