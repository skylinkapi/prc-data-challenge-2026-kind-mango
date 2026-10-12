"""P1: the precision of the take-off time marks a regime at LFPG.

At LFPG every movement time is a whole minute plus organiser noise of at most `MVT_GRID_JITTER_S` seconds.
97 departures of 2025 have an `MVT` off that grid. Their take-off time comes from another source, while the
off-block stayed near `LOBT`: the label has a median of 4,453 s against 1,191 s for on-grid rows with the same
`MVT - LOBT` and `MVT - AOBT_3`. The trees cannot see the flag. `serve` raises the value of a flagged row to
`MVT - LOBT` minus the 2025 median `MVT - LOBT - y` of the flagged rows; a lower value is not changed.
`measure` prices the rule on the January and July 2025 hold-out with the offset of the other ten months.
`serve --also FILE` reads another organiser release of the same movements. Each release draws its own noise, so a
row that lies off the grid in one release is off the grid (final phase: the ranking file of the first phase).
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.lobt_window import window
from src_v3.measure_tail_rules import HOLD

COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]
REPORT = C.MODELS / "mvt_precision.holdout.json"
log = logging.getLogger(__name__)


def flagged(paths: list, label: bool = False, known: pd.Series | None = None) -> pd.DataFrame:
    """Departures at `MVT_PRECISION_AIRPORTS` whose `MVT` lies off the minute grid, with `ml` = `MVT - LOBT` in seconds.

    `known` holds ids that another release of the same movements shows off the grid."""
    d = pd.concat([pd.read_parquet(p, columns=COLS + ["TAXITIME_SEC_mvt"] * label) for p in paths], ignore_index=True)
    s = d["MVT_TIME_UTC_mvt"].dt.second
    off = (s > C.MVT_GRID_JITTER_S) & (s < 60 - C.MVT_GRID_JITTER_S)
    if known is not None:
        off |= d["MVT_ID_mvt"].isin(known)
    d = d[(d["PHASE_mvt"] == "DEP") & d["ADEP_mvt"].isin(C.MVT_PRECISION_AIRPORTS) & off].copy()
    d["ml"] = (d["MVT_TIME_UTC_mvt"] - d["LOBT_flt"]).dt.total_seconds()
    d["month"] = d["MVT_TIME_UTC_mvt"].dt.month
    return d


def raise_to_rule(pred: np.ndarray, ml: np.ndarray, offset: float) -> np.ndarray:
    """`pred` raised to `ml - offset` where `ml` is known, then projected onto the LOBT window."""
    return window(np.where(np.isnan(ml), pred, np.fmax(pred, ml - offset)), ml)


def median_offset(rows: pd.DataFrame) -> float:
    return float(np.nanmedian(rows["ml"] - rows["TAXITIME_SEC_mvt"]))


def measure() -> None:
    from src_v3.measure_nm_join import lobt_terms, outside_lirf, project
    paths = sorted(C.TRAIN_DIR.glob("training_2025-*.parquet"))
    f = flagged(paths, label=True)
    loo = np.concatenate([f.loc[f["month"] == m, "ml"] - median_offset(f[f["month"] != m]) for m in sorted(f["month"].unique())])
    y_loo = np.concatenate([f.loc[f["month"] == m, "TAXITIME_SEC_mvt"] for m in sorted(f["month"].unique())])
    o = outside_lirf().merge(lobt_terms(paths), on="MVT_ID_mvt", how="left", validate="1:1")
    served = project(o["final"].values, o)
    h = o[["MVT_ID_mvt"]].assign(served=served).merge(f[f["month"].isin(HOLD)], on="MVT_ID_mvt", validate="1:1")
    offset = median_offset(f[~f["month"].isin(HOLD)])
    new = raise_to_rule(h["served"].values, h["ml"].values, offset)
    y = h["TAXITIME_SEC_mvt"].values.astype(float)
    d = ((new - y) ** 2 - (h["served"].values - y) ** 2) / C.HOLDOUT_ROWS
    report = {"rows_2025": len(f), "label_quantiles_2025": f["TAXITIME_SEC_mvt"].quantile([.05, .5, .95]).round(0).tolist(),
              "offset_all_months_s": median_offset(f), "offset_fit_months_s": offset,
              "loo_month_rmse_s": float(np.sqrt(np.mean((loo - y_loo) ** 2))),
              "holdout_rows": len(h), "served_rmse_s": float(np.sqrt(np.mean((h["served"].values - y) ** 2))),
              "rule_rmse_s": float(np.sqrt(np.mean((new - y) ** 2))),
              "mse": round(float(d.sum()), 1), "body": round(float(d[y <= C.BODY_MAX_S].sum()), 1),
              "ranking_rows": len(flagged([C.RANK]))}
    REPORT.write_text(json.dumps(report, indent=1))
    log.info("%s -> %s", report, REPORT.name)


def serve(src: str, out: str, also: list[str]) -> None:
    offset = median_offset(flagged(sorted(C.TRAIN_DIR.glob("training_2025-*.parquet")), label=True))
    sub = pd.read_parquet(C.ROOT / "submission" / src)
    known = flagged([C.ROOT / p for p in also])["MVT_ID_mvt"] if also else None
    f = sub[["MVT_ID_mvt"]].merge(flagged([C.RANK], known=known)[["MVT_ID_mvt", "ml"]], on="MVT_ID_mvt", how="left",
                                  validate="1:1")
    pred = sub["TAXITIME_SEC_mvt"].values.astype(float)
    new = raise_to_rule(pred, f["ml"].values, offset)
    sub.assign(TAXITIME_SEC_mvt=new.astype("float64")).to_parquet(C.ROOT / "submission" / out, index=False)
    log.info("offset %.1f s; %d flagged rows, %d raised, mean raise %.0f s", offset, int(f["ml"].notna().sum()),
             int((new > pred).sum()), float((new - pred)[new > pred].mean()) if (new > pred).any() else 0.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("measure", "serve"))
    ap.add_argument("src", nargs="?", help="serve: served file in submission/")
    ap.add_argument("out", nargs="?", help="serve: output file in submission/")
    ap.add_argument("--also", nargs="+", default=[], help="serve: other organiser releases of the same movements.")
    args = ap.parse_args()
    measure() if args.mode == "measure" else serve(args.src, args.out, args.also)


if __name__ == "__main__":
    main()
