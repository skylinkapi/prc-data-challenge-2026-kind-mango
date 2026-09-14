"""H4: coverage monitor for the served columns and the OPDI record counts.

Per column, non-null and non-zero share for 2025, January 2026 and July 2026,
per airport. The 110 base columns are strict: the build fails when a non-null
share drops by more than 20 points against 2025, or when a non-zero drop
exceeds 20 points both against full-2025 and against the same month of 2025
(the month match absorbs seasonality). The LIRF-only columns (ec_*, opdi_*,
fbrate_*) are informational: the deployed LIRF head serves them with the
known 2026 NaN pattern, validated live. The OPDI table counts the
taxi_out_v2 records per airport and month (appendix A3) and flags an airport
whose January or July 2026 count is under 80 % of its 2025 monthly mean; L6
reads that flag for LSZH.

2025 sides: models/h1_frame_cache.parquet (base) and models/lirf_frame_cache.parquet
(LIRF extras, with the v23 rate maps applied). 2026 side: a dump from
predict_v30.main(dump_features=...).
Writes models/coverage_monitor.json; exit 1 on a strict failure.
"""
import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from build_h1_frame_cache import OUT as H1_CACHE
from eval_v33_holdout import LIRF_CACHE, read_list
from predict_v23 import add_rate_features_scoring
from train_r_all_v40 import TEMPO_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
OPDI = os.path.join(ROOT, "external", "opdi", "taxi_out_v2.parquet")
FRAME_RANK = os.path.join(MODELS, "v2", "cache", "frame_rank.parquet")
OUT = os.path.join(MODELS, "coverage_monitor.json")
DROP_POINTS = 20.0
log = logging.getLogger(__name__)


def shares(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Non-null and non-zero share per column on one frame slice."""
    have = [c for c in cols if c in df.columns]
    d = df[have].astype("object")
    nn = d.notna().mean()
    nz = (1 - (d.isna() | (d == 0))).mean()
    return pd.DataFrame({"nonnull": nn, "nonzero": nz})


def compare(base_cols, ref_all, ref_month, new, apt, period, fails):
    """One airport-period table; append strict failures to fails."""
    rows = []
    for c in base_cols:
        if c not in new.index:
            rows.append({"column": c, "airport": apt, "period": period, "missing_in_2026": True})
            fails.append(f"{c} at {apt} in {period}: absent from the 2026 dump")
            continue
        nn_drop = 100 * (ref_all.loc[c, "nonnull"] - new.loc[c, "nonnull"])
        nz_drop = 100 * (ref_all.loc[c, "nonzero"] - new.loc[c, "nonzero"])
        nz_drop_m = 100 * (ref_month.loc[c, "nonzero"] - new.loc[c, "nonzero"])
        rows.append({"column": c, "airport": apt, "period": period,
                     "nonnull_2025": float(ref_all.loc[c, "nonnull"]),
                     "nonnull_2026": float(new.loc[c, "nonnull"]),
                     "nonnull_drop_pts": float(nn_drop),
                     "nonzero_2025": float(ref_all.loc[c, "nonzero"]),
                     "nonzero_2026": float(new.loc[c, "nonzero"]),
                     "nonzero_drop_pts": float(nz_drop),
                     "nonzero_drop_month_matched_pts": float(nz_drop_m)})
        if nn_drop > DROP_POINTS:
            fails.append(f"{c} at {apt} in {period}: non-null share dropped {nn_drop:.0f} points")
        if nz_drop > DROP_POINTS and nz_drop_m > DROP_POINTS:
            fails.append(f"{c} at {apt} in {period}: non-zero share dropped {nz_drop:.0f} points")
    return rows


def lirf_2025(cols: list[str]) -> pd.DataFrame:
    """Shares of the LIRF-only columns on the 2025 LIRF frame."""
    dep = pd.read_parquet(LIRF_CACHE)
    with open(os.path.join(MODELS, "lirf_regime_v23.rate_maps.pkl"), "rb") as f:
        bundle = pickle.load(f)
    dep = add_rate_features_scoring(dep, bundle["maps"], bundle["base_rate"])
    return shares(dep, cols)


def opdi_counts() -> dict:
    """Taxi-out record counts per airport: 2025 monthly mean, Jan 2026, Jul 2026."""
    d = pd.read_parquet(OPDI, columns=["osm_airport", "pushback_ts"])
    d["year"] = d["pushback_ts"].dt.year
    d["month"] = d["pushback_ts"].dt.month
    out = {}
    for a, g in d.groupby("osm_airport"):
        m25 = g[g["year"] == 2025].groupby("month").size()
        jan25, jul25 = int(m25.get(1, 0)), int(m25.get(7, 0))
        jan26 = int(((g["year"] == 2026) & (g["month"] == 1)).sum())
        jul26 = int(((g["year"] == 2026) & (g["month"] == 7)).sum())
        within = (jan26 >= 0.8 * jan25 and jul26 >= 0.8 * jul25) if jan25 and jul25 else False
        out[a] = {"mean_2025": round(float(m25.mean()), 1), "jan_2025": jan25, "jul_2025": jul25,
                  "jan_2026": jan26, "jul_2026": jul26, "within_20pct_month_matched": within}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="parquet from predict_v30.main(dump_features=...)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    base_cols = read_list("lgbm_r_all_v40.features.txt")
    rn_cols = read_list("lirf_regime_v41.features.txt")
    pfb_cols = read_list("lirf_regime_v23.features.txt")
    extra_cols = sorted((set(rn_cols) | set(pfb_cols)) - set(base_cols))

    dep25 = pd.read_parquet(H1_CACHE, columns=base_cols)
    dump = pd.read_parquet(args.dump)
    tempo_missing = [c for c in TEMPO_COLS if c not in dump.columns]
    if tempo_missing:
        fr = pd.read_parquet(FRAME_RANK, columns=["MVT_ID_mvt", *tempo_missing])
        dump = dump.merge(fr, on="MVT_ID_mvt", how="left", validate="1:1")
        log.info("joined %d tempo columns from the src_v2 ranking frame", len(tempo_missing))
    fails, table = [], []
    for apt in sorted(dump["ADEP_mvt"].astype(str).unique()):
        ref = dep25[dep25["ADEP_mvt"].astype(str) == apt]
        new = dump[dump["ADEP_mvt"].astype(str) == apt]
        ref_all = shares(ref, base_cols)
        for period, months in (("2026_jan", {1}), ("2026_jul", {7})):
            ref_m = shares(ref[ref["month"].isin(months)], base_cols)
            table += compare(base_cols, ref_all, ref_m, shares(new[new["month"].isin(months)], base_cols),
                             apt, period, fails)
    report = {"strict_columns": len(base_cols), "failures": fails, "table": table,
              "lirf_only_2025": lirf_2025(extra_cols)["nonnull"].round(3).to_dict(),
              "lirf_only_2026": {p: shares(dump[(dump["ADEP_mvt"].astype(str) == "LIRF")
                                                & dump["month"].isin(m)], extra_cols)["nonnull"].round(3).to_dict()
                                 for p, m in (("jan", {1}), ("jul", {7}))},
              "opdi_taxi_out_counts": opdi_counts()}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    log.info("strict columns %d; failures %d; wrote %s", len(base_cols), len(fails), args.out)
    for x in fails:
        log.error("COVERAGE: %s", x)
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
