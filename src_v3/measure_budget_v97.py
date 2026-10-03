"""Error budget of the served v97 recipe on the 2025 hold-out, projected onto the 2026 coverage mix.

The hold-out prediction follows the v97 serve path outside LIRF: the v97
member blend, the MLP member at `AUTO_MLP_WEIGHT`, the v95 stack
cross-fitted January against July, and the A5 stand bound. Each row gets
a label class (`labels.classify`) and a coverage class: `adsb_own` (a
clock at the assigned stand), `adsb_seg` (an ADS-B ground segment only),
`opdi_only` or `none`. The projection multiplies the hold-out mean squared
error of each airport and coverage cell by the 2026 row count of that
cell. It assumes no drift inside a cell.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3 import stack_pushback_v90 as sp
from src_v3.build_adsb_clock import OUT_RANK as ADSB_RANK
from src_v3.build_nm_clocks_v73 import load_dep
from src_v3.build_opdi_own import OUT_RANK as OPDI_RANK
from src_v3.labels import classify
from src_v3.tune_stack_v93 import frame as stack_frame

BLEND = C.MODELS / "blend_v97_holdout.parquet"
ADSB_TRAIN = C.ADSB_DIR / "v97" / "adsb_clock_train.parquet"   # the frozen table the v97 stack read
RESULT = C.ROOT / "submission" / "kind-mango_v97.result.json"
REPORT = C.MODELS / "budget_v97.holdout.json"
log = logging.getLogger(__name__)


def coverage(f: pd.DataFrame) -> np.ndarray:
    own = f[["adsb_own_move", "adsb_own_last"]].notna().any(axis=1)
    return np.select([own, f["adsb_seg"].notna(), f["opdi_first"].notna()], ["adsb_own", "adsb_seg", "opdi_only"], "none")


def holdout_rows() -> pd.DataFrame:
    """Hold-out rows outside LIRF with the served prediction `final`, the prediction `base` without the stack, and the clocks."""
    f = stack_frame(C.AUTO_MLP_WEIGHT, False, adsb_train=ADSB_TRAIN, blend_path=BLEND)
    s, _ = sp.crossfit(f)
    final = f.set_index("MVT_ID_mvt")["base"].copy()
    final.loc[s["MVT_ID_mvt"].values] = np.fmax(s["stack"].values, sp.stand_bound(s))
    f["final"] = final.loc[f["MVT_ID_mvt"].values].values
    dep = load_dep(sorted(C.TRAIN_DIR.glob("training_2025-0[17]-*.parquet")))
    dep["sd"] = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    f = f.merge(dep[["MVT_ID_mvt", "sd"]], on="MVT_ID_mvt", how="left", validate="1:1")
    f["apt"] = f["apt"].astype(str)
    f["cls"] = classify(f["y"].values, f["sd"].values, f["apt"].values)
    f["cov"] = coverage(f)
    return f


def table(f: pd.DataFrame, by: list[str]) -> list[dict]:
    """Rows, MSE with and without the stack on the hold-out scale, and RMSE, per group."""
    g = f.assign(e2=(f["final"] - f["y"]) ** 2, e2_no_stack=(f["base"] - f["y"]) ** 2).groupby(by)
    t = pd.DataFrame({"rows": g.size(), "mse": g["e2"].sum() / C.HOLDOUT_ROWS,
                      "mse_no_stack": g["e2_no_stack"].sum() / C.HOLDOUT_ROWS, "rmse": np.sqrt(g["e2"].mean())})
    return t.round(1).reset_index().to_dict("records")


def clocks(f: pd.DataFrame) -> dict:
    """Clean rows: share of each clock within 60 s after the airport median bias; best clock per row on own-stand rows."""
    c = f[f["cls"] == "clean"]
    out = {}
    for k in C.BUDGET_CLOCKS:
        m = c[k].notna()
        d = c.loc[m, k] - c.loc[m, "y"]
        r = (d - d.groupby(c.loc[m, "apt"]).transform("median")).abs()
        out[k] = {"rows": int(m.sum()), "within_60s": round(float((r <= 60).mean()), 3),
                  "served_rmse": round(float(np.sqrt(((c.loc[m, "final"] - c.loc[m, "y"]) ** 2).mean())), 1)}
    own = c[c["cov"] == "adsb_own"]
    best = own[list(C.BUDGET_CLOCKS)].sub(own["y"], axis=0).abs().min(axis=1)
    out["own_rows_best_clock_rmse"] = round(float(np.sqrt((best ** 2).mean())), 1)
    out["own_rows_served_rmse"] = round(float(np.sqrt(((own["final"] - own["y"]) ** 2).mean())), 1)
    return out


def segment_rows(f: pd.DataFrame) -> dict:
    """Clean segment-only rows: why no stand clock, RMSE by first-point distance to the stand; and segments near the lookback."""
    s = f[(f["cls"] == "clean") & (f["cov"] == "adsb_seg")]
    e = s["final"] - s["y"]
    band = pd.cut(s["adsb_own_d0"], list(C.BUDGET_D0_EDGES_M))
    by_d0 = {str(k): {"rows": int(v.size), "rmse": round(float(np.sqrt((v ** 2).mean())), 1)}
             for k, v in e.groupby(band, observed=True)}
    long = f[f["adsb_seg"] > C.BUDGET_LONG_SEGMENT_S]
    return {"rows": len(s), "first_point_moving": round(float((s["adsb_gs0"] >= C.ADSB_MOVING_KT).mean()), 3),
            "assigned_stand_unknown": round(float(s["adsb_own_d0"].isna().mean()), 3), "rmse_by_first_point_distance_m": by_d0,
            "long_segments": {"rows": len(long), "by_airport": long["apt"].value_counts().to_dict(),
                              "label_median": float(long["y"].median()), "label_above_7200": int((long["y"] > C.Y_CLEAN_MAX).sum())}}


def clean_concentration(f: pd.DataFrame) -> dict:
    """Share of clean rows above each absolute error, and their share of the clean squared error."""
    e = (f.loc[f["cls"] == "clean", "final"] - f.loc[f["cls"] == "clean", "y"]).abs()
    return {str(k): {"rows": round(float((e > k).mean()), 4), "mse": round(float((e[e > k] ** 2).sum() / (e ** 2).sum()), 3)}
            for k in C.BUDGET_ERR_EDGES_S}


def coverage_2025_by_month() -> dict:
    """Share of 2025 departures with an ADS-B ground segment and with a stand clock, per airport and month."""
    d = load_dep(sorted(C.TRAIN_DIR.glob("training_2025-*.parquet")))[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]]
    d = d.merge(pd.read_parquet(ADSB_TRAIN), on="MVT_ID_mvt", how="left", validate="1:1")
    d = d.assign(month=d["MVT_TIME_UTC_mvt"].dt.month, segment=d["adsb_seg"].notna(),
                 own=d[["adsb_own_move", "adsb_own_last"]].notna().any(axis=1))
    return {k: d.pivot_table(index="ADEP_mvt", columns="month", values=k, aggfunc="mean").round(2).to_dict("index")
            for k in ("segment", "own")}


def rank_coverage() -> pd.DataFrame:
    """2026 departures outside LIRF with their airport and coverage class."""
    r = load_dep([C.RANK])[["MVT_ID_mvt", "ADEP_mvt"]]
    for path in (ADSB_RANK, OPDI_RANK):
        r = r.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="1:1")
    r = r[r["ADEP_mvt"] != "LIRF"]
    return r.assign(apt=r["ADEP_mvt"].astype(str), cov=coverage(r))


def project(f: pd.DataFrame, r: pd.DataFrame) -> dict:
    """Live-scale MSE of the normal classes per airport and coverage cell, and the remainder for LIRF and the tail."""
    n = f[f["cls"].isin(C.BUDGET_NORMAL_CLASSES)]
    e2 = ((n["final"] - n["y"]) ** 2).groupby([n["apt"], n["cov"]]).agg(["size", "mean"])
    pool = ((n["final"] - n["y"]) ** 2).groupby(n["cov"]).mean()
    cells = []
    for (a, c), rows in r.groupby(["apt", "cov"]).size().items():
        own = (a, c) in e2.index and e2.loc[(a, c), "size"] >= C.BUDGET_MIN_CELL_ROWS
        cells.append({"apt": a, "cov": c, "rows_2026": int(rows), "source": "cell" if own else "pooled",
                      "live_mse": round(float(rows * (e2.loc[(a, c), "mean"] if own else pool[c]) / C.RANK_ROWS), 1)})
    t = pd.DataFrame(cells)
    live = json.loads(RESULT.read_text())["score"] ** 2
    return {"cells": cells, "by_coverage": t.groupby("cov")["live_mse"].sum().round(0).to_dict(),
            "by_airport": t.groupby("apt")["live_mse"].sum().round(0).to_dict(), "normal_classes": round(float(t["live_mse"].sum())),
            "live_mse_v97": round(live), "remainder_lirf_and_tail": round(live - float(t["live_mse"].sum()))}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    f = holdout_rows()
    r = rank_coverage()
    share = {"holdout_2025": pd.crosstab(f["apt"], f["cov"], normalize="index").round(3).to_dict("index"),
             "rank_2026": pd.crosstab(r["apt"], r["cov"], normalize="index").round(3).to_dict("index"),
             "rows_2026": r.groupby(["apt", "cov"]).size().unstack(fill_value=0).to_dict("index"),
             "by_month_2025": coverage_2025_by_month()}
    report = {"rows": len(f), "mse": round(float(((f["final"] - f["y"]) ** 2).sum() / C.HOLDOUT_ROWS)),
              "mse_no_stack": round(float(((f["base"] - f["y"]) ** 2).sum() / C.HOLDOUT_ROWS)),
              "by_class": table(f, ["cls"]), "by_coverage": table(f, ["cov"]), "by_airport_class": table(f, ["apt", "cls"]),
              "by_airport_coverage": table(f, ["apt", "cov"]), "clean_concentration": clean_concentration(f),
              "clocks": clocks(f), "segment_rows": segment_rows(f), "coverage_share": share, "projection_2026": project(f, r)}
    REPORT.write_text(json.dumps(report, indent=1))
    log.info("hold-out MSE %d (no stack %d); 2026 normal classes %d, remainder %d -> %s", report["mse"], report["mse_no_stack"],
             report["projection_2026"]["normal_classes"], report["projection_2026"]["remainder_lirf_and_tail"], REPORT.name)


if __name__ == "__main__":
    main()
