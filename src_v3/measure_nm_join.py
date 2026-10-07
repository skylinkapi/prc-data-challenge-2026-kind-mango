"""Nineteenth pass: hold-out price of the NM join rules on the served v115 path.

Every 2025 departure with an NM record has `|BLOCK - LOBT| <= 3,606` s (2,062,577 rows, no exception):
the organiser joined a movement to its NM flight only inside that window. Three rules follow from it.

1. `window`: on a record row the label lies in `MVT - LOBT` plus or minus 3,606 s. The served value is
   projected onto that interval.
2. `copy_record`: at LIRF a schedule copy (`BLOCK = SCHED`) is impossible on a record row with
   `|SCHED - LOBT| > 3,606` s, so the gate is 0 there.
3. `copy_linked`: a departure without a record failed the join. Where the arrival row of the same flight at
   another of the ten airports carries the NM `LOBT` and `|SCHED - LOBT| <= 3,606` s, a schedule copy would
   have joined, so at LIRF the gate is 0 there.

Outside LIRF the hold-out prediction follows the v115 serve path: the v97 member blend, the MLP and the
v115 TabM member at `V115_SHARES`, the stack cross-fitted January against July on the v115 clock tables,
and the A5 stand bound. At LIRF it follows the head with the out-of-fold g3 gate (nested isotonic map), the
out-of-sample v41 `R_norm`, the Step A band table with the date-crossing fix and the ITY340 rule. For a
served ranking file the report counts the rows each rule moves. For the window it also gives the live gain
it guarantees: a projection onto an interval that holds the label cannot raise the error of any row.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict_v23 import apply_stepA_v22
from src_v3 import config as C
from src_v3 import stack_pushback_v90 as sp
from src_v3.build_nm_clocks_v73 import OUT_TRAIN as NM_TRAIN
from src_v3.build_nm_clocks_v73 import load_dep
from src_v3.catboost_gate import OOF as GATE_OOF
from src_v3.catboost_gate import gate_frame
from src_v3.labels import classify
from src_v3.lobt_window import window
from src_v3.measure_gate_clocks import nested_map, r_norm_v41
from src_v3.measure_tail_rules import HOLD
from src_v3.mlp_member import out_paths
from src_v3.step_a_date import BAND_TABLE, adjust
from train_r_all_v40 import TEMPO_COLS, V2_FRAME

REPORT = C.MODELS / "nm_join.holdout.json"
log = logging.getLogger(__name__)


def seconds(a: pd.Series, b: pd.Series) -> np.ndarray:
    return (a - b).dt.total_seconds().values


def arrival_lobt(paths: list) -> pd.DataFrame:
    """`LOBT` of the arrival row of the same flight number at the destination, for departures without a record."""
    cols = ["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt", "ADES_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]
    m = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    dep = m[(m["PHASE_mvt"] == "DEP") & m["FLIGHT_ID_mvt"].isna()]
    arr = m[(m["PHASE_mvt"] == "ARR") & m["FLIGHT_ID_mvt"].notna()][["FLIGHT_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]]
    j = dep[["MVT_ID_mvt", "FLIGHT_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]].merge(arr, on=["FLIGHT_mvt", "ADES_mvt"], suffixes=("", "_arr"))
    j["lag"] = seconds(j["MVT_TIME_UTC_mvt_arr"], j["MVT_TIME_UTC_mvt"])
    j = j[j["lag"].between(*C.ARR_LINK_S, inclusive="neither")].sort_values("lag").drop_duplicates("MVT_ID_mvt")
    return j[["MVT_ID_mvt", "LOBT_flt"]].rename(columns={"LOBT_flt": "lobt_arr"})


def lobt_terms(paths: list) -> pd.DataFrame:
    """`MVT - LOBT` and `SCHED - LOBT` of every departure; rows without a record take the arrival-side `LOBT`."""
    d = load_dep(paths, ["FLIGHT_ID_mvt"]).merge(arrival_lobt(paths), on="MVT_ID_mvt", how="left", validate="1:1")
    record = d["FLIGHT_ID_mvt"].notna().values
    lobt = d["LOBT_flt"].where(record, d["lobt_arr"])
    return pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].values, "record": record, "linked": ~record & lobt.notna().values,
                         "ml": seconds(d["MVT_TIME_UTC_mvt"], lobt), "sl": seconds(d["SCHED_TIME_UTC_mvt"], lobt)})


def project(pred: np.ndarray, t: pd.DataFrame) -> np.ndarray:
    """The served window rule on record rows; the arrival-side `LOBT` of a row without a record does not bound it."""
    return window(pred, np.where(t["record"].values, t["ml"].values, np.nan))


def outside_lirf() -> pd.DataFrame:
    """Hold-out rows outside LIRF with the label, the airport, the label class and the v115 served value `final`."""
    rows = pd.read_parquet(C.MODELS / "blend_v97_holdout.parquet")
    mlp, tabm = (pd.read_parquet(out_paths(c)[0]) for c in ("v97", C.V115_TABM))
    if not np.array_equal(rows["MVT_ID_mvt"].values, tabm["MVT_ID_mvt"].values, equal_nan=True):
        raise AssertionError("The TabM hold-out file and the v97 blend differ in row order. Rerun `tabm_member measure`.")
    w = C.V115_SHARES
    rows["base"] = w[0] * rows["base"] + w[1] * mlp["pred"].values + w[2] * tabm["pred"].values
    dep = load_dep(sorted(C.TRAIN_DIR.glob("training_2025-0[17]-*.parquet")), ["CALLSIGN_flt"])
    f = sp.frame(rows.dropna(subset=["MVT_ID_mvt"]), (NM_TRAIN, sp.OPDI_TRAIN, C.ADSB_DIR / "v115" / "adsb_clock_train.parquet"), dep)
    s, _ = sp.crossfit(f)
    final = f.set_index("MVT_ID_mvt")["base"].copy()
    final.loc[s["MVT_ID_mvt"].values] = np.fmax(s["stack"].values, sp.stand_bound(s))
    sd = seconds(f["MVT_TIME_UTC_mvt"], dep.set_index("MVT_ID_mvt").loc[f["MVT_ID_mvt"].values, "SCHED_TIME_UTC_mvt"].reset_index(drop=True))
    apt = f["apt"].astype(str).values
    return pd.DataFrame({"MVT_ID_mvt": f["MVT_ID_mvt"].values, "y": f["y"].values, "apt": apt,
                         "cls": classify(f["y"].values, sd, apt), "final": final.loc[f["MVT_ID_mvt"].values].values})


def lirf_head() -> pd.DataFrame:
    """LIRF hold-out rows with the gate `p`, `R_norm`, `sd`, the served value `pred` and the rule rows `fixed`."""
    dep, _ = gate_frame()
    p = nested_map(pd.read_parquet(GATE_OOF)["lgb"].values, dep["is_fb"].values, dep["month"])
    hold = dep["month"].isin(HOLD).values
    d = dep[hold].assign(p=p[hold]).merge(pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *TEMPO_COLS]),
                                          on="MVT_ID_mvt", how="left", validate="1:1")
    d["r"], sd = r_norm_v41(d), d["sched_delay"].astype(float).values
    pred = np.where(np.isnan(sd), d["r"], d["p"] * sd + (1 - d["p"]) * d["r"])
    band = json.loads(BAND_TABLE.read_text())
    step_a, cell = apply_stepA_v22(d["r"].values, sd, d["ADEP_mvt"].astype(str).values, d["FLIGHT_ID_mvt"].values, band)
    pred = adjust(d, np.where(cell, step_a, pred), band)
    ity = ~cell & (sd > C.ITY_SD_THRESHOLD)
    pred = np.where(ity, C.P24_ITY * (86400 + C.NORMAL_MEAN_LIRF) + (1 - C.P24_ITY) * sd, pred)
    return pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].values, "y": d["TAXITIME_SEC_mvt"].astype(float).values, "sd": sd,
                         "r": d["r"].values, "p": d["p"].values, "pred": np.clip(pred, 0, None),
                         "fixed": cell | ity | np.isnan(sd)})


def lirf_arms(h: pd.DataFrame) -> dict:
    """Served LIRF values under each rule: the gate set to 0 where a schedule copy is impossible, and the window."""
    far = np.abs(h["sl"].values) > C.LOBT_WINDOW_S
    no_copy = {"copy_record": h["record"].values & far, "copy_linked": h["linked"].values & ~far}
    no_copy["copy_both"] = no_copy["copy_record"] | no_copy["copy_linked"]
    out = {}
    for k, m in no_copy.items():
        q = np.where(m, 0.0, h["p"].values)
        out[k] = np.where(h["fixed"], h["pred"], np.clip(q * h["sd"] + (1 - q) * h["r"], 0, None))
    out["window"] = project(h["pred"].values, h)
    out["all"] = project(out["copy_both"], h)
    return out


def price(new: np.ndarray, old: np.ndarray, y: np.ndarray, by: np.ndarray) -> dict:
    """MSE change on the hold-out scale: all rows, the body (`y <= BODY_MAX_S`), the tail, and per group."""
    d = pd.Series(((new - y) ** 2 - (old - y) ** 2) / C.HOLDOUT_ROWS)
    body = y <= C.BODY_MAX_S
    return {"rows_moved": int((new != old).sum()), "mse": round(float(d.sum()), 1), "body": round(float(d[body].sum()), 1),
            "tail": round(float(d[~body].sum()), 1), "by_group": d.groupby(by).sum().round(1).to_dict()}


def served(name: str, t: pd.DataFrame) -> dict:
    """Rows of a served ranking file that each rule moves, and the live gain the window guarantees."""
    s = pd.read_parquet(C.ROOT / "submission" / name).merge(t, on="MVT_ID_mvt", how="left", validate="1:1")
    s = s.merge(load_dep([C.RANK])[["MVT_ID_mvt", "ADEP_mvt"]], on="MVT_ID_mvt", how="left", validate="1:1")
    pred = s["TAXITIME_SEC_mvt"].values.astype(float)
    proj = project(pred, s)
    lirf, far = (s["ADEP_mvt"] == "LIRF").values, np.abs(s["sl"].values) > C.LOBT_WINDOW_S
    return {"file": name, "window_rows": int((proj != pred).sum()),
            "window_guaranteed_gain_mse": round(float(((proj - pred) ** 2).sum() / C.RANK_ROWS), 1),
            "lirf_copy_record_rows": int((lirf & s["record"].values & far).sum()),
            "lirf_copy_linked_rows": int((lirf & s["linked"].values & ~far).sum())}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--served", default="kind-mango_v116.parquet", help="Served ranking file in submission/.")
    args = ap.parse_args()
    t25 = lobt_terms(sorted(C.TRAIN_DIR.glob("training_2025-*.parquet")))
    o = outside_lirf().merge(t25, on="MVT_ID_mvt", how="left", validate="1:1")
    h = lirf_head().merge(t25, on="MVT_ID_mvt", how="left", validate="1:1")
    arms = lirf_arms(h)
    report = {"outside_lirf_window": price(project(o["final"].values, o), o["final"].values, o["y"].values, o["apt"].values),
              "lirf": {k: price(v, h["pred"].values, h["y"].values, np.where(h["record"], "record", "no_record"))
                       for k, v in arms.items()},
              "served": served(args.served, lobt_terms([C.RANK]))}
    REPORT.write_text(json.dumps(report, indent=1))
    log.info("outside LIRF %.0f, LIRF all rules %.0f hold-out MSE; %s -> %s", report["outside_lirf_window"]["mse"],
             report["lirf"]["all"]["mse"], report["served"], REPORT.name)


if __name__ == "__main__":
    main()
