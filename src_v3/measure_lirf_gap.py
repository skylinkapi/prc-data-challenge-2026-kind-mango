"""Where the LIRF error sits, and which observable can close it.

1. Bands: the v41 head on the LIRF hold-out rows of January and July 2025
   (`R_norm` and gate out of sample; band table and rate maps in-sample,
   eleventh pass F5), split by schedule-delay band and flight-record status,
   against a perfect gate (`p` = 1 on fallback rows, 0 elsewhere; Step A
   and ITY340 rows keep their value). The difference is the ceiling of any
   gate change. The 2026 row counts per cell come with it.
2. Proxy: per airline-month cell of 2025 at each airport, the departure
   fallback rate against the arrival rate of in-block within
   `FB_TOL_DEFAULT` of the schedule, across cells and within each airline
   over months; at LIRF also per airline-day on late departures. The
   ranking file keeps arrival block times, so these rates exist for 2026.
3. Context: LIRF ADS-B coverage, and departures whose block equals `EOBT_1`.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import eval_v33_holdout as ev
from predict_v23 import add_rate_features_scoring
from src_v3 import config as C
from src_v3.build_adsb_clock import OUT_RANK as ADSB_RANK
from src_v3.build_adsb_clock import OUT_TRAIN as ADSB_TRAIN
from src_v3.build_nm_clocks_v73 import load_dep

REPORT = C.MODELS / "lirf_gap.holdout.json"
RAW = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_ID_mvt",
       "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "EOBT_1_flt"]
SPEC = ev.STACKS["v41"]
log = logging.getLogger(__name__)


def band_of(sd: np.ndarray) -> np.ndarray:
    return pd.cut(sd, [-np.inf, *C.LIRF_SD_BANDS_S, np.inf]).astype(str)


def lirf_holdout() -> pd.DataFrame:
    """LIRF hold-out rows with the v41 `R_norm` columns; the src_v2 frame supplies the columns the cache lacks."""
    d = ev.lirf_frame()
    d = d[d["month"].isin(ev.HOLDOUT_MONTHS)].reset_index(drop=True)
    missing = [c for c in ev.read_list(SPEC["r_norm_features"]) if c not in d.columns]
    if missing:
        d = d.merge(pd.read_parquet(ev.V2_FRAME, columns=["MVT_ID_mvt", *missing]), on="MVT_ID_mvt", how="left", validate="1:1")
    return d


def head_parts(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`R_norm` member mean and calibrated gate of the v41 head, as `eval_v33_holdout.lirf_head` computes them."""
    models = Path(ev.MODELS)
    with open(models / "lirf_regime_v23.rate_maps.pkl", "rb") as f:
        bundle = pickle.load(f)
    d = add_rate_features_scoring(d.copy(), bundle["maps"], bundle["base_rate"])
    for c in ev.CAT_COLS:
        d[c] = pd.Categorical(d[c], categories=pd.Index(d[c].dropna().unique()))
    feat = ev.read_list(SPEC["r_norm_features"])
    r_norm = np.mean([np.clip(lgb.Booster(model_file=str(models / m)).predict(d[feat]), 0, None)
                      for m in SPEC["r_norm_files"]], axis=0)
    with open(models / "lirf_regime_v23.isotonic.pkl", "rb") as f:
        iso = pickle.load(f)
    gate = lgb.Booster(model_file=str(models / "lgbm_p_fb_lirf_v23.txt"))
    return r_norm, iso.transform(gate.predict(d[ev.read_list("lirf_regime_v23.features.txt")]))


def bands(mv26: pd.DataFrame) -> dict:
    """LIRF hold-out MSE by delay band and record status, served and with a perfect gate, with the 2026 rows per cell."""
    d = lirf_holdout()
    pred = ev.lirf_head(d.copy(), SPEC["r_norm_files"], ev.read_list(SPEC["r_norm_features"]), SPEC["r_norm_clip"])
    r_norm, p = head_parts(d)
    y, sd = d["TAXITIME_SEC_mvt"].values.astype(float), d["sched_delay"].values.astype(float)
    rule = ~np.isclose(pred, np.clip(np.where(np.isnan(sd), r_norm, p * sd + (1 - p) * r_norm), 0, None))
    fb = np.abs(y - sd) <= C.FB_TOL_DEFAULT
    perfect = np.where(rule, pred, np.where(fb, sd, r_norm))
    t = pd.DataFrame({"band": band_of(sd), "record": d["FLIGHT_ID_mvt"].notna().values, "fb": fb, "p": p,
                      "mse": (pred - y) ** 2 / C.HOLDOUT_ROWS, "mse_perfect_gate": (perfect - y) ** 2 / C.HOLDOUT_ROWS})
    g = t.groupby(["band", "record"]).agg(rows=("fb", "size"), fallback_rate=("fb", "mean"), gate_mean=("p", "mean"),
                                         mse=("mse", "sum"), mse_perfect_gate=("mse_perfect_gate", "sum"))
    dep26 = mv26[(mv26["phase"] == "DEP") & (mv26["apt"] == "LIRF")]
    g["rows_2026"] = dep26.groupby([band_of(dep26["sd"].values), dep26["record"].values]).size()
    genuine = ~fb & (y < C.Y_24H_MIN)
    return {"rows": len(t), "mse": round(float(t["mse"].sum())), "mse_perfect_gate": round(float(t["mse_perfect_gate"].sum())),
            "rule_rows": int(rule.sum()), "rule_mse": round(float(t.loc[rule, "mse"].sum())),
            "r_norm_rmse_genuine": round(float(np.sqrt(((r_norm - y)[genuine] ** 2).mean())), 1),
            "by_band": g.round(3).reset_index().to_dict("records")}


def movements(paths: list) -> pd.DataFrame:
    """Movements with airport, airline prefix, operator, day, delay against schedule and the artefact flags."""
    d = pd.concat([pd.read_parquet(p, columns=RAW) for p in paths], ignore_index=True)

    def near(a: str, b: str) -> np.ndarray:
        return ((d[a] - d[b]).dt.total_seconds().abs() <= C.FB_TOL_DEFAULT).values

    return pd.DataFrame({"phase": d["PHASE_mvt"].values, "apt": np.where(d["PHASE_mvt"] == "DEP", d["ADEP_mvt"], d["ADES_mvt"]),
                         "pre": d["FLIGHT_mvt"].astype(str).str[:3].values, "op": d["AIRCRAFT_OPERATOR_flt"].astype(str).values,
                         "m": d["MVT_TIME_UTC_mvt"].dt.month.values, "day": d["MVT_TIME_UTC_mvt"].dt.floor("D").values,
                         "sd": (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds().values,
                         "mvt_eobt1": (d["MVT_TIME_UTC_mvt"] - d["EOBT_1_flt"]).dt.total_seconds().values,
                         "record": d["FLIGHT_ID_mvt"].notna().values,
                         "art": near("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"), "eobt_copy": near("BLOCK_TIME_UTC_mvt", "EOBT_1_flt")})


def proxy(mv: pd.DataFrame, apt: str) -> dict:
    """Correlation of the departure fallback rate and the arrival artefact rate over the airline-month cells of `apt`."""
    dep, arr = mv[(mv["phase"] == "DEP") & (mv["apt"] == apt)], mv[(mv["phase"] == "ARR") & (mv["apt"] == apt)]
    out = {}
    for key in ("pre", "op"):
        g = pd.DataFrame({"dep_n": dep.groupby([key, "m"]).size(), "dep": dep.groupby([key, "m"])["art"].mean(),
                          "arr_n": arr.groupby([key, "m"]).size(), "arr": arr.groupby([key, "m"])["art"].mean()}).dropna()
        g = g[(g["dep_n"] >= C.ARTEFACT_MIN_CELL_ROWS) & (g["arr_n"] >= C.ARTEFACT_MIN_CELL_ROWS)]
        within = g.groupby(level=0)[["dep", "arr"]].transform(lambda v: v - v.mean())
        out[key] = {"cells": len(g), "corr_across": round(float(np.corrcoef(g["dep"], g["arr"])[0, 1]), 2),
                    "corr_within_airline": round(float(np.corrcoef(within["dep"], within["arr"])[0, 1]), 2)}
    return out


def day_buckets(mv: pd.DataFrame) -> dict:
    """Late LIRF departures: fallback rate per bucket of the same airline's same-day arrival artefact share, and AUCs."""
    lirf = mv[mv["apt"] == "LIRF"]
    arr, dep = lirf[lirf["phase"] == "ARR"], lirf[lirf["phase"] == "DEP"]
    k, base = C.ARTEFACT_SMOOTH_K, arr["art"].mean()
    s = arr.groupby(["pre", "day"])["art"].agg(["sum", "size"])
    dep = dep.merge(((s["sum"] + k * base) / (s["size"] + k)).rename("share").reset_index(), on=["pre", "day"], how="left")
    tot, mon = dep.groupby("pre")["art"].agg(["sum", "size"]), dep.groupby(["pre", "m"])["art"].agg(["sum", "size"])
    j = dep[["pre", "m"]].join(tot, on="pre").join(mon, on=["pre", "m"], rsuffix="_m")
    dep["static"] = (j["sum"] - j["sum_m"] + k * dep["art"].mean()) / (j["size"] - j["size_m"] + k)   # leave own month out
    late = dep[dep["sd"] > C.LATE_SD_S].assign(share=lambda x: x["share"].fillna(base))
    b = pd.cut(late["share"], [-0.01, *C.ARTEFACT_SHARE_EDGES, 1.0]).astype(str)
    return {"rows": len(late), "fallback_rate": round(float(late["art"].mean()), 3),
            "auc_static_airline_rate": round(float(roc_auc_score(late["art"], late["static"])), 3),
            "auc_airline_day_share": round(float(roc_auc_score(late["art"], late["share"])), 3),
            "by_bucket": late.groupby(b)["art"].agg(rows="size", fallback_rate="mean").round(3).to_dict("index")}


def shift_2026(mv25: pd.DataFrame, mv26: pd.DataFrame) -> dict:
    """LIRF arrival artefact rate per month of 2025 and 2026, and per airline prefix with the most 2026 departures."""
    sel = {n: m[m["apt"] == "LIRF"] for n, m in (("25", mv25[mv25["m"].isin(ev.HOLDOUT_MONTHS)]), ("26", mv26))}
    a25, a26 = (sel[n][sel[n]["phase"] == "ARR"] for n in ("25", "26"))
    d25, d26 = (sel[n][sel[n]["phase"] == "DEP"] for n in ("25", "26"))

    def rate(frame: pd.DataFrame, pre: str, month: int = 0) -> float:
        return round(float(frame.loc[(frame["pre"] == pre) & ((frame["m"] == month) | (month == 0)), "art"].mean()), 3)

    return {"arr_by_month_2025": a25.groupby("m")["art"].mean().round(3).to_dict(),
            "arr_by_month_2026": a26.groupby("m")["art"].mean().round(3).to_dict(),
            "by_prefix": {p: {"departures_2026": int((d26["pre"] == p).sum()), "dep_fallback_2025": rate(d25, p),
                              "arr_2025_jan_jul": rate(a25, p), "arr_2026_jan": rate(a26, p, 1), "arr_2026_jul": rate(a26, p, 7)}
                          for p in d26["pre"].value_counts().head(10).index}}


def eobt_copies(mv: pd.DataFrame) -> dict:
    """Genuine departures (block off the schedule) whose block equals `EOBT_1`, per airport and, at LIRF, per planned-delay band."""
    g = mv[(mv["phase"] == "DEP") & ~mv["art"] & mv["mvt_eobt1"].notna()]
    lirf = g[g["apt"] == "LIRF"]
    b = pd.cut(lirf["mvt_eobt1"], [-np.inf, 900, 1800, 3600, 7200, np.inf]).astype(str)
    return {"share_by_airport": g.groupby("apt")["eobt_copy"].mean().round(3).to_dict(),
            "lirf_by_mvt_eobt1": lirf.groupby(b)["eobt_copy"].agg(rows="size", share="mean").round(3).to_dict("index")}


def lirf_adsb() -> dict:
    """LIRF departures with an ADS-B ground segment, and with the first point within 300 m of the assigned stand."""
    out = {}
    for name, paths, table in (("2025", sorted(C.TRAIN_DIR.glob("training_*.parquet")), ADSB_TRAIN), ("2026", [C.RANK], ADSB_RANK)):
        d = load_dep(paths)[["MVT_ID_mvt", "ADEP_mvt"]].merge(pd.read_parquet(table), on="MVT_ID_mvt", validate="1:1")
        d = d[d["ADEP_mvt"] == "LIRF"]
        seg = d["adsb_seg"].notna()
        out[name] = {"departures": len(d), "with_segment": int(seg.sum()),
                     "first_point_within_300m_of_stand": round(float((d.loc[seg, "adsb_own_d0"] <= 300).mean()), 3)}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mv25, mv26 = movements(sorted(C.TRAIN_DIR.glob("training_2025-*.parquet"))), movements([C.RANK])
    report = {"bands": bands(mv26), "proxy": {a: proxy(mv25, a) for a in C.TARGETS}, "lirf_day_buckets": day_buckets(mv25),
              "lirf_shift_2026": shift_2026(mv25, mv26), "eobt_copies": eobt_copies(mv25), "lirf_adsb": lirf_adsb()}
    REPORT.write_text(json.dumps(report, indent=1, default=str))
    b = report["bands"]
    log.info("LIRF hold-out MSE %d, perfect gate %d; proxy at LIRF %s -> %s", b["mse"], b["mse_perfect_gate"],
             report["proxy"]["LIRF"], REPORT.name)


if __name__ == "__main__":
    main()
