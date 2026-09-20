"""MD4: leave-one-month-out operator encoders + base retrain (v60).

The v58 experiment showed that MB8 (all-months in-sample encoders)
amplifies the L1 leak the audit named and regresses +1.29 s live. MD4 is
the correct fix: each row's encoded value uses the median/count computed
on rows in ALL OTHER months. At serving on 2026, the full-year encoder
map applies unchanged.

For each key K in the deployed KEYS list (op_apt, op_apt_rwy,
op_apt_hbin, op, op_apt_stand):

  - For each fold month m in 1..12:
    build agg_m = fit_encoders(train_slice_where_month != m)[K]
    for training rows where month == m, look up their encoded value from
    agg_m.
  - Save the full-year encoder (fit on all 12 months) as the SERVING
    encoder in lgbm_r_all_v60.encoders.pkl.

The training-time openc_* columns come from the fold-specific maps. The
serving-time openc_* columns come from the full-year map.
"""
from __future__ import annotations

import glob
import json
import logging
import math
import pickle
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features_operator import KEYS, MIN_COUNT, _hour_bin, fit_encoders
from train_lgbm_v21 import CAT_COLS
from train_r_all_v26 import DEFAULT_SEEDS

from src_v3 import config as C
from src_v3.frames import (H1, H1_IDS, V2_TRAIN, TEMPO_P2575,
                           V40_TEMPO, V44_QUARTILES, V45_PLAN, v47_feature_names)

TUNED = C.ROOT / "models" / "tune_lgbm_v43.tuned_params.json"
V46_REPORT = C.ROOT / "models" / "lgbm_r_all_v46.holdout.json"
PLAN_TRAIN_V57 = C.ROOT / "models" / "plan_taxi_res_train_v57.parquet"
log = logging.getLogger(__name__)


def _load_raw_all_months() -> pd.DataFrame:
    """All 2025 DEP rows at target airports for encoder fitting."""
    parts = []
    for f in sorted(glob.glob(str(C.TRAIN_DIR / "*.parquet"))):
        t = pd.read_parquet(f, columns=[
            "MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt", "ADEP_mvt", "RUNWAY_mvt",
            "STAND_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt",
            "TAXITIME_SEC_mvt"])
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(C.TARGETS)]
        t = t[t["TAXITIME_SEC_mvt"].astype(float) > 0]
        ts = pd.to_datetime(t["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
        t["hour"] = ts.dt.hour
        t["month"] = ts.dt.month
        parts.append(t[["MVT_ID_mvt", "AIRCRAFT_OPERATOR_flt", "ADEP_mvt",
                        "RUNWAY_mvt", "STAND_mvt", "hour", "month",
                        "TAXITIME_SEC_mvt"]])
    return pd.concat(parts, ignore_index=True)


def _oof_encoded_columns(train: pd.DataFrame) -> pd.DataFrame:
    """For each row, compute the OOF-encoded columns using per-month fits.

    Returns a DataFrame with one row per input row and the openc_* columns
    that features_operator.apply_encoders would produce, but each row's
    values come from an encoder fit that EXCLUDES the row's own month.
    """
    train = train.copy()
    train["_hour_bin"] = _hour_bin(train["hour"])
    train["_row_idx"] = np.arange(len(train))

    # Precompute per-fold-month encoders for each key
    log.info("computing per-month encoder fits (12 folds x %d keys)", len(KEYS))
    fold_encoders: dict[int, dict] = {}
    for m in range(1, 13):
        other = train[train["month"] != m]
        # Reuse the audit's fit_encoders which expects 'hour' and computes
        # _hour_bin internally.
        fenc = fit_encoders(other)
        fold_encoders[m] = fenc

    # For each key, look up encoded columns per row from that row's fold
    out = pd.DataFrame({"MVT_ID_mvt": train["MVT_ID_mvt"].values,
                        "_row_idx": train["_row_idx"].values})
    for name, keys in KEYS:
        med_arr = np.full(len(train), np.nan)
        cnt_arr = np.full(len(train), np.nan)
        std_arr = np.full(len(train), np.nan)
        for m in range(1, 13):
            mask = (train["month"].values == m)
            if not mask.any():
                continue
            keys_actual, tbl = fold_encoders[m][name]
            sub = train.loc[mask, keys_actual]
            merged = sub.merge(tbl, on=keys_actual, how="left")
            med_arr[mask] = merged[f"openc_{name}_median"].values
            cnt_arr[mask] = merged[f"openc_{name}_count"].values
            std_arr[mask] = merged[f"openc_{name}_std"].values
        out[f"openc_{name}_median"] = med_arr
        out[f"openc_{name}_count"] = cnt_arr
        out[f"openc_{name}_std"] = std_arr
        log.info("  key %s: coverage %.3f",
                 name, np.isfinite(med_arr).mean())

    # Coalesce openc_taxi_median (finest to coarsest)
    coarsen_order = ["op_apt_stand", "op_apt_rwy", "op_apt_hbin", "op_apt", "op"]
    out["openc_taxi_median"] = np.nan
    for name in coarsen_order:
        col = f"openc_{name}_median"
        if col in out.columns:
            out["openc_taxi_median"] = out["openc_taxi_median"].fillna(out[col])
    # Fall back to per-fold global (approx: use full-year median)
    global_med = float(train["TAXITIME_SEC_mvt"].median())
    out["openc_taxi_median"] = out["openc_taxi_median"].fillna(global_med)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(TUNED) as f:
        tuned = json.load(f)["tuned"]
    with open(V46_REPORT) as f:
        v46_iters = json.load(f)["v46"]["best_iters"]

    log.info("=== step 1: load raw all-months DEP rows ===")
    t0 = time.time()
    raw = _load_raw_all_months()
    log.info("raw %d rows in %.0fs", len(raw), time.time() - t0)

    log.info("=== step 2: fit full-year encoders (for serving) ===")
    serving_encoders = fit_encoders(raw)
    with open(C.ROOT / "models" / "lgbm_r_all_v60.encoders.pkl", "wb") as f:
        pickle.dump(serving_encoders, f)
    log.info("wrote v60 serving encoders")

    log.info("=== step 3: compute OOF encoder columns for training rows ===")
    t0 = time.time()
    oof = _oof_encoded_columns(raw)
    log.info("OOF encoder columns computed in %.0fs", time.time() - t0)

    log.info("=== step 4: build v60 frame ===")
    dep = pd.read_parquet(H1)
    dep["MVT_ID_mvt"] = pd.read_parquet(H1_IDS)["MVT_ID_mvt"].values
    # Drop the old openc_* columns before merging in the OOF versions by
    # MVT_ID_mvt. Merging (not positional) protects against row order drift.
    old_openc = [c for c in dep.columns if c.startswith("openc_")]
    dep = dep.drop(columns=old_openc)
    oof_cols = [c for c in oof.columns if c.startswith("openc_")]
    # Drop NaN MVT_ID_mvt rows from oof before merging; the H1 cache has
    # 169 rows with a null id (raw-file NaNs) that can never match. Those
    # rows will inherit NaN openc_* which LightGBM's use_missing handles.
    oof_ok = oof[oof["MVT_ID_mvt"].notna()].drop_duplicates(subset=["MVT_ID_mvt"])
    dep = dep.merge(oof_ok[["MVT_ID_mvt"] + oof_cols],
                    on="MVT_ID_mvt", how="left")
    tempo = pd.read_parquet(V2_TRAIN, columns=["MVT_ID_mvt", *V40_TEMPO])
    dep = dep.merge(tempo, on="MVT_ID_mvt", how="left")
    quart = pd.read_parquet(TEMPO_P2575, columns=["MVT_ID_mvt", *V44_QUARTILES])
    dep = dep.merge(quart, on="MVT_ID_mvt", how="left")
    plan = pd.read_parquet(PLAN_TRAIN_V57, columns=["MVT_ID_mvt", *V45_PLAN])
    dep = dep.merge(plan, on="MVT_ID_mvt", how="left")

    log.info("=== step 5: MB3 filter + retrain ===")
    adep = dep["ADEP_mvt"].astype(str)
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    keep = (adep != "LIRF") & (y > 0) & (y <= 80_000)
    dep = dep[keep].reset_index(drop=True)
    for c in CAT_COLS:
        if c in dep.columns and str(dep[c].dtype) == "object":
            dep[c] = dep[c].astype("category")
    log.info("MB3 filter: %d rows", len(dep))

    feat = v47_feature_names()
    n_v46_train = int(0.88 * 10 / 12 * 2_084_659)
    v60_scale = len(dep) / n_v46_train
    scaled = [int(math.ceil(i * v60_scale)) for i in v46_iters]
    log.info("v46 iters %s -> v60 %s (x %.3f)", v46_iters, scaled, v60_scale)

    dt = lgb.Dataset(dep[feat], label=dep["TAXITIME_SEC_mvt"].values,
                     categorical_feature=CAT_COLS,
                     params={"linear_tree": True, "feature_pre_filter": False})
    for seed, n_iter in zip(DEFAULT_SEEDS, scaled):
        p = {**tuned, "seed": seed, "bagging_seed": seed,
             "feature_fraction_seed": seed, "linear_tree": True,
             "bagging_freq": 5, "verbosity": -1,
             "num_threads": C.NUM_THREADS,
             "deterministic": C.DETERMINISTIC}
        t0 = time.time()
        b = lgb.train(p, dt, num_boost_round=n_iter)
        b.save_model(str(C.ROOT / "models" / f"lgbm_r_all_v60_s{seed}.txt"))
        log.info("v60 seed %d %d rounds in %.0fs", seed, n_iter, time.time() - t0)

    with open(C.ROOT / "models" / "lgbm_r_all_v60.features.txt", "w") as f:
        f.write("\n".join(feat))
    (C.ROOT / "models" / "lgbm_r_all_v60.meta.json").write_text(json.dumps({
        "tuned": tuned, "v46_iters": v46_iters, "scaled_iters": scaled,
        "scale_factor": v60_scale, "n_served": len(dep),
        "changed": "MD4 leave-one-month-out operator encoders; serving uses full-year map",
    }, indent=1))
    log.info("wrote v60 base members")


if __name__ == "__main__":
    main()
