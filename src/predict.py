"""Apply lgbm_v4 to ranking.parquet and produce a validated submission file.

Reuses the exact same feature pipeline as train_lgbm_v4.py. Validates that
the output id set matches submitting.parquet exactly (rejection would waste
the entry).
"""
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol, eurocontrol_numeric_cols
from features_operator import apply_encoders, operator_numeric_cols
from features_taxi_distance import add_taxi_distance, TAXI_NUM_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")
MODEL = os.path.join(ROOT, "models", "lgbm_v4.txt")
FEATS = os.path.join(ROOT, "models", "lgbm_v4.features.txt")
ENC = os.path.join(ROOT, "models", "lgbm_v4.encoders.pkl")

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]


def load_training_categories() -> dict:
    """Union of category values across training parquets, DEP + target ICAOs."""
    import glob
    tdir = os.path.join(ROOT, "training")
    seen = {c: set() for c in CAT_COLS}
    for f in sorted(glob.glob(os.path.join(tdir, "*.parquet"))):
        t = pd.read_parquet(f, columns=list({*CAT_COLS, "PHASE_mvt"}))
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(TARGET_ICAOS)]
        for c in CAT_COLS:
            seen[c].update(t[c].dropna().unique())
    return {c: pd.Index(list(v)) for c, v in seen.items()}


def main(out_name: str = "kind-mango_v1.parquet"):
    t0 = time.time()
    print("Loading ranking + template...")
    rank = pd.read_parquet(RANK)
    tmpl = pd.read_parquet(SUB_TMPL)
    print(f"  ranking rows: {len(rank):,}   template rows: {len(tmpl):,}")

    rank["mvt_ts"] = pd.to_datetime(rank["MVT_TIME_UTC_mvt"], errors="coerce")
    rank["sched_ts"] = pd.to_datetime(rank["SCHED_TIME_UTC_mvt"], errors="coerce")

    # DEP subset to predict (10 target ICAOs, must equal template id set)
    dep = rank[(rank["PHASE_mvt"] == "DEP") & rank["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)

    t = time.time()
    print("Adding weather...")
    dep = add_weather(dep)
    print(f"  {time.time()-t:.1f}s")

    t = time.time()
    print("Adding congestion (using full ranking DEP+ARR)...")
    dep = add_congestion(dep, rank[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    print(f"  {time.time()-t:.1f}s")

    t = time.time()
    print("Adding Eurocontrol daily features...")
    dep = add_eurocontrol(dep)
    print(f"  {time.time()-t:.1f}s")

    t = time.time()
    print("Adding operator encoders...")
    with open(ENC, "rb") as f:
        encoders = pickle.load(f)
    dep = apply_encoders(dep, encoders)
    print(f"  {time.time()-t:.1f}s")

    t = time.time()
    print("Adding taxi distance...")
    dep = add_taxi_distance(dep)
    print(f"  {time.time()-t:.1f}s")

    # Encode categoricals against training-side vocabularies
    print("Encoding categoricals from training vocab...")
    train_cats = load_training_categories()
    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=train_cats[c])

    # Load model + feature list
    booster = lgb.Booster(model_file=MODEL)
    with open(FEATS) as f:
        feat = f.read().splitlines()
    missing = [c for c in feat if c not in dep.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    print(f"Predicting on {len(dep):,} rows...")
    preds = booster.predict(dep[feat])
    dep["TAXITIME_SEC_mvt"] = preds

    # Merge into template exactly
    out = tmpl[["MVT_ID_mvt"]].merge(
        dep[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]],
        on="MVT_ID_mvt", how="left"
    )

    # Validation
    print("\n=== VALIDATION ===")
    assert len(out) == len(tmpl), f"row count mismatch: {len(out)} vs {len(tmpl)}"
    assert set(out["MVT_ID_mvt"]) == set(tmpl["MVT_ID_mvt"]), "id-set mismatch"
    print(f"[ok] Row count = {len(out):,} matches template")
    print(f"[ok] Id set matches exactly")
    n_nan = out["TAXITIME_SEC_mvt"].isna().sum()
    if n_nan:
        median = float(np.nanmedian(out["TAXITIME_SEC_mvt"]))
        print(f"[warn] {n_nan} NaN predictions, filling with global median {median:.0f}s")
        out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].fillna(median)
    n_neg = (out["TAXITIME_SEC_mvt"] < 60).sum()
    n_huge = (out["TAXITIME_SEC_mvt"] > 7200).sum()
    print(f"[info] predictions < 60s : {n_neg}   > 7200s: {n_huge}")
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].clip(60, 7200)
    print(f"[ok] Values clipped to [60, 7200]")

    # Summary stats
    print(f"\nPrediction summary:")
    print(out["TAXITIME_SEC_mvt"].describe(percentiles=[.05, .5, .95]).round(1).to_string())

    out_path = os.path.join(ROOT, "submission", out_name)
    out.to_parquet(out_path)
    print(f"\nSaved -> {out_path}   ({os.path.getsize(out_path)/1024:.1f} KB)")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v1.parquet"
    main(name)
