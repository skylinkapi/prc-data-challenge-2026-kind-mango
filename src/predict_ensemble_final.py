"""Apply v5+v7+v9+v10 ensemble to ranking.parquet.
Weights from ensemble_final.py grid search: v10=0.40, v9=0.30, v7=0.25, v5=0.05, v6=0.00.
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
from features_eurocontrol import add_eurocontrol
from features_operator import apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")
MODELS_DIR = os.path.join(ROOT, "models")

WEIGHTS = {"v5": 0.05, "v7": 0.25, "v9": 0.30, "v10": 0.40}

CAT_COLS = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "AIRCRAFT_OPERATOR_flt", "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt"]


def load_training_categories():
    import glob
    tdir = os.path.join(ROOT, "training")
    seen = {c: set() for c in CAT_COLS}
    for f in sorted(glob.glob(os.path.join(tdir, "*.parquet"))):
        t = pd.read_parquet(f, columns=list({*CAT_COLS, "PHASE_mvt"}))
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(TARGET_ICAOS)]
        for c in CAT_COLS:
            seen[c].update(t[c].dropna().unique())
    return {c: pd.Index(list(v)) for c, v in seen.items()}


def main(out_name: str = "kind-mango_v3.parquet"):
    t0 = time.time()
    print("Loading ranking + template...")
    rank = pd.read_parquet(RANK)
    tmpl = pd.read_parquet(SUB_TMPL)
    print(f"  ranking rows: {len(rank):,}   template rows: {len(tmpl):,}")

    rank["mvt_ts"] = pd.to_datetime(rank["MVT_TIME_UTC_mvt"], errors="coerce")
    rank["sched_ts"] = pd.to_datetime(rank["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = rank[(rank["PHASE_mvt"] == "DEP") & rank["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)

    print("Building features...")
    dep = add_weather(dep)
    dep = add_congestion(dep, rank[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    with open(os.path.join(MODELS_DIR, "lgbm_v5.encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    dep = apply_encoders(dep, encoders)
    dep = add_taxi_distance(dep)

    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, rank[["ADEP_mvt", "PHASE_mvt", "mvt_ts", "sched_ts", "RUNWAY_mvt"]],
                       daily_ec=ec_daily)
    dep = add_osm_path(dep)

    train_cats = load_training_categories()
    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=train_cats[c])

    preds = {}
    for v, w in WEIGHTS.items():
        b = lgb.Booster(model_file=os.path.join(MODELS_DIR, f"lgbm_{v}.txt"))
        with open(os.path.join(MODELS_DIR, f"lgbm_{v}.features.txt")) as f:
            feat = f.read().splitlines()
        missing = [c for c in feat if c not in dep.columns]
        if missing:
            raise RuntimeError(f"Missing cols for {v}: {missing}")
        preds[v] = b.predict(dep[feat])
        print(f"  {v} predicted ({w:.2f} weight)")

    ens = sum(WEIGHTS[v] * preds[v] for v in WEIGHTS)
    dep["TAXITIME_SEC_mvt"] = ens

    out = tmpl[["MVT_ID_mvt"]].merge(
        dep[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]],
        on="MVT_ID_mvt", how="left"
    )

    assert len(out) == len(tmpl), f"row count mismatch"
    assert set(out["MVT_ID_mvt"]) == set(tmpl["MVT_ID_mvt"]), "id set mismatch"
    n_nan = out["TAXITIME_SEC_mvt"].isna().sum()
    if n_nan:
        med = float(np.nanmedian(out["TAXITIME_SEC_mvt"]))
        print(f"[warn] {n_nan} NaN, filling with median {med:.0f}s")
        out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].fillna(med)
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].clip(60, 7200)

    print(f"\n[ok] rows={len(out):,}, id set matches, dtypes {list(out.dtypes)}")
    print(f"Summary:\n{out['TAXITIME_SEC_mvt'].describe(percentiles=[.05,.5,.95]).round(1).to_string()}")

    path = os.path.join(ROOT, "submission", out_name)
    out.to_parquet(path)
    print(f"\nSaved -> {path}   ({os.path.getsize(path)/1024:.1f} KB)")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v3.parquet"
    main(name)
