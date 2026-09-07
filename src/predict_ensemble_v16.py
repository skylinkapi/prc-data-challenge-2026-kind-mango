"""5-model ensemble with SSL v16: v7/v10/v11/v15/v16.
Weights: v15=0.55, v16=0.15, v7=0.15, v11=0.10, v10=0.05."""
import os, pickle, sys, time, gc, glob
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
from features_opdi import add_opdi
from features_opdi_live import add_opdi_live
from features_ssl import add_ssl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")
MODELS = os.path.join(ROOT, "models")

WEIGHTS = {"v7": 0.15, "v10": 0.05, "v11": 0.10, "v15": 0.55, "v16": 0.15}

CAT_COLS = ["ADEP_mvt","ADES_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt",
            "WK_TBL_CAT_flt","MARKET_SEGMENT_flt","AIRCRAFT_OPERATOR_flt",
            "FLIGHT_RULE_mvt","FLIGHT_TYPE_flt"]


def load_training_categories_and_ssl():
    tdir = os.path.join(ROOT, "training")
    seen = {c: set() for c in CAT_COLS}
    ssl_rows = []
    for f in sorted(glob.glob(os.path.join(tdir, "*.parquet"))):
        t = pd.read_parquet(f, columns=list({*CAT_COLS, "PHASE_mvt",
                                             "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                             "TAXITIME_SEC_mvt"}))
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(TARGET_ICAOS)]
        for c in CAT_COLS:
            seen[c].update(t[c].dropna().unique())
        t["mvt_ts"] = pd.to_datetime(t["MVT_TIME_UTC_mvt"], errors="coerce")
        t["sched_ts"] = pd.to_datetime(t["SCHED_TIME_UTC_mvt"], errors="coerce")
        t["sched_delay"] = (t["mvt_ts"] - t["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
        t = t[t["TAXITIME_SEC_mvt"].between(30, 7200)]
        ssl_rows.append(t[["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "sched_delay"]])
    return ({c: pd.Index(list(v)) for c, v in seen.items()},
            pd.concat(ssl_rows, ignore_index=True))


def main(out_name="kind-mango_v9.parquet"):
    t0 = time.time()
    print("Loading ranking...")
    rank = pd.read_parquet(RANK)
    tmpl = pd.read_parquet(SUB_TMPL)
    rank["mvt_ts"] = pd.to_datetime(rank["MVT_TIME_UTC_mvt"], errors="coerce")
    rank["sched_ts"] = pd.to_datetime(rank["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = rank[(rank["PHASE_mvt"] == "DEP") & rank["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)

    print("Loading training cats + SSL...")
    train_cats, training_ssl = load_training_categories_and_ssl()

    print("Features...")
    rank_ctx = rank[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del rank; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, rank_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    with open(os.path.join(MODELS, "lgbm_v15.encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    dep = apply_encoders(dep, encoders)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, rank_ctx, daily_ec=ec_daily)
    del rank_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    dep = add_ssl(dep, training_ssl)
    del training_ssl; gc.collect()

    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=train_cats[c])

    preds = {}
    for v, w in WEIGHTS.items():
        b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_{v}.txt"))
        with open(os.path.join(MODELS, f"lgbm_{v}.features.txt")) as f:
            feat = f.read().splitlines()
        missing = [c for c in feat if c not in dep.columns]
        if missing:
            raise RuntimeError(f"Missing cols for {v}: {missing}")
        preds[v] = b.predict(dep[feat])
        print(f"  {v} ({w:.2f})")

    ens = sum(WEIGHTS[v] * preds[v] for v in WEIGHTS)
    dep["TAXITIME_SEC_mvt"] = ens

    out = tmpl[["MVT_ID_mvt"]].merge(dep[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]],
                                     on="MVT_ID_mvt", how="left")
    assert len(out) == len(tmpl)
    n_nan = out["TAXITIME_SEC_mvt"].isna().sum()
    if n_nan:
        med = float(np.nanmedian(out["TAXITIME_SEC_mvt"]))
        out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].fillna(med)
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].clip(60, 7200)
    print(f"[ok] rows={len(out):,}")

    path = os.path.join(ROOT, "submission", out_name)
    out.to_parquet(path)
    print(f"Saved -> {path}   ({os.path.getsize(path)/1024:.1f} KB)   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v9.parquet"
    main(name)
