"""Fit NNLS on hold-out for {v7,v11,v15,v16,v20} and apply to ranking.
v20 replaces v17 from v10 ensemble.
"""
import os, pickle, sys, time, gc, glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_eurocontrol import add_eurocontrol
from features_operator import fit_encoders, apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path
from features_opdi import add_opdi
from features_opdi_live import add_opdi_live
from features_ssl import add_ssl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
TAGS = ["v7", "v11", "v15", "v16", "v20"]
HOLDOUT_MONTHS = {1, 7}

CAT_COLS = ["ADEP_mvt","ADES_mvt","RUNWAY_mvt","STAND_mvt","AIRCRAFT_TYPE_mvt",
            "WK_TBL_CAT_flt","MARKET_SEGMENT_flt","AIRCRAFT_OPERATOR_flt",
            "FLIGHT_RULE_mvt","FLIGHT_TYPE_flt"]

SUS_RWY = {"NA", "-", "?", "N/A", "", "NAN"}
SUS_TYP = {"ZZZZ", "GLID", "UNKN", "", "-", "NAN"}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load_train_stack():
    """Build train dataframe with v15+ssl feature stack (superset for our 5 models)."""
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(ROOT, "training", "*.parquet")))
    m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_delay"] = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep = dep[dep["TAXITIME_SEC_mvt"].between(30, 7200)]

    ssl_ref = dep[["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "sched_delay"]].copy()
    m_ctx = m[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m_ctx, daily_ec=ec_daily)
    del m_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    dep = add_ssl(dep, ssl_ref)
    del ssl_ref; gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    encoders = fit_encoders(dep[~hold])
    dep = apply_encoders(dep, encoders)

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train = dep.loc[~hold]
    test = dep.loc[hold].copy()
    for c in CAT_COLS:
        cats = train[c].astype("category").cat.categories
        test[c] = pd.Categorical(test[c], categories=cats)
    del train, dep; gc.collect()
    print(f"  Hold-out ready in {time.time()-t0:.1f}s   n={len(test):,}")
    return test


def predict_tag(tag, X):
    b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_{tag}.txt"))
    with open(os.path.join(MODELS, f"lgbm_{tag}.features.txt")) as f:
        feat = f.read().splitlines()
    return b.predict(X[feat])


def fit_weights():
    test = load_train_stack()
    y = test["TAXITIME_SEC_mvt"].values.astype(float)

    P = {}
    for t in TAGS:
        p = predict_tag(t, test)
        P[t] = p
        print(f"  {t:5s} hold RMSE {rmse(y, p):.3f}s")

    stack = np.column_stack([P[t] for t in TAGS])
    w, _ = nnls(stack, y)
    ens = stack @ w
    print(f"NNLS RMSE {rmse(y, ens):.3f}s   sum {w.sum():.3f}")
    for t, wi in zip(TAGS, w):
        print(f"    {t}: {wi:.4f}")
    w_norm = w / w.sum()
    ens_n = stack @ w_norm
    print(f"Normalised RMSE {rmse(y, ens_n):.3f}s")
    del test; gc.collect()
    return {t: float(wi) for t, wi in zip(TAGS, w_norm)}


def apply_to_ranking(weights, out_name):
    print(f"\nApplying to ranking with weights: {weights}")
    RANK = os.path.join(ROOT, "submission", "ranking.parquet")
    SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")

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

    # v20 cleanup remap
    mm = dep["AIRCRAFT_TYPE_mvt"].isna() & dep["AIRCRAFT_TYPE_flt"].notna()
    dep.loc[mm, "AIRCRAFT_TYPE_mvt"] = dep.loc[mm, "AIRCRAFT_TYPE_flt"]
    rwy_up = dep["RUNWAY_mvt"].astype(str).str.strip().str.upper()
    dep.loc[rwy_up.isin(SUS_RWY), "RUNWAY_mvt"] = np.nan
    typ_up = dep["AIRCRAFT_TYPE_mvt"].astype(str).str.strip().str.upper()
    dep.loc[typ_up.isin(SUS_TYP), "AIRCRAFT_TYPE_mvt"] = "OTHER_TYPE"

    # Rebuild train cats (for LGBM categorical alignment we use v15 encoders)
    print("Loading v15 encoders for consistency...")
    with open(os.path.join(MODELS, "lgbm_v15.encoders.pkl"), "rb") as f:
        encoders_v15 = pickle.load(f)

    # SSL training reference
    print("Loading SSL train reference...")
    ssl_rows = []
    tcats = {c: set() for c in CAT_COLS}
    for f in sorted(glob.glob(os.path.join(ROOT, "training", "*.parquet"))):
        t = pd.read_parquet(f, columns=list({*CAT_COLS, "PHASE_mvt",
                                             "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                             "TAXITIME_SEC_mvt", "AIRCRAFT_TYPE_flt"}))
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(TARGET_ICAOS)]
        t = t[t["TAXITIME_SEC_mvt"].between(30, 7200)]
        for c in CAT_COLS:
            tcats[c].update(t[c].dropna().unique())
        t["mvt_ts"] = pd.to_datetime(t["MVT_TIME_UTC_mvt"], errors="coerce")
        t["sched_ts"] = pd.to_datetime(t["SCHED_TIME_UTC_mvt"], errors="coerce")
        t["sched_delay"] = (t["mvt_ts"] - t["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
        ssl_rows.append(t[["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "sched_delay"]])
    training_ssl = pd.concat(ssl_rows, ignore_index=True)
    tcats["STAND_mvt"].add("OTHER")
    tcats["AIRCRAFT_TYPE_mvt"].add("OTHER_TYPE")
    train_cats = {c: pd.Index(list(v)) for c, v in tcats.items()}

    print("Features...")
    rank_ctx = rank[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del rank; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, rank_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    dep = apply_encoders(dep, encoders_v15)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, rank_ctx, daily_ec=ec_daily)
    del rank_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    dep = add_ssl(dep, training_ssl)
    del training_ssl; gc.collect()

    stand_set = set(train_cats["STAND_mvt"])
    dep["STAND_mvt"] = dep["STAND_mvt"].where(dep["STAND_mvt"].isin(stand_set), other="OTHER")
    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=train_cats[c])

    ens = np.zeros(len(dep))
    for t, w in weights.items():
        p = predict_tag(t, dep)
        ens += w * p
        print(f"  {t} ({w:.4f})")
    dep["TAXITIME_SEC_mvt"] = ens

    out = tmpl[["MVT_ID_mvt"]].merge(dep[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]],
                                     on="MVT_ID_mvt", how="left")
    n_nan = out["TAXITIME_SEC_mvt"].isna().sum()
    if n_nan:
        med = float(np.nanmedian(out["TAXITIME_SEC_mvt"]))
        out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].fillna(med)
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].clip(60, 7200)
    path = os.path.join(ROOT, "submission", out_name)
    out.to_parquet(path)
    print(f"[ok] rows={len(out):,}   NaN filled: {n_nan}")
    print(f"Saved -> {path}   ({os.path.getsize(path)/1024:.1f} KB)")


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v12.parquet"
    t0 = time.time()
    print("=== Fitting weights on hold-out ===")
    weights = fit_weights()
    with open(os.path.join(MODELS, "ensemble_v20swap_weights.txt"), "w") as f:
        for t, w in weights.items():
            f.write(f"{t}: {w}\n")
    print(f"\n=== Applying to ranking -> {out_name} ===")
    apply_to_ranking(weights, out_name)
    print(f"\nTotal wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
