"""v18 = v16 + Section 6.3 rule (LIRF no-flight-record, 3600 < sd <= 14400).

Pipeline:
  1. v21 taxi_hat
  2. Step A LIRF band rule (sd > 14400, null flight) -> replaces v21 for 43 rows
  3. 6.3 rule: LIRF null-flight, 3600 < sd <= 14400 -> mix
     p_shrunk = 0.6 * detector.predict(x)
     p_final  = p_shrunk * sd + (1 - p_shrunk) * 1150 (normal_mean)
"""
import os, pickle, sys, time, gc, glob, json
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
from train_lgbm_v21 import add_obt_features, CAT_COLS
from eval_v21_stepA import apply_rule
from train_lirf_noflt_detector import CAT_INPUTS as DET_CAT, NUM_INPUTS as DET_NUM, SD_LO, SD_HI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")
MODELS = os.path.join(ROOT, "models")
SHRINK = 0.6


def load_training_categories():
    tdir = os.path.join(ROOT, "training")
    seen = {c: set() for c in CAT_COLS}
    for f in sorted(glob.glob(os.path.join(tdir, "*.parquet"))):
        t = pd.read_parquet(f, columns=list({*CAT_COLS, "PHASE_mvt", "TAXITIME_SEC_mvt"}))
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(TARGET_ICAOS)]
        t = t[t["TAXITIME_SEC_mvt"].astype(float) > 0]
        for c in CAT_COLS:
            seen[c].update(t[c].dropna().unique())
    return {c: pd.Index(list(v)) for c, v in seen.items()}


def main(out_name="kind-mango_v18.parquet"):
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
    sd = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds()
    dep["sched_delay"] = sd
    dep["sd"] = sd
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep["flt_id_null"] = dep["FLIGHT_ID_mvt"].isna().astype(np.int8)
    dep = add_obt_features(dep)
    dep["flt_prefix"] = dep["FLIGHT_mvt"].astype(str).str[:3].fillna("UNK")
    dep["stand_prefix"] = dep["STAND_mvt"].astype(str).str.extract(r"^([A-Za-z]+)")[0].fillna("UNK")

    print("Loading training categories...")
    train_cats = load_training_categories()

    print("Features...")
    rank_ctx = rank[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del rank; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, rank_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    with open(os.path.join(MODELS, "lgbm_v21.encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    dep = apply_encoders(dep, encoders)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, rank_ctx, daily_ec=ec_daily)
    del rank_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    gc.collect()

    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=train_cats[c])

    # v21 predictions
    b = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        feat = f.read().splitlines()
    taxi_hat = np.clip(b.predict(dep[feat]), 0, None)

    # Step A: LIRF null-flight sd>14400
    with open(os.path.join(MODELS, "lirf_band_table.json")) as f:
        table_data = json.load(f)
    sd_r = dep["sched_delay"].values.astype(float)
    adep_r = dep["ADEP_mvt"].astype(str).values
    fltid_r = dep["FLIGHT_ID_mvt"].values
    p_A, cell_A = apply_rule(taxi_hat, sd_r, adep_r, fltid_r, table_data)
    print(f"Step A: {cell_A.sum()} rows")

    # 6.3: LIRF null-flight 3600 < sd <= 14400
    det = lgb.Booster(model_file=os.path.join(MODELS, "lirf_noflt_detector.txt"))
    with open(os.path.join(MODELS, "lirf_noflt_detector_meta.json")) as f:
        meta = json.load(f)
    normal_mean = meta["normal_mean"]

    mask63 = ((adep_r == "LIRF") & pd.isna(fltid_r) &
              (sd_r > SD_LO) & (sd_r <= SD_HI) & ~np.isnan(sd_r))
    print(f"6.3 cell: {mask63.sum()} rows (expected ~330)")

    for c in DET_CAT:
        dep[c] = dep[c].astype("category")

    p_final = p_A.copy()
    if mask63.sum() > 0:
        p_raw = det.predict(dep.loc[mask63, DET_CAT + DET_NUM])
        p_shrunk = SHRINK * p_raw
        pred = p_shrunk * sd_r[mask63] + (1 - p_shrunk) * normal_mean
        p_final[mask63] = pred
        print(f"  raw p_fb  mean {p_raw.mean():.3f}  max {p_raw.max():.3f}")
        print(f"  shrunk    mean {p_shrunk.mean():.3f}")
        print(f"  new pred  mean {pred.mean():.0f}   min {pred.min():.0f}  max {pred.max():.0f}")

    dep["TAXITIME_SEC_mvt"] = p_final
    out = tmpl[["MVT_ID_mvt"]].merge(dep[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]],
                                     on="MVT_ID_mvt", how="left")
    n_nan = out["TAXITIME_SEC_mvt"].isna().sum()
    if n_nan:
        med = float(np.nanmedian(out["TAXITIME_SEC_mvt"]))
        out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].fillna(med)
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].clip(lower=0)
    print(f"[ok] rows={len(out):,}   NaN filled: {n_nan}")

    path = os.path.join(ROOT, "submission", out_name)
    out.to_parquet(path)
    print(f"Saved -> {path}   ({os.path.getsize(path)/1024:.1f} KB)   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v18.parquet"
    main(name)
