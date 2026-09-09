"""v21 submission: R_all_v21 (linear_tree + fixes + temperature) + tail rules.

Same pipeline as v20 but using R_all_v21 as base.
"""
import os, pickle, sys, time, gc, glob, json
import numpy as np
import pandas as pd
import lightgbm as lgb

from features_weather import add_weather, TARGET_ICAOS
from features_congestion import add_congestion
from features_congestion_v2 import add_congestion_v2
from features_eurocontrol import add_eurocontrol
from features_operator import apply_encoders
from features_taxi_distance import add_taxi_distance
from features_advanced import add_advanced
from features_osm_path import add_osm_path
from features_opdi import add_opdi
from features_opdi_live import add_opdi_live
from features_turnaround import add_turnaround
from features_disruption import add_disruption
from train_lgbm_v21 import add_obt_features, CAT_COLS
from train_r_all_v21 import SIGNED_LOG_BASE, SIGNED_LOG_COLS, signed_log
from eval_v21_stepA import apply_rule
from train_lirf_noflt_detector import CAT_INPUTS as NF_CAT, NUM_INPUTS as NF_NUM, SD_LO, SD_HI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")
MODELS = os.path.join(ROOT, "models")
SHRINK_63 = 0.6
P24_ITY = 5.0/6.0
ITY_SD_THRESHOLD = 70000


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


def main(out_name="kind-mango_v21.parquet"):
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
    dep["stand_prefix"] = dep["STAND_mvt"].astype(str).str.slice(0, 1).fillna("UNK")
    # signed-log copies
    for base, col in zip(SIGNED_LOG_BASE, SIGNED_LOG_COLS):
        dep[col] = signed_log(dep[base])

    print("Loading training categories...")
    train_cats = load_training_categories()

    print("Features...")
    ctx = rank[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts","sched_ts",
                "RUNWAY_mvt","STAND_mvt","TAXITIME_SEC_mvt","FLIGHT_ID_mvt",
                "BLOCK_TIME_UTC_mvt"]].copy()
    ctx_adv = rank[rank["ADEP_mvt"].isin(TARGET_ICAOS)][
        ["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del rank; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_congestion_v2(dep, ctx[["ADEP_mvt","ADES_mvt","PHASE_mvt","mvt_ts",
                                       "RUNWAY_mvt","TAXITIME_SEC_mvt"]])
    dep = add_eurocontrol(dep)
    with open(os.path.join(MODELS, "lgbm_r_all_v21.encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    dep = apply_encoders(dep, encoders)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, ctx_adv, daily_ec=ec_daily)
    del ctx_adv, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    dep = add_turnaround(dep, ctx)
    dep = add_disruption(dep, ctx)
    del ctx; gc.collect()

    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=train_cats[c])

    # R_all v21
    b = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_r_all_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_r_all_v21.features.txt")) as f:
        feat = f.read().splitlines()
    r_all = np.clip(b.predict(dep[feat]), 0, None)
    print(f"R_all v21 predictions: min {r_all.min():.0f}  max {r_all.max():.0f}  mean {r_all.mean():.0f}")

    # v21 v1 (old model) for Step A band rule 'rest' component
    b21 = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        feat21 = f.read().splitlines()
    taxi21 = np.clip(b21.predict(dep[feat21]), 0, None)

    sd_r = dep["sched_delay"].values.astype(float)
    adep_r = dep["ADEP_mvt"].astype(str).values
    fltid_r = dep["FLIGHT_ID_mvt"].values

    # v16 = v21_old + Step A
    with open(os.path.join(MODELS, "lirf_band_table.json")) as f:
        band = json.load(f)
    v16, cell_A = apply_rule(taxi21, sd_r, adep_r, fltid_r, band)

    # Base: R_all_v21 everywhere except LIRF
    p_final = r_all.copy()
    lirf_mask = (adep_r == "LIRF")
    p_final[lirf_mask] = v16[lirf_mask]

    # 6.3 rule
    det63 = lgb.Booster(model_file=os.path.join(MODELS, "lirf_noflt_detector.txt"))
    with open(os.path.join(MODELS, "lirf_noflt_detector_meta.json")) as f:
        meta63 = json.load(f)
    for c in NF_CAT:
        if c in dep.columns and not isinstance(dep[c].dtype, pd.CategoricalDtype):
            dep[c] = dep[c].astype("category")
    mask63 = (adep_r == "LIRF") & pd.isna(fltid_r) & (sd_r > SD_LO) & (sd_r <= SD_HI) & ~np.isnan(sd_r)
    if mask63.sum():
        p_raw = det63.predict(dep.loc[mask63, NF_CAT + NF_NUM])
        p_shrunk = SHRINK_63 * p_raw
        p_final[mask63] = p_shrunk * sd_r[mask63] + (1 - p_shrunk) * meta63["normal_mean"]
    print(f"Step A: {cell_A.sum()} rows  |  6.3 rule: {mask63.sum()} rows")

    # ITY340 rule
    ity_mask = lirf_mask & (sd_r > ITY_SD_THRESHOLD) & ~np.isnan(sd_r) & ~cell_A
    print(f"ITY340 rule candidates: {ity_mask.sum()}")
    for i in np.where(ity_mask)[0]:
        p_final[i] = P24_ITY * (86400 + r_all[i]) + (1 - P24_ITY) * r_all[i]
        print(f"  row: sd={sd_r[i]:.0f}  new_pred={p_final[i]:.0f}")

    p_final = np.clip(p_final, 0, None)
    dep["TAXITIME_SEC_mvt"] = p_final

    out = tmpl[["MVT_ID_mvt"]].merge(dep[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]],
                                     on="MVT_ID_mvt", how="left")
    n_nan = out["TAXITIME_SEC_mvt"].isna().sum()
    if n_nan:
        med = float(np.nanmedian(out["TAXITIME_SEC_mvt"]))
        out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].fillna(med)
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].clip(lower=0)
    print(f"[ok] rows={len(out):,}   NaN filled: {n_nan}")
    print(f"     pred: min {out['TAXITIME_SEC_mvt'].min():.0f}  max {out['TAXITIME_SEC_mvt'].max():.0f}")

    path = os.path.join(ROOT, "submission", out_name)
    out.to_parquet(path)
    print(f"Saved -> {path}   ({os.path.getsize(path)/1024:.1f} KB)   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v21.parquet"
    main(name)
