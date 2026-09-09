"""Submission: v20 combined = per-airport winner selection.

Per airport choice (from eval_v20_combined.py on Jan+Jul 2025 hold-out):
  LIRF: v18 (v16 + 6.3 rule)
  EHAM: v18
  EGLL: R_norm + detector soft mix
  LEBL: R_norm + detector soft mix
  LTFM: R_norm + detector soft mix
  EDDF: R_norm alone
  EDDM: R_norm alone
  LEMD: R_norm alone
  LFPG: R_norm alone
  LSZH: R_norm alone

Note: filename v19 is our next submission number after kind-mango_v18.parquet.
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
from train_lirf_noflt_detector import CAT_INPUTS as NF_CAT, NUM_INPUTS as NF_NUM, SD_LO, SD_HI
from train_v18_detectors import CAT_INPUTS as DET_CAT, NUM_INPUTS as DET_NUM, SD_MIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")
MODELS = os.path.join(ROOT, "models")
SHRINK_63 = 0.6


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


def main(out_name="kind-mango_v19.parquet"):
    t0 = time.time()
    with open(os.path.join(MODELS, "v20_apt_choice.json")) as f:
        apt_choice = json.load(f)
    print(f"per-airport choice: {apt_choice}")

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
    dep["ades_region"] = dep["ADES_mvt"].astype(str).str[:2].fillna("UNK")

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

    # v21 for v18 baseline
    b21 = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f: feat21 = f.read().splitlines()
    taxi21 = np.clip(b21.predict(dep[feat21]), 0, None)

    # R_norm
    b_rn = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_r_norm.txt"))
    with open(os.path.join(MODELS, "lgbm_r_norm.features.txt")) as f: feat_rn = f.read().splitlines()
    r_norm = np.clip(b_rn.predict(dep[feat_rn]), 0, None)

    sd_r = dep["sched_delay"].values.astype(float)
    adep_r = dep["ADEP_mvt"].astype(str).values
    fltid_r = dep["FLIGHT_ID_mvt"].values

    # v16 = v21 + Step A
    with open(os.path.join(MODELS, "lirf_band_table.json")) as f: band = json.load(f)
    v16, cell_A = apply_rule(taxi21, sd_r, adep_r, fltid_r, band)
    print(f"Step A: {cell_A.sum()} rows")

    # v18 = v16 + 6.3 rule (LIRF null sd 3600-14400)
    det63 = lgb.Booster(model_file=os.path.join(MODELS, "lirf_noflt_detector.txt"))
    with open(os.path.join(MODELS, "lirf_noflt_detector_meta.json")) as f:
        meta63 = json.load(f)
    for c in NF_CAT:
        if c in dep.columns and not isinstance(dep[c].dtype, pd.CategoricalDtype):
            dep[c] = dep[c].astype("category")
    mask63 = (adep_r == "LIRF") & pd.isna(fltid_r) & (sd_r > SD_LO) & (sd_r <= SD_HI) & ~np.isnan(sd_r)
    v18 = v16.copy()
    if mask63.sum():
        p_raw = det63.predict(dep.loc[mask63, NF_CAT + NF_NUM])
        p_shrunk = SHRINK_63 * p_raw
        v18[mask63] = p_shrunk * sd_r[mask63] + (1 - p_shrunk) * meta63["normal_mean"]
    print(f"6.3 rule: {mask63.sum()} rows")

    for c in DET_CAT:
        if c in dep.columns and not isinstance(dep[c].dtype, pd.CategoricalDtype):
            dep[c] = dep[c].astype("category")

    # Final: per-airport winner
    p_final = v18.copy()
    for apt, choice in apt_choice.items():
        apt_mask = (adep_r == apt)
        if apt_mask.sum() == 0 or choice == "v18":
            continue
        if choice == "R_norm":
            p_final[apt_mask] = r_norm[apt_mask]
            print(f"  {apt}: R_norm alone -> {apt_mask.sum()} rows")
        elif choice == "R+mix":
            det_path = os.path.join(MODELS, f"v18_det_{apt}.txt")
            det = lgb.Booster(model_file=det_path)
            sub_mask = apt_mask & (sd_r > SD_MIN) & ~np.isnan(sd_r)
            p_final[apt_mask] = r_norm[apt_mask]
            if sub_mask.sum():
                p_fb = det.predict(dep.loc[sub_mask, DET_CAT + DET_NUM])
                p_final[sub_mask] = p_fb * sd_r[sub_mask] + (1 - p_fb) * r_norm[sub_mask]
            print(f"  {apt}: R+mix -> {apt_mask.sum()} rows ({sub_mask.sum()} with detector)")

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
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v19.parquet"
    main(name)
