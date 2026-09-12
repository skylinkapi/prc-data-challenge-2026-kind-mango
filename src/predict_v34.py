"""v34 = v33 with hold-out leakage removed from the LIRF head statistics.

Audit Item 2. What changed vs v33:
  - Fallback-rate scoring maps built from fit months only (base_rate,
    per-key smoothed rates and counts). Model file: lgbm_p_fb_lirf_v34.txt.
  - LIRF band table rebuilt from fit months. File: lirf_band_table_v34.json.
  - NORMAL_MEAN_LIRF and R_NORM_CLIP re-estimated on fit months.
    File: models/v34_constants.json.

Base R_all_v26 (3-seed) and R_norm_LIRF (5-seed) are unchanged: their
training and their encoders already exclude {1, 7}.
"""
import os, pickle, time, gc, json, sys
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
from predict_v23 import add_rate_features_scoring, apply_stepA_v22
from predict_v30 import load_training_categories
from train_r_all_v24_seeds import SEEDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")
MODELS = os.path.join(ROOT, "models")

P24_ITY = 5.0/6.0
ITY_SD_THRESHOLD = 70000
DEFAULT_BASE_SEEDS = list(SEEDS)
DEFAULT_R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in (42, 43, 44, 45, 46)]


def load_v34_constants():
    with open(os.path.join(MODELS, "v34_constants.json")) as f:
        c = json.load(f)
    return float(c["NORMAL_MEAN_LIRF"]), float(c["R_NORM_CLIP"])


def main(out_name="kind-mango_v34.parquet",
         base_seeds=None, r_norm_files=None):
    base_seeds = base_seeds or DEFAULT_BASE_SEEDS
    r_norm_files = r_norm_files or DEFAULT_R_NORM_FILES
    NORMAL_MEAN_LIRF, R_NORM_CLIP = load_v34_constants()
    print(f"v34 constants: NORMAL_MEAN_LIRF={NORMAL_MEAN_LIRF:.2f}  "
          f"R_NORM_CLIP={R_NORM_CLIP:.2f}")

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
    for base, col in zip(SIGNED_LOG_BASE, SIGNED_LOG_COLS):
        dep[col] = signed_log(dep[base])

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
    with open(os.path.join(MODELS, "lgbm_r_all_v26.encoders.pkl"), "rb") as f:
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

    # v34 rate maps (fit months only)
    with open(os.path.join(MODELS, "lirf_regime_v34.rate_maps.pkl"), "rb") as f:
        rate_bundle = pickle.load(f)
    dep = add_rate_features_scoring(dep, rate_bundle["maps"], rate_bundle["base_rate"])

    openc = [c for c in dep.columns if c.startswith("openc_")]
    with open(os.path.join(MODELS, "lirf_regime.encoders.pkl"), "rb") as f:
        dep_lirf = apply_encoders(dep.drop(columns=openc), pickle.load(f))
    if not (dep_lirf["MVT_ID_mvt"].values == dep["MVT_ID_mvt"].values).all():
        raise RuntimeError("The LIRF encoder frame lost row alignment. apply_encoders reordered "
                           "the rows. Check the merge keys in features_operator.KEYS.")

    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=train_cats[c])
        dep_lirf[c] = pd.Categorical(dep_lirf[c], categories=train_cats[c])

    print(f"R_all_v26 mean over {len(base_seeds)} seeds {base_seeds}...")
    with open(os.path.join(MODELS, "lgbm_r_all_v26.features.txt")) as f:
        feat_all = f.read().splitlines()
    preds = []
    for s in base_seeds:
        b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_all_v26_s{s}.txt"))
        preds.append(np.clip(b.predict(dep[feat_all]), 0, None))
        del b; gc.collect()
    r_all = np.mean(preds, axis=0)
    del preds; gc.collect()

    print(f"LIRF regime head, {len(r_norm_files)} R_norm member(s), clip at {R_NORM_CLIP:.0f} s...")
    with open(os.path.join(MODELS, "lirf_regime.features.txt")) as f:
        feat_norm = f.read().splitlines()
    r_norm_members = []
    for fn in r_norm_files:
        b_n = lgb.Booster(model_file=os.path.join(MODELS, fn))
        r_norm_members.append(np.clip(b_n.predict(dep_lirf[feat_norm]), 0, None))
        del b_n; gc.collect()
    r_norm_lirf_raw = np.mean(r_norm_members, axis=0)
    r_norm_lirf = np.minimum(r_norm_lirf_raw, R_NORM_CLIP)
    n_clipped = (r_norm_lirf_raw > R_NORM_CLIP).sum()
    print(f"  R_norm_LIRF clipped {n_clipped} predictions at {R_NORM_CLIP:.2f}s")
    del r_norm_members; gc.collect()

    b_fb = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_p_fb_lirf_v34.txt"))
    with open(os.path.join(MODELS, "lirf_regime_v34.features.txt")) as f:
        feat_fb_v34 = f.read().splitlines()
    with open(os.path.join(MODELS, "lirf_regime_v34.isotonic.pkl"), "rb") as f:
        iso = pickle.load(f)
    p_fb_raw = b_fb.predict(dep_lirf[feat_fb_v34])
    p_fb_cal = iso.transform(p_fb_raw)
    del b_fb; gc.collect()

    sd_r = dep["sched_delay"].values.astype(float)
    adep_r = dep["ADEP_mvt"].astype(str).values
    fltid_r = dep["FLIGHT_ID_mvt"].values

    p_final = r_all.copy()
    lirf_mask = (adep_r == "LIRF")
    valid_sd = ~np.isnan(sd_r)
    mixture = np.where(valid_sd,
                       p_fb_cal * sd_r + (1 - p_fb_cal) * r_norm_lirf,
                       r_norm_lirf)
    p_final[lirf_mask] = mixture[lirf_mask]

    with open(os.path.join(MODELS, "lirf_band_table_v34.json")) as f:
        band = json.load(f)
    p_final, cell_A = apply_stepA_v22(p_final, sd_r, adep_r, fltid_r, band)
    print(f"Step A: {cell_A.sum()} rows")

    ity_mask = lirf_mask & (sd_r > ITY_SD_THRESHOLD) & valid_sd & ~cell_A
    print(f"ITY340 rule: {ity_mask.sum()} rows")
    for i in np.where(ity_mask)[0]:
        p_final[i] = P24_ITY * (86400 + NORMAL_MEAN_LIRF) + (1 - P24_ITY) * sd_r[i]
        print(f"  row sd={sd_r[i]:.0f}  new_pred={p_final[i]:.0f}")

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

    path = os.path.join(ROOT, "submission", out_name)
    out.to_parquet(path)
    print(f"Saved -> {path}   ({os.path.getsize(path)/1024:.1f} KB)   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v34.parquet"
    main(name)
