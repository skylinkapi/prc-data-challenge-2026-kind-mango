"""v30 = v29 with two changes that do not depend on a hold-out difference.

  1. The LIRF models read the LIRF-only encoders they were trained with.
     v29 served them the all-airport encoders (MODEL_ANALYSIS.md section 2).
  2. The Step A band table comes from build_lirf_band_table_v30.py, whose classes
     are mutually exclusive. NOSOS431 moves from 121,410 s to its sd (section 3).
"""
import os, pickle, time, gc, glob, json, sys
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
from features_plan import add_plan_times, add_plan_residual
from train_lgbm_v21 import add_obt_features, CAT_COLS
from train_r_all_v21 import SIGNED_LOG_BASE, SIGNED_LOG_COLS, signed_log
from train_lirf_regime_v23 import RATE_KEYS, RATE_FEATURES
from predict_v23 import add_rate_features_scoring, apply_stepA_v22
from train_r_all_v24_seeds import SEEDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")
MODELS = os.path.join(ROOT, "models")
P24_ITY = 5.0/6.0
ITY_SD_THRESHOLD = 70000
R_NORM_CLIP = 4431.0
NORMAL_MEAN_LIRF = 1150.0


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


DEFAULT_BASE_SEEDS = list(SEEDS)
DEFAULT_R_NORM_FILES = ["lgbm_r_norm_lirf.txt"]
DEFAULT_P_FB_MEMBERS = [("lgbm_p_fb_lirf_v23.txt", "lirf_regime_v23.isotonic.pkl")]
DEFAULT_P_FB_FEATURES = "lirf_regime_v23.features.txt"


def predict_p_fb(frame, members, features_file):
    """Mean of the calibrated fallback probabilities over (booster, isotonic) members."""
    with open(os.path.join(MODELS, features_file)) as f:
        feat = f.read().splitlines()
    probs = []
    for model_file, iso_file in members:
        with open(os.path.join(MODELS, iso_file), "rb") as f:
            iso = pickle.load(f)
        probs.append(iso.transform(lgb.Booster(model_file=os.path.join(MODELS, model_file))
                                   .predict(frame[feat])))
    return np.mean(probs, axis=0)


def main(out_name="kind-mango_v30.parquet",
         base_seeds=None, r_norm_files=None,
         p_fb_members=None, p_fb_features=None,
         per_member_base_clip=True, fill_zero_rows_per_airport=False,
         base_model="lgbm_r_all_v26", use_plan_features=False, extra_columns=None,
         r_norm_features="lirf_regime.features.txt", r_norm_clip=R_NORM_CLIP,
         dump_features=None, stepa_normal_rnorm=False, r_norm_post=None):
    base_seeds = base_seeds or DEFAULT_BASE_SEEDS
    r_norm_files = r_norm_files or DEFAULT_R_NORM_FILES
    p_fb_members = p_fb_members or DEFAULT_P_FB_MEMBERS
    p_fb_features = p_fb_features or DEFAULT_P_FB_FEATURES
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
    if use_plan_features:
        medians = pd.read_parquet(os.path.join(MODELS, f"{base_model}.route_medians.parquet"))["plan_block_median"]
        dep = add_plan_residual(add_plan_times(dep), medians)
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
    with open(os.path.join(MODELS, f"{base_model}.encoders.pkl"), "rb") as f:
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
    if extra_columns is not None:
        dep = dep.merge(extra_columns, on="MVT_ID_mvt", how="left")

    with open(os.path.join(MODELS, "lirf_regime_v23.rate_maps.pkl"), "rb") as f:
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

    print(f"{base_model} mean over {len(base_seeds)} seeds {base_seeds}...")
    with open(os.path.join(MODELS, f"{base_model}.features.txt")) as f:
        feat_all = f.read().splitlines()
    preds = []
    for s in base_seeds:
        b = lgb.Booster(model_file=os.path.join(MODELS, f"{base_model}_s{s}.txt"))
        raw = b.predict(dep[feat_all])
        preds.append(np.clip(raw, 0, None) if per_member_base_clip else raw)
        del b; gc.collect()
    r_all = np.mean(preds, axis=0)
    del preds; gc.collect()

    # LIRF regime head + R_norm clip
    print(f"LIRF regime head, {len(r_norm_files)} R_norm member(s), clip at 4,431 s...")
    with open(os.path.join(MODELS, r_norm_features)) as f:
        feat_norm = f.read().splitlines()
    if dump_features is not None:
        with open(os.path.join(MODELS, p_fb_features)) as f:
            feat_pfb = f.read().splitlines()
        dump_cols = ["MVT_ID_mvt", "ADEP_mvt", "month"] + sorted(
            (set(feat_all) | set(feat_norm) | set(feat_pfb)) - {"MVT_ID_mvt", "ADEP_mvt", "month"})
        dep[[c for c in dump_cols if c in dep.columns]].to_parquet(dump_features)
        print(f"feature dump -> {dump_features} ({len(dump_cols)} columns)")
    r_norm_members = []
    for fn in r_norm_files:
        b_n = lgb.Booster(model_file=os.path.join(MODELS, fn))
        r_norm_members.append(np.clip(b_n.predict(dep_lirf[feat_norm]), 0, None))
        del b_n; gc.collect()
    r_norm_lirf_raw = np.mean(r_norm_members, axis=0)
    r_norm_lirf = r_norm_lirf_raw if r_norm_clip is None else np.minimum(r_norm_lirf_raw, r_norm_clip)
    if r_norm_post is not None:
        r_norm_lirf = r_norm_post(dep_lirf, r_norm_lirf)
    n_clipped = 0 if r_norm_clip is None else (r_norm_lirf_raw > r_norm_clip).sum()
    print(f"  R_norm_LIRF clipped {n_clipped} predictions at {r_norm_clip}s")
    del r_norm_members; gc.collect()

    print(f"p_fb mean over {len(p_fb_members)} calibrated member(s)...")
    p_fb_cal = predict_p_fb(dep_lirf, p_fb_members, p_fb_features)

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

    with open(os.path.join(MODELS, "lirf_band_table_v30.json")) as f:
        band = json.load(f)
    stepa_in = p_final.copy()
    if stepa_normal_rnorm:
        # T1: the band table already weights the schedule; the mixture would count it twice.
        stepa_in[lirf_mask] = r_norm_lirf[lirf_mask]
    stepa_out, cell_A = apply_stepA_v22(stepa_in, sd_r, adep_r, fltid_r, band)
    p_final[cell_A] = stepa_out[cell_A]
    print(f"Step A: {cell_A.sum()} rows")

    # STEP 2 FIX: ITY340 uses constant terms
    ity_mask = lirf_mask & (sd_r > ITY_SD_THRESHOLD) & valid_sd & ~cell_A
    print(f"ITY340 rule (fixed): {ity_mask.sum()} rows")
    for i in np.where(ity_mask)[0]:
        p_final[i] = P24_ITY * (86400 + NORMAL_MEAN_LIRF) + (1 - P24_ITY) * sd_r[i]
        print(f"  row sd={sd_r[i]:.0f}  new_pred={p_final[i]:.0f}")

    p_final = np.clip(p_final, 0, None)
    if fill_zero_rows_per_airport:
        # F7 repair: exactly-0 rows lose the calibrated floor of the ensemble.
        # Fill each with the median of positive predictions at its own airport.
        zero = (p_final == 0)
        for a in np.unique(adep_r[zero]):
            positive_here = (adep_r == a) & (p_final > 0)
            if positive_here.any():
                p_final[zero & (adep_r == a)] = float(np.median(p_final[positive_here]))
        print(f"F7 fill: {int(zero.sum())} zero rows replaced with per-airport medians")
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
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v30.parquet"
    main(name)
