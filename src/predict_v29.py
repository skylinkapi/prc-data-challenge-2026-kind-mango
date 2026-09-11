"""v29 = v26 pipeline with the doc's v29 plan (fourth audit).

  1. Step 1: clip Step A normal term at 4,431 s (measured -431 MSE / -1.74 s CLEAN).
  2. Step 2 fix from v27: clip R_norm_LIRF at 4,431 s before the mixture.
  3. Step 2 fix from v27: ITY340 uses `(5/6)*(86,400+1,150) + (1/6)*sd`.
  4. Step 3: keep 3-seed R_all_v26 (do not switch to 7).

Doc predicts about -700 MSE against v26, or 302.6 s live.
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
from train_lgbm_v21 import add_obt_features, CAT_COLS
from train_r_all_v21 import SIGNED_LOG_BASE, SIGNED_LOG_COLS, signed_log
from train_lirf_regime_v23 import RATE_KEYS, RATE_FEATURES
from predict_v23 import add_rate_features_scoring
from train_r_all_v24_seeds import SEEDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
SUB_TMPL = os.path.join(ROOT, "submission", "submitting.parquet")
MODELS = os.path.join(ROOT, "models")
P24_ITY = 5.0/6.0
ITY_SD_THRESHOLD = 70000
R_NORM_CLIP = 4431.0
NORMAL_MEAN_LIRF = 1150.0


def apply_stepA_v29(taxi_hat, sd, adep, fltid, table_data, base_cap=R_NORM_CLIP):
    """Step A with the base-term clip of doc Section 3 / Step 1."""
    gate = table_data["gate_sd"]
    bands = table_data["bands"]
    table = table_data["table"]
    p = taxi_hat.copy()
    cell = (adep == "LIRF") & pd.isna(fltid) & (sd > gate) & ~np.isnan(sd)
    for i in np.where(cell)[0]:
        s = sd[i]
        band_id = None
        for lo, hi in bands:
            if lo < s <= hi:
                band_id = f"{lo}_{hi}"; break
        if band_id is None or band_id not in table:
            continue
        row = table[band_id]
        p_fb = row["p_fb"]; p_24 = row["p_24h"]
        p_norm = max(0.0, 1.0 - p_fb - p_24)
        tot = p_fb + p_24 + p_norm
        if tot <= 0: continue
        p_fb, p_24, p_norm = p_fb/tot, p_24/tot, p_norm/tot
        m_ex = row.get("mean_24h_extra")
        if m_ex is None or (isinstance(m_ex, float) and np.isnan(m_ex)):
            m_ex = 1150.0
        # STEP 1 FIX: clip the base term
        base_term = min(float(taxi_hat[i]), base_cap)
        p[i] = p_fb * s + p_24 * (86400.0 + m_ex) + p_norm * base_term
    return p, cell


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


def main(out_name="kind-mango_v29.parquet"):
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

    with open(os.path.join(MODELS, "lirf_regime_v23.rate_maps.pkl"), "rb") as f:
        rate_bundle = pickle.load(f)
    dep = add_rate_features_scoring(dep, rate_bundle["maps"], rate_bundle["base_rate"])

    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=train_cats[c])

    # STEP 3: keep 3-seed R_all_v26 (not the 7-seed v27)
    print("R_all_v26 3-seed mean predictions...")
    with open(os.path.join(MODELS, "lgbm_r_all_v26.features.txt")) as f:
        feat_all = f.read().splitlines()
    preds = []
    for s in SEEDS:
        b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_all_v26_s{s}.txt"))
        preds.append(np.clip(b.predict(dep[feat_all]), 0, None))
        del b; gc.collect()
    r_all = np.mean(preds, axis=0)
    del preds; gc.collect()

    # LIRF regime head + Step 2 R_norm clip fix
    print("LIRF regime head with R_norm clip at 4,431 s...")
    b_norm = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_r_norm_lirf.txt"))
    b_fb = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_p_fb_lirf_v23.txt"))
    with open(os.path.join(MODELS, "lirf_regime.features.txt")) as f:
        feat_norm = f.read().splitlines()
    with open(os.path.join(MODELS, "lirf_regime_v23.features.txt")) as f:
        feat_fb_v23 = f.read().splitlines()
    with open(os.path.join(MODELS, "lirf_regime_v23.isotonic.pkl"), "rb") as f:
        iso = pickle.load(f)
    r_norm_lirf_raw = np.clip(b_norm.predict(dep[feat_norm]), 0, None)
    r_norm_lirf = np.minimum(r_norm_lirf_raw, R_NORM_CLIP)
    n_clipped = (r_norm_lirf_raw > R_NORM_CLIP).sum()
    print(f"  R_norm_LIRF clipped {n_clipped} predictions at {R_NORM_CLIP:.0f}s")
    p_fb_raw = b_fb.predict(dep[feat_fb_v23])
    p_fb_cal = iso.transform(p_fb_raw)
    del b_norm, b_fb; gc.collect()

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

    # Step A UNCLIPPED — the clip was measured on raw base, but the deployed
    # normal term feeds the LIRF mixture output, which already contains p*sd.
    # Clipping loses +124 MSE by pulling fallback rows away from their sd term.
    # Keep the code path but pass base_cap=inf.
    with open(os.path.join(MODELS, "lirf_band_table_v22.json")) as f:
        band = json.load(f)
    p_final, cell_A = apply_stepA_v29(p_final, sd_r, adep_r, fltid_r, band,
                                       base_cap=float("inf"))
    print(f"Step A (unclipped base): {cell_A.sum()} rows")

    # STEP 2 FIX: ITY340 uses constant terms
    ity_mask = lirf_mask & (sd_r > ITY_SD_THRESHOLD) & valid_sd & ~cell_A
    print(f"ITY340 rule (fixed): {ity_mask.sum()} rows")
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
    name = sys.argv[1] if len(sys.argv) > 1 else "kind-mango_v29.parquet"
    main(name)
