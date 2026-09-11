"""Label-free ambiguity price for v32 ensemble changes (audit section 5).

Loads features once, scores all 5 R_norm_LIRF members on LIRF rows and all 7
base seeds on all rows, then reports A on the 2026 ranking set.
"""
import os, sys, pickle, gc, glob, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
from predict_v23 import add_rate_features_scoring
from predict_v30 import load_training_categories, R_NORM_CLIP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "submission", "ranking.parquet")
MODELS = os.path.join(ROOT, "models")
BASE_SEEDS = [42, 43, 44, 45, 46, 47, 48]
R_NORM_SEEDS = [42, 43, 44, 45, 46]


def build_dep_frames():
    t0 = time.time()
    print("Loading ranking...")
    rank = pd.read_parquet(RANK)
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

    openc = [c for c in dep.columns if c.startswith("openc_")]
    with open(os.path.join(MODELS, "lirf_regime.encoders.pkl"), "rb") as f:
        dep_lirf_full = apply_encoders(dep.drop(columns=openc), pickle.load(f))
    if not (dep_lirf_full["MVT_ID_mvt"].values == dep["MVT_ID_mvt"].values).all():
        raise RuntimeError("LIRF encoder frame lost row alignment.")

    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=train_cats[c])
        dep_lirf_full[c] = pd.Categorical(dep_lirf_full[c], categories=train_cats[c])
    print(f"  Feature build wall {time.time()-t0:.1f}s")
    return dep, dep_lirf_full


def main():
    dep, dep_lirf_full = build_dep_frames()
    adep = dep["ADEP_mvt"].astype(str).values
    lirf_mask = (adep == "LIRF")
    dep_lirf = dep_lirf_full[lirf_mask].reset_index(drop=True)
    sd = dep["sched_delay"].values.astype(float)
    sd_lirf = sd[lirf_mask]
    valid_sd_lirf = ~np.isnan(sd_lirf)

    # Base per-seed predictions on all ADEP rows
    print(f"\nBase members ({len(BASE_SEEDS)} seeds)...")
    with open(os.path.join(MODELS, "lgbm_r_all_v26.features.txt")) as f:
        feat_all = f.read().splitlines()
    base_preds = []
    for s in BASE_SEEDS:
        b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_all_v26_s{s}.txt"))
        p = np.clip(b.predict(dep[feat_all]), 0, None)
        base_preds.append(p)
        print(f"  seed {s}: mean={p.mean():.1f}")
        del b; gc.collect()
    base_preds = np.stack(base_preds, axis=0)  # (7, N)

    # R_norm_LIRF per-member predictions on LIRF rows (clipped at 0 per member,
    # then clipped at R_NORM_CLIP mean-side per section 1.3 step 4 order)
    print(f"\nR_norm_LIRF members ({len(R_NORM_SEEDS)} seeds)...")
    with open(os.path.join(MODELS, "lirf_regime.features.txt")) as f:
        feat_norm = f.read().splitlines()
    norm_preds = []
    for s in R_NORM_SEEDS:
        b = lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_norm_lirf_s{s}.txt"))
        p = np.clip(b.predict(dep_lirf[feat_norm]), 0, None)
        norm_preds.append(p)
        print(f"  s{s}: mean={p.mean():.1f}")
        del b; gc.collect()
    norm_preds = np.stack(norm_preds, axis=0)  # (5, n_lirf)
    norm_clipped = np.minimum(norm_preds, R_NORM_CLIP)

    # Post-mixture per-member LIRF predictions (section 5.1 "final stacked").
    # Need p_fb_cal for LIRF rows.
    print("\nBuilding p_fb_cal for LIRF rows...")
    b_fb = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_p_fb_lirf_v23.txt"))
    with open(os.path.join(MODELS, "lirf_regime_v23.features.txt")) as f:
        feat_fb_v23 = f.read().splitlines()
    with open(os.path.join(MODELS, "lirf_regime_v23.isotonic.pkl"), "rb") as f:
        iso = pickle.load(f)
    p_fb_raw = b_fb.predict(dep_lirf[feat_fb_v23])
    p_fb_cal = iso.transform(p_fb_raw)
    del b_fb; gc.collect()

    # Post-mixture per-member LIRF (each member scoring with the mean base).
    #   mixture = p_fb * sd + (1 - p_fb) * r_norm_lirf (or r_norm alone if sd NaN)
    mix_members = []
    for m in norm_clipped:
        mm = np.where(valid_sd_lirf,
                      p_fb_cal * sd_lirf + (1 - p_fb_cal) * m,
                      m)
        mix_members.append(mm)
    mix_members = np.stack(mix_members, axis=0)  # (5, n_lirf)
    mean_mix = mix_members.mean(axis=0)
    A_5_post = float(((mix_members - mean_mix[None, :]) ** 2).mean())

    # Pre-mixture A for context
    mean_n = norm_preds.mean(axis=0)
    A_5_pre = float(((norm_preds - mean_n[None, :]) ** 2).mean())
    mean_nc = norm_clipped.mean(axis=0)
    A_5_clipped = float(((norm_clipped - mean_nc[None, :]) ** 2).mean())

    mean_b = base_preds.mean(axis=0)
    A_7 = float(((base_preds - mean_b[None, :]) ** 2).mean())
    n_lirf = norm_preds.shape[1]
    n_all = base_preds.shape[1]

    # A_5 is measured on LIRF rows only (26,899), so its impact on total MSE
    # (mean over all 344,841 rows) is scaled by n_lirf / n_all.
    A_5 = A_5_post
    gain_norm = A_5 * n_lirf / n_all
    gain_base = A_7 * 4.0 / 18.0

    print(f"\n=== Ambiguity on 2026 ranking set ===")
    print(f"  A_5 raw R_norm       = {A_5_pre:.2f} MSE   (LIRF rows n={n_lirf})")
    print(f"  A_5 R_norm clipped   = {A_5_clipped:.2f} MSE")
    print(f"  A_5 post-mixture     = {A_5_post:.2f} MSE   <- section 5.1 quantity")
    print(f"  A_7 base             = {A_7:.2f} MSE   (all rows n={n_all})")
    print(f"\n  expected gain (5-member R_norm mean vs 1 member, scaled to all rows) = {gain_norm:.2f}")
    print(f"  expected gain (7-member base mean vs 3-member subset)                = {gain_base:.2f}")
    total = gain_norm + gain_base
    curr = 91192.0
    expected_rmse = float(np.sqrt(curr - total))
    print(f"\n  total expected MSE gain vs v30 = {total:.2f}")
    print(f"  v30 live MSE                   = {curr:.0f}")
    print(f"  expected v32 live RMSE         = {expected_rmse:.2f} s")

    out = {
        "A_5_raw": A_5_pre, "A_5_clipped": A_5_clipped, "A_5_post_mixture": A_5_post,
        "A_7": A_7, "gain_norm": gain_norm, "gain_base": gain_base,
        "total_gain_mse": total, "expected_rmse": expected_rmse,
        "n_lirf": int(n_lirf), "n_all": int(n_all),
        "base_seeds": BASE_SEEDS, "r_norm_seeds": R_NORM_SEEDS,
    }
    path = os.path.join(ROOT, "submission", "v32_price.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
