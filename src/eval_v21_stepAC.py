"""Eval: v21 alone vs v21 + Step A rule vs v21 + Step A rule + Step C detector."""
import glob, os, pickle, json, gc, time
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
from train_fallback_detector import CAT_INPUTS as DET_CAT, NUM_INPUTS as DET_NUM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    t0 = time.time()
    print("Loading + features...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    dep["sched_sec"] = dep["sched_ts"].dt.second
    dep["mvt_sec"] = dep["mvt_ts"].dt.second
    sd = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds()
    dep["sched_delay"] = sd
    dep["sd"] = sd
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
    dep["flt_id_null"] = dep["FLIGHT_ID_mvt"].isna().astype(np.int8)
    dep = add_obt_features(dep)
    dep = dep[dep["TAXITIME_SEC_mvt"].astype(float) > 0]

    m_ctx = m[["ADEP_mvt","PHASE_mvt","mvt_ts","sched_ts","RUNWAY_mvt"]].copy()
    del m; gc.collect()

    dep = add_weather(dep)
    dep = add_congestion(dep, m_ctx[["ADEP_mvt","PHASE_mvt","mvt_ts","RUNWAY_mvt"]])
    dep = add_eurocontrol(dep)
    with open(os.path.join(MODELS, "lgbm_v21.encoders.pkl"), "rb") as f:
        enc = pickle.load(f)
    dep = apply_encoders(dep, enc)
    dep = add_taxi_distance(dep)
    ec_daily = pd.read_parquet(os.path.join(ROOT, "external", "eurocontrol", "daily_features.parquet"))
    dep = add_advanced(dep, m_ctx, daily_ec=ec_daily)
    del m_ctx, ec_daily; gc.collect()
    dep = add_osm_path(dep)
    dep = add_opdi(dep)
    dep = add_opdi_live(dep)
    gc.collect()

    hold = dep["month"].isin(HOLDOUT_MONTHS)
    train_cats = {c: dep.loc[~hold, c].astype("category").cat.categories for c in CAT_COLS}
    test = dep.loc[hold].copy()
    for c in CAT_COLS:
        test[c] = pd.Categorical(test[c], categories=train_cats[c])
    del dep; gc.collect()

    # v21 baseline
    b = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        feat = f.read().splitlines()
    taxi_hat = np.clip(b.predict(test[feat]), 0, None)
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)
    adep_te = test["ADEP_mvt"].astype(str).values
    fltid_te = test["FLIGHT_ID_mvt"].values

    # Step A rule
    with open(os.path.join(MODELS, "lirf_band_table.json")) as f:
        table_data = json.load(f)
    p_A, cell_A = apply_rule(taxi_hat, sd_te, adep_te, fltid_te, table_data)

    # Step C detector (LIRF + EGLL). Soft mix: p_final = p_fb*sd + (1-p_fb)*taxi_hat_A
    # But NOT on cells already handled by Step A
    p_AC = p_A.copy()
    p_fb_full = np.zeros(len(y))
    for apt in ["LIRF", "EGLL"]:
        det_path = os.path.join(MODELS, f"fallback_detector_{apt}.txt")
        if not os.path.exists(det_path): continue
        det = lgb.Booster(model_file=det_path)
        # Need detector inputs on the full test set
        sub_mask = (adep_te == apt) & (sd_te > 3600) & ~cell_A  # don't overlap Step A
        if sub_mask.sum() == 0: continue
        sub = test.loc[sub_mask].copy()
        # Categorical alignment: use categories the detector was trained on. We
        # cannot recover those exactly here, so re-cast on the current train_cats.
        for c in DET_CAT:
            if c in sub.columns:
                sub[c] = sub[c].astype("category")
        # Numeric: some names differ (sched_delay vs sd)
        sub["sd"] = sub["sched_delay"]
        det_feat = DET_CAT + DET_NUM
        p_fb = det.predict(sub[det_feat])
        p_fb_full[np.where(sub_mask)[0]] = p_fb
        # Soft mixture; only apply where p_fb > 0.5 to guard against low-conf blends
        # Actually the doc says "soft mixture" period; try no threshold first
        p_AC[sub_mask] = p_fb * sd_te[sub_mask] + (1 - p_fb) * p_A[sub_mask]
        print(f"  {apt}: applied to {sub_mask.sum()} rows,  mean p_fb {p_fb.mean():.3f}")

    print("\n=== Hold-out comparison ===")
    print(f"                          v21 alone    +ruleA (Step A)    +ruleA+detC (Step A+C)")
    clean = (y >= 30) & (y <= 7200)
    print(f"CLEAN         {rmse(y[clean], taxi_hat[clean]):>8.2f}          {rmse(y[clean], p_A[clean]):>8.2f}                {rmse(y[clean], p_AC[clean]):>8.2f}")
    print(f"FULL          {rmse(y, taxi_hat):>8.2f}          {rmse(y, p_A):>8.2f}                {rmse(y, p_AC):>8.2f}")

    print("\nPer airport (full, +A+C - +A delta):")
    for apt in ["LIRF", "EGLL"]:
        mask = (adep_te == apt) & (sd_te > 3600) & ~cell_A
        if mask.sum() == 0: continue
        r_A = rmse(y[mask], p_A[mask])
        r_AC = rmse(y[mask], p_AC[mask])
        print(f"  {apt} sd>3600 non-cell (n={mask.sum()}):  +A {r_A:.2f}   +A+C {r_AC:.2f}   delta {r_AC-r_A:+.2f}")

    # Sweep threshold: only mix when p_fb > threshold
    print("\nSweep detector threshold (soft mix only when p_fb > thr):")
    for thr in [0.0, 0.3, 0.5, 0.7, 0.9]:
        p = p_A.copy()
        m2 = (p_fb_full > thr) & ~cell_A
        if m2.sum() > 0:
            p[m2] = p_fb_full[m2] * sd_te[m2] + (1 - p_fb_full[m2]) * p_A[m2]
        print(f"  thr={thr}: n_apply={m2.sum():>5}  clean {rmse(y[clean], p[clean]):.2f}  full {rmse(y, p):.2f}")

    print(f"\nWall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
