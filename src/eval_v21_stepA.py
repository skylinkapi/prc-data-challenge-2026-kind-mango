"""Step B eval: v21 alone vs v21 + Step A LIRF band rule on hold-out.
Prints three numbers: clean RMSE, cell MSE contribution, full RMSE with rule.
"""
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
TRAIN_DIR = os.path.join(ROOT, "training")
HOLDOUT_MONTHS = {1, 7}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def band_key(sd, bands):
    for lo, hi in bands:
        if lo < sd <= hi:
            return f"{lo}_{hi}"
    return None


def apply_rule(taxi_hat, sd, adep, flt_id, table_data):
    """Return p_final. Applies rule only to cell (LIRF + null + sd>gate)."""
    gate = table_data["gate_sd"]
    bands = table_data["bands"]
    table = table_data["table"]
    p = taxi_hat.copy()
    is_lirf = (adep == "LIRF")
    is_null = pd.isna(flt_id)
    cell = is_lirf & is_null & (sd > gate) & ~np.isnan(sd)
    for i in np.where(cell)[0]:
        b = band_key(sd[i], bands)
        if b is None or b not in table: continue
        row = table[b]
        p_fb = row["p_fb"]
        p_24 = row["p_24h"]
        p_norm = max(0.0, 1.0 - p_fb - p_24)
        # renormalise if fb+24>1
        total = p_fb + p_24 + p_norm
        if total <= 0: continue
        p_fb, p_24, p_norm = p_fb/total, p_24/total, p_norm/total
        mean_extra = row.get("mean_24h_extra")
        if mean_extra is None or (isinstance(mean_extra, float) and np.isnan(mean_extra)):
            mean_extra = 1150.0  # empirical from bands 50k-100k
        p[i] = p_fb * sd[i] + p_24 * (86400.0 + mean_extra) + p_norm * taxi_hat[i]
    return p, cell


def main():
    t0 = time.time()
    print("Loading + features (v21 stack)...")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")

    dep = m[m["PHASE_mvt"] == "DEP"].copy()
    dep["hour"] = dep["mvt_ts"].dt.hour
    dep["dow"] = dep["mvt_ts"].dt.dayofweek
    dep["month"] = dep["mvt_ts"].dt.month
    sd = (dep["mvt_ts"] - dep["sched_ts"]).dt.total_seconds()
    dep["sched_delay"] = sd
    dep["sd_mod_86400"] = sd.mod(86400).where(sd.notna())
    dep["flt_null"] = dep["AIRCRAFT_OPERATOR_flt"].isna().astype(np.int8)
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

    print(f"Loaded hold-out n={len(test):,}   ({time.time()-t0:.1f}s)")

    b = lgb.Booster(model_file=os.path.join(MODELS, "lgbm_v21.txt"))
    with open(os.path.join(MODELS, "lgbm_v21.features.txt")) as f:
        feat = f.read().splitlines()
    taxi_hat = np.clip(b.predict(test[feat]), 0, None)
    y = test["TAXITIME_SEC_mvt"].values.astype(float)
    sd_te = test["sched_delay"].values.astype(float)
    adep_te = test["ADEP_mvt"].astype(str).values
    fltid_te = test["FLIGHT_ID_mvt"].values

    with open(os.path.join(MODELS, "lirf_band_table.json")) as f:
        table_data = json.load(f)

    p_rule, cell_mask = apply_rule(taxi_hat, sd_te, adep_te, fltid_te, table_data)

    print(f"\nCell rows on hold-out: {cell_mask.sum()}")
    print("\n=== Three-number report ===")
    clean = (y >= 30) & (y <= 7200)
    print(f"                            v21 alone    v21 + rule")
    print(f"CLEAN (30<=y<=7200)         {rmse(y[clean], taxi_hat[clean]):>8.2f}      {rmse(y[clean], p_rule[clean]):>8.2f}")
    print(f"CELL contribution MSE       {float(np.sum((y[cell_mask]-taxi_hat[cell_mask])**2)):>10.0f}    {float(np.sum((y[cell_mask]-p_rule[cell_mask])**2)):>10.0f}")
    print(f"FULL                        {rmse(y, taxi_hat):>8.2f}      {rmse(y, p_rule):>8.2f}")

    print("\nCell-row detail (first 20):")
    idx = np.where(cell_mask)[0][:20]
    show = pd.DataFrame({
        "sd": sd_te[idx].astype(int),
        "y": y[idx].astype(int),
        "v21": taxi_hat[idx].astype(int),
        "rule": p_rule[idx].astype(int),
    })
    print(show.to_string())

    print(f"\nWall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
