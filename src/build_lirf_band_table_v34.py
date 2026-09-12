"""v34 band table + constants — audit Item 2. Fit months only.

Rebuilds the LIRF Step A band table and the two mixture constants
(NORMAL_MEAN_LIRF, R_NORM_CLIP) on months 2..12 minus {1, 7}. Excludes the
hold-out months so the hold-out remains an honest evaluator (F2 in the
eighth audit).
"""
import glob, os, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")

HOLDOUT_MONTHS = {1, 7}
FIT_MONTHS = {2, 3, 4, 5, 6, 8, 9, 10, 11, 12}

BANDS = [(14400, 16000), (16000, 18000), (18000, 20000), (20000, 22000),
         (22000, 25000), (25000, 40000), (40000, 50000), (50000, 60000),
         (60000, 70000), (70000, 100000), (100000, 10**9)]
GATE_SD = 14400
FB_TOL = 60


def band_key(sd, bands=BANDS):
    for lo, hi in bands:
        if lo < sd <= hi:
            return f"{lo}_{hi}"
    return None


def load_lirf_fit_months():
    frames = []
    for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.parquet"))):
        t = pd.read_parquet(f, columns=["ADEP_mvt", "PHASE_mvt", "FLIGHT_ID_mvt",
                                        "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                        "TAXITIME_SEC_mvt"])
        t = t[(t["PHASE_mvt"] == "DEP") & (t["ADEP_mvt"] == "LIRF")]
        frames.append(t)
    m = pd.concat(frames, ignore_index=True)
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    m["month"] = m["mvt_ts"].dt.month
    m["sd"] = (m["mvt_ts"] - m["sched_ts"]).dt.total_seconds()
    m["y"] = m["TAXITIME_SEC_mvt"].astype(float)
    m["null_flt"] = m["FLIGHT_ID_mvt"].isna()
    return m[m["month"].isin(FIT_MONTHS)].copy()


def build_band_table(m):
    cell = m[m["null_flt"] & (m["sd"] > GATE_SD) & m["y"].notna()].copy()
    cell["is_fb"] = (cell["y"] - cell["sd"]).abs() < FB_TOL
    cell["is_24h"] = (cell["y"] > 80000) & ~cell["is_fb"]
    cell["is_normal"] = ~cell["is_fb"] & ~cell["is_24h"]
    cell["band"] = cell["sd"].apply(band_key)
    print(f"Cell rows (LIRF + null + sd>{GATE_SD}, fit months): {len(cell):,}")

    table = {}
    for band_id in [f"{lo}_{hi}" for lo, hi in BANDS]:
        sub = cell[cell["band"] == band_id]
        if len(sub) == 0: continue
        p_fb = float(sub["is_fb"].mean())
        p_24 = float(sub["is_24h"].mean())
        p_norm = float(sub["is_normal"].mean())
        norm_rows = sub[sub["is_normal"]]
        h24_rows = sub[sub["is_24h"]]
        mean_norm = float(norm_rows["y"].mean()) if len(norm_rows) else np.nan
        mean_24h_extra = float((h24_rows["y"] - 86400).mean()) if len(h24_rows) else np.nan
        table[band_id] = {"n": int(len(sub)), "p_fb": p_fb, "p_24h": p_24,
                          "p_norm": p_norm, "mean_norm": mean_norm,
                          "mean_24h_extra": mean_24h_extra}
        print(f"  {band_id:20s} n={len(sub):>4}  P_fb={p_fb:.3f}  "
              f"P_24h={p_24:.3f}  mean_y={float(sub['y'].mean()):>7.0f}")
    return table


def build_constants(m):
    """Constants match the recipe used at predict time.

    NORMAL_MEAN_LIRF: mean of LIRF genuine taxi (|y-sd|>=FB_TOL and y<80000
    and y>0). Fed to ITY340 as the normal-regime taxi expectation.

    R_NORM_CLIP: 99.9th percentile of the same population. Bounds the
    R_norm_LIRF output (M5/M5-inheritor in the audit).
    """
    g = m[m["y"] > 0].copy()
    g = g[(np.abs(g["y"] - g["sd"]) >= FB_TOL) & (g["y"] < 80000)]
    print(f"Genuine LIRF rows on fit months: {len(g):,}")
    normal_mean = float(g["y"].mean())
    r_norm_clip = float(np.quantile(g["y"], 0.999))
    return normal_mean, r_norm_clip


def main():
    m = load_lirf_fit_months()
    print(f"LIRF rows on fit months {sorted(FIT_MONTHS)}: {len(m):,}")

    table = build_band_table(m)
    normal_mean, r_norm_clip = build_constants(m)

    print(f"\nNORMAL_MEAN_LIRF = {normal_mean:.2f}")
    print(f"R_NORM_CLIP      = {r_norm_clip:.2f}")

    band_out = os.path.join(MODELS, "lirf_band_table_v34.json")
    with open(band_out, "w") as f:
        json.dump({"gate_sd": GATE_SD, "bands": BANDS, "table": table,
                   "fit_months": sorted(FIT_MONTHS)}, f, indent=2)
    print(f"Saved -> {band_out}")

    const_out = os.path.join(MODELS, "v34_constants.json")
    with open(const_out, "w") as f:
        json.dump({"NORMAL_MEAN_LIRF": normal_mean,
                   "R_NORM_CLIP": r_norm_clip,
                   "fit_months": sorted(FIT_MONTHS),
                   "fb_tol": FB_TOL,
                   "gate_sd": GATE_SD}, f, indent=2)
    print(f"Saved -> {const_out}")


if __name__ == "__main__":
    main()
