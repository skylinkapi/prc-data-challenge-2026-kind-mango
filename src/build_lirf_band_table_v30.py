"""Refined Step A band table (v30): the v22 bands with mutually exclusive classes.

A row with abs(y - sd) < 60 and y > 80,000 counts as fallback only, because y = sd
already explains it. The v22 builder counted it in both classes, which set the top
band to P(fb) = P(24h) = 1 and drove NOSOS431 to 121,410 s. MODEL_ANALYSIS.md
section 3 gives the measurement.
"""
import glob, os, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")

# Refined bands from doc appendix
BANDS = [(14400, 16000), (16000, 18000), (18000, 20000), (20000, 22000),
         (22000, 25000), (25000, 40000), (40000, 50000), (50000, 60000),
         (60000, 70000), (70000, 100000), (100000, 10**9)]
GATE_SD = 14400


def band_key(sd, bands=BANDS):
    for lo, hi in bands:
        if lo < sd <= hi:
            return f"{lo}_{hi}"
    return None


def main():
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
    m["sd"] = (m["mvt_ts"] - m["sched_ts"]).dt.total_seconds()
    m["y"] = m["TAXITIME_SEC_mvt"].astype(float)
    m["null_flt"] = m["FLIGHT_ID_mvt"].isna()

    cell = m[m["null_flt"] & (m["sd"] > GATE_SD) & m["y"].notna()].copy()
    cell["is_fb"] = (cell["y"] - cell["sd"]).abs() < 60
    cell["is_24h"] = (cell["y"] > 80000) & ~cell["is_fb"]
    cell["is_normal"] = ~cell["is_fb"] & ~cell["is_24h"]
    cell["band"] = cell["sd"].apply(band_key)
    print(f"Cell rows (LIRF + null + sd>{GATE_SD}): {len(cell):,}")

    print(f"\n=== v30 band table ===")
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

    out = os.path.join(MODELS, "lirf_band_table_v30.json")
    with open(out, "w") as f:
        json.dump({"gate_sd": GATE_SD, "bands": BANDS, "table": table}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
