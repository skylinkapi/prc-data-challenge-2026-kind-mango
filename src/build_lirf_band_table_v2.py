"""Step A variant: extend rule down to sd > 3,600 s.
Same as build_lirf_band_table but with GATE_SD = 3,600.
"""
import glob, os, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training")
MODELS = os.path.join(ROOT, "models")

BANDS = [(3600, 7200), (7200, 14400), (14400, 25000), (25000, 40000),
         (40000, 50000), (50000, 70000), (70000, 100000), (100000, 10**9)]
GATE_SD = 3600


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
    cell["is_24h"] = cell["y"] > 80000
    cell["is_normal"] = ~cell["is_fb"] & ~cell["is_24h"]
    cell["band"] = cell["sd"].apply(band_key)
    print(f"Cell rows (LIRF + null + sd>{GATE_SD}): {len(cell):,}")

    print(f"\n=== Extended band table (gate {GATE_SD}) ===")
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
        table[band_id] = {"n": int(len(sub)), "p_fb": p_fb, "p_24h": p_24, "p_norm": p_norm,
                          "mean_norm": mean_norm, "mean_24h_extra": mean_24h_extra}
        print(f"  {band_id:15s} n={len(sub):>5}  P_fb={p_fb:.3f}  P_24h={p_24:.3f}  "
              f"P_norm={p_norm:.3f}  mean_norm={mean_norm if not np.isnan(mean_norm) else 'nan':>7}")

    out = os.path.join(MODELS, "lirf_band_table_v2.json")
    with open(out, "w") as f:
        json.dump({"gate_sd": GATE_SD, "bands": BANDS, "table": table}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
