"""Score the v33 stack end to end on the 2025 hold-out months 1 and 7.

Non-LIRF rows: the 3 v26 members from the cached 97-column frame. LIRF rows:
the 5 R_norm members, the v23 p_fb gate, the v30 band table and the ITY340
rule, exactly as predict_v30.main serves them. The rate maps and the band
table read all 12 months, so the LIRF numbers are optimistic (eleventh pass F5).
Writes models/v33.holdout.json with the per-class table.
"""
import json
import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from predict_v23 import add_rate_features_scoring, apply_stepA_v22
from predict_v30 import ITY_SD_THRESHOLD, NORMAL_MEAN_LIRF, P24_ITY, R_NORM_CLIP
from train_lgbm_v21 import CAT_COLS
from train_lirf_regime import HOLDOUT_MONTHS, MODELS, build_features
from train_r_all_v40 import CACHE, class_table, classify

R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in (42, 43, 44, 45, 46)]
log = logging.getLogger(__name__)


def read_list(name: str) -> list[str]:
    with open(os.path.join(MODELS, name)) as f:
        return f.read().splitlines()


def lirf_head(dep: pd.DataFrame, members: list[str] = R_NORM_FILES,
              feat_norm: list[str] | None = None) -> np.ndarray:
    """v33 LIRF prediction: mixture, band table and ITY340 on one LIRF frame."""
    with open(os.path.join(MODELS, "lirf_regime_v23.rate_maps.pkl"), "rb") as f:
        bundle = pickle.load(f)
    dep = add_rate_features_scoring(dep, bundle["maps"], bundle["base_rate"])
    cats = {c: pd.Index(dep[c].dropna().unique()) for c in CAT_COLS}
    for c in CAT_COLS:
        dep[c] = pd.Categorical(dep[c], categories=cats[c])
    feat_norm = feat_norm or read_list("lirf_regime.features.txt")
    preds = [np.clip(lgb.Booster(model_file=os.path.join(MODELS, fn)).predict(dep[feat_norm]), 0, None)
             for fn in members]
    r_norm = np.minimum(np.mean(preds, axis=0), R_NORM_CLIP)
    with open(os.path.join(MODELS, "lirf_regime_v23.isotonic.pkl"), "rb") as f:
        iso = pickle.load(f)
    p_fb = iso.transform(lgb.Booster(model_file=os.path.join(MODELS, "lgbm_p_fb_lirf_v23.txt"))
                         .predict(dep[read_list("lirf_regime_v23.features.txt")]))
    sd = dep["sched_delay"].values.astype(float)
    valid = ~np.isnan(sd)
    pred = np.where(valid, p_fb * sd + (1 - p_fb) * r_norm, r_norm)
    with open(os.path.join(MODELS, "lirf_band_table_v30.json")) as f:
        band = json.load(f)
    adep = dep["ADEP_mvt"].astype(str).values
    pred, cell = apply_stepA_v22(pred, sd, adep, dep["FLIGHT_ID_mvt"].values, band)
    ity = (sd > ITY_SD_THRESHOLD) & valid & ~cell
    pred[ity] = P24_ITY * (86400 + NORMAL_MEAN_LIRF) + (1 - P24_ITY) * sd[ity]
    return np.clip(pred, 0, None)


def base_non_lirf() -> pd.DataFrame:
    """v26 3-member mean on the cached hold-out rows outside LIRF."""
    feat = read_list("v36_tune_cache.parquet.feat.txt")
    t = pd.read_parquet(CACHE)
    t = t[t["month"].isin(HOLDOUT_MONTHS) & (t["ADEP_mvt"].astype(str) != "LIRF")]
    pred = np.mean([np.clip(lgb.Booster(model_file=os.path.join(MODELS, f"lgbm_r_all_v26_s{s}.txt"))
                            .predict(t[feat]), 0, None) for s in (42, 43, 44)], axis=0)
    return pd.DataFrame({"y": t["TAXITIME_SEC_mvt"].values.astype(float),
                         "sd": t["sched_delay"].values.astype(float),
                         "apt": t["ADEP_mvt"].astype(str).values, "pred": pred})


def main() -> None:
    lirf, _ = build_features()
    lirf = lirf[lirf["month"].isin(HOLDOUT_MONTHS)].copy()
    lirf_pred = lirf_head(lirf)
    rows = pd.concat([base_non_lirf(),
                      pd.DataFrame({"y": lirf["TAXITIME_SEC_mvt"].values.astype(float),
                                    "sd": lirf["sched_delay"].values.astype(float),
                                    "apt": "LIRF", "pred": lirf_pred})], ignore_index=True)
    cls = classify(rows["y"].values, rows["sd"].values)
    report = class_table(rows["pred"].values, rows["y"].values, cls, rows["apt"].values)
    report["n"] = int(len(rows))
    with open(os.path.join(MODELS, "v33.holdout.json"), "w") as f:
        json.dump(report, f, indent=1)
    log.info("v33 stack hold-out: n %d FULL %.2f CLEAN %.2f class %s", report["n"],
             report["full_rmse"], report["clean_rmse"], {k: round(v) for k, v in report["class_mse"].items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
