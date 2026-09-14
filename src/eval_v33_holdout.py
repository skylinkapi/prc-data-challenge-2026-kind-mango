"""Score any stack end to end on the 2025 hold-out months 1 and 7.

H2 of the fourteenth pass: the v33 hold-out script generalised to any stack.
The defaults reproduce models/v33.holdout.json; --stack picks a named stack
(v33, v40, v41) and the --r-norm-* and --base-* overrides describe a lever.
The base reads the H1 frame cache; the LIRF frame from
train_lirf_regime.build_features is cached at models/lirf_frame_cache.parquet
(pass --rebuild-lirf after a change to a feature function).

Non-LIRF rows: the base members on the H1 cache. LIRF rows: the R_norm
members, the v23 p_fb gate, the v30 band table and the ITY340 rule, exactly as
predict_v30.main serves them. The rate maps and the band table read all 12
months, so the LIRF numbers are optimistic (eleventh pass F5).
"""
import argparse
import json
import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_h1_frame_cache import load_frame as load_h1_frame
from predict_v23 import add_rate_features_scoring, apply_stepA_v22
from predict_v30 import ITY_SD_THRESHOLD, NORMAL_MEAN_LIRF, P24_ITY, R_NORM_CLIP
from train_lgbm_v21 import CAT_COLS
from train_lirf_regime import HOLDOUT_MONTHS, MODELS, build_features
from train_r_all_v40 import TEMPO_COLS, V2_FRAME, class_table, classify

R_NORM_FILES = [f"lgbm_r_norm_lirf_s{s}.txt" for s in (42, 43, 44, 45, 46)]
LIRF_CACHE = os.path.join(MODELS, "lirf_frame_cache.parquet")
STACKS = {
    "v33": {"base_model": "lgbm_r_all_v26", "base_seeds": [42, 43, 44],
            "r_norm_files": R_NORM_FILES, "r_norm_features": "lirf_regime.features.txt",
            "r_norm_clip": R_NORM_CLIP},
    "v40": {"base_model": "lgbm_r_all_v40", "base_seeds": [42, 43, 44],
            "r_norm_files": R_NORM_FILES, "r_norm_features": "lirf_regime.features.txt",
            "r_norm_clip": R_NORM_CLIP},
    "v41": {"base_model": "lgbm_r_all_v40", "base_seeds": [42, 43, 44],
            "r_norm_files": [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in (42, 43, 44, 45, 46)],
            "r_norm_features": "lirf_regime_v41.features.txt", "r_norm_clip": None},
}
log = logging.getLogger(__name__)


def read_list(name: str) -> list[str]:
    with open(os.path.join(MODELS, name)) as f:
        return f.read().splitlines()


def lirf_frame(rebuild: bool = False) -> pd.DataFrame:
    """build_features output for LIRF, cached on disk."""
    if not rebuild and os.path.exists(LIRF_CACHE):
        return pd.read_parquet(LIRF_CACHE)
    dep, _ = build_features()
    dep.to_parquet(LIRF_CACHE)
    log.info("rebuilt the LIRF frame cache: %d rows", len(dep))
    return dep


def lirf_head(dep: pd.DataFrame, members: list[str] = R_NORM_FILES,
              feat_norm: list[str] | None = None,
              r_norm_clip: float | None = R_NORM_CLIP) -> np.ndarray:
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
    r_norm = np.mean(preds, axis=0)
    if r_norm_clip is not None:
        r_norm = np.minimum(r_norm, r_norm_clip)
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


def base_non_lirf(model: str, seeds: list[int]) -> pd.DataFrame:
    """Base member mean on the cached hold-out rows outside LIRF."""
    feat_file = os.path.join(MODELS, f"{model}.features.txt")
    dep, feat = load_h1_frame()
    if os.path.exists(feat_file):
        feat = read_list(f"{model}.features.txt")
    t = dep[dep["month"].isin(HOLDOUT_MONTHS) & (dep["ADEP_mvt"].astype(str) != "LIRF")]
    pred = np.mean([np.clip(lgb.Booster(model_file=os.path.join(MODELS, f"{model}_s{s}.txt"))
                            .predict(t[feat]), 0, None) for s in seeds], axis=0)
    return pd.DataFrame({"y": t["TAXITIME_SEC_mvt"].values.astype(float),
                         "sd": t["sched_delay"].values.astype(float),
                         "apt": t["ADEP_mvt"].astype(str).values, "pred": pred})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack", default="v33", choices=sorted(STACKS))
    ap.add_argument("--base-model")
    ap.add_argument("--base-seeds", nargs="+", type=int)
    ap.add_argument("--r-norm-files", nargs="+")
    ap.add_argument("--r-norm-features")
    ap.add_argument("--r-norm-clip", help="seconds, or 'none'")
    ap.add_argument("--rebuild-lirf", action="store_true")
    ap.add_argument("--tag", help="output tag; default is the stack name")
    ap.add_argument("--out")
    return ap.parse_args()


def resolve_stack(args: argparse.Namespace) -> dict:
    spec = dict(STACKS[args.stack])
    for arg, key in (("base_model", "base_model"), ("base_seeds", "base_seeds"),
                     ("r_norm_files", "r_norm_files"), ("r_norm_features", "r_norm_features")):
        if getattr(args, arg) is not None:
            spec[key] = getattr(args, arg)
    if args.r_norm_clip is not None:
        spec["r_norm_clip"] = None if args.r_norm_clip == "none" else float(args.r_norm_clip)
    return spec


def main() -> None:
    args = parse_args()
    spec = resolve_stack(args)
    log.info("stack %s", spec)
    lirf = lirf_frame(args.rebuild_lirf)
    lirf = lirf[lirf["month"].isin(HOLDOUT_MONTHS)].copy()
    feat_norm = read_list(spec["r_norm_features"])
    missing = [c for c in feat_norm if c not in lirf.columns]
    if missing:
        extra = pd.read_parquet(V2_FRAME, columns=["MVT_ID_mvt", *missing])
        lirf = lirf.merge(extra, on="MVT_ID_mvt", how="left", validate="1:1")
        log.info("joined %d columns from the src_v2 frame", len(missing))
    lirf_pred = lirf_head(lirf, spec["r_norm_files"], feat_norm, spec["r_norm_clip"])
    rows = pd.concat([base_non_lirf(spec["base_model"], spec["base_seeds"]),
                      pd.DataFrame({"y": lirf["TAXITIME_SEC_mvt"].values.astype(float),
                                    "sd": lirf["sched_delay"].values.astype(float),
                                    "apt": "LIRF", "pred": lirf_pred})], ignore_index=True)
    cls = classify(rows["y"].values, rows["sd"].values)
    report = class_table(rows["pred"].values, rows["y"].values, cls, rows["apt"].values)
    report["n"] = int(len(rows))
    report["stack"] = spec
    tag = args.tag or args.stack
    out = args.out or os.path.join(MODELS, f"{tag}.holdout.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    log.info("%s hold-out: n %d FULL %.2f CLEAN %.2f class %s -> %s", tag, report["n"],
             report["full_rmse"], report["clean_rmse"],
             {k: round(v) for k, v in report["class_mse"].items()}, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
