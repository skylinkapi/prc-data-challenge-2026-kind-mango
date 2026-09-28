"""L3 (WINNING_PLAN, MB7): CatBoost as a second base class on the v65 frame.

CatBoost reads the same 118 columns and MB3 rows as the v65 LightGBM base.
It encodes the categorical columns with ordered target statistics, fitted
the same way at training and at serving. The full-year fit takes its round
count from the hold-out fit (`measure_catboost_blend.py`), scaled by the
row ratio as in v57.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS

from src_v3 import config as C
from src_v3.build_plan_nm_taxi_v65 import OUT_TRAIN as PLAN_NM_TRAIN
from src_v3.frames import v47_feature_names
from src_v3.train_v57_base import _load_frame_v57

HOLDOUT_REPORT = C.MODELS / "catboost_blend.holdout.json"
log = logging.getLogger(__name__)


def paths(tag: str) -> tuple[Path, Path, Path]:
    """Model file, hold-out report and metadata of one CatBoost base version."""
    report = HOLDOUT_REPORT if tag == "v67" else C.MODELS / f"catboost_blend_{tag}.holdout.json"
    return (C.ROOT / "models" / f"catboost_r_all_{tag}.cbm", report,
            C.ROOT / "models" / f"catboost_r_all_{tag}.meta.json")


def load_frame(extra: tuple[Path, ...] = ()) -> tuple[pd.DataFrame, list[str]]:
    """v65 training frame on the MB3 rows plus the `extra` column files, and its feature names."""
    dep = _load_frame_v57()
    feat = v47_feature_names()
    for path in (PLAN_NM_TRAIN, *extra):
        cols = pd.read_parquet(path)
        dep = dep.merge(cols, on="MVT_ID_mvt", how="left", validate="m:1")
        feat += [c for c in cols.columns if c != "MVT_ID_mvt"]
    y = dep["TAXITIME_SEC_mvt"].astype(float)
    dep = dep[(dep["ADEP_mvt"].astype(str) != "LIRF") & (y > 0) & (y <= 80_000)]
    return dep.reset_index(drop=True), feat


def to_pool(frame: pd.DataFrame, feat: list[str], has_label: bool = True) -> Pool:
    """Pool with the categorical columns as strings; a missing category becomes 'nan'."""
    x = frame[feat].copy()
    for c in CAT_COLS:
        x[c] = x[c].astype(str)
    label = frame["TAXITIME_SEC_mvt"].values.astype(float) if has_label else None
    return Pool(x, label=label, cat_features=CAT_COLS)


def fit(train: pd.DataFrame, feat: list[str], stop: pd.DataFrame | None = None,
        iterations: int | None = None, max_rounds: int | None = None,
        params: dict | None = None) -> CatBoostRegressor:
    """Fit with early stop on `stop` up to `max_rounds`, or for a fixed round count."""
    params = dict(C.CATBOOST_PARAMS if params is None else params)
    if max_rounds is not None:
        params.update(iterations=max_rounds)
    if iterations is not None:
        params.update(iterations=iterations, od_type=None, od_wait=None)
    m = CatBoostRegressor(**{k: v for k, v in params.items() if v is not None})
    m.fit(to_pool(train, feat), eval_set=None if stop is None else to_pool(stop, feat),
          use_best_model=stop is not None)
    return m


def predict(model: CatBoostRegressor, frame: pd.DataFrame, feat: list[str],
            ntree_end: int = 0, clip: bool = True) -> np.ndarray:
    """Prediction clipped at 0 unless `clip` is False, as the LightGBM members are served; `ntree_end` 0 uses every tree."""
    raw = model.predict(to_pool(frame, feat, has_label=False), ntree_end=ntree_end)
    return np.clip(raw, 0, None) if clip else raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v67", help="Version tag of the model and hold-out report.")
    ap.add_argument("--rounds-from", help="Tag of the hold-out report that sets the round count; default is --tag.")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Parquets of extra training columns keyed on MVT_ID_mvt, relative to the repo root.")
    ap.add_argument("--anchor-col", help="Learn `y - clip(anchor, 0, NM_CLOCK_CLIP_S)` on the rows with this column (v75: mvt_aobt3).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model_path, _, meta_path = paths(args.tag)
    report = json.loads(paths(args.rounds_from or args.tag)[1].read_text())
    dep, feat = load_frame(tuple(C.ROOT / p for p in args.extra))
    if args.anchor_col:
        dep = dep[dep[args.anchor_col].notna()].reset_index(drop=True)
        dep["TAXITIME_SEC_mvt"] = dep["TAXITIME_SEC_mvt"] - dep[args.anchor_col].clip(0, C.NM_CLOCK_CLIP_S)
    n_iter = int(math.ceil(report["best_iteration"] * len(dep) / report["n_fit"]))
    log.info("full-year CatBoost %s: %d rows, %d rounds", args.tag, len(dep), n_iter)
    params = C.CATBOOST_BY_TAG[args.tag]
    fit(dep, feat, iterations=n_iter, params=params).save_model(str(model_path))
    meta_path.write_text(json.dumps({
        "params": params, "iterations": n_iter, "n_rows": len(dep),
        "features": feat, "anchor": args.anchor_col}, indent=1))


if __name__ == "__main__":
    main()
