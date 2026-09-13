"""C2 constant-leaf, C3 clean-only base regressor with 3 seeds (C5).

Objective: L2 (RMSE). Trains on clean rows only, months 2..6 and 8..12.
Uses months 11 and 12 as the blocked stop set (P3-compatible: stop months
are inside the fit months, hold-out months 1 and 7 never enter here).
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src_v2 import config as C
from src_v2 import features as F
from src_v2 import labels as L

log = logging.getLogger(__name__)

STOP_MONTHS = (11, 12)


def _categorify(df: pd.DataFrame, vocab: dict[str, list]) -> pd.DataFrame:
    for c in F.CAT_COLS:
        df[c] = pd.Categorical(df[c], categories=vocab[c])
    for c in df.columns:
        if c in F.CAT_COLS:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_extension_array_dtype(s):
            df[c] = s.astype("float32")
    return df


def build_vocab(dep: pd.DataFrame) -> dict[str, list]:
    return {c: sorted(dep[c].dropna().unique().tolist()) for c in F.CAT_COLS}


def train(dep: pd.DataFrame) -> dict:
    fit = dep[dep["month"].isin(C.FIT_MONTHS)].copy()
    cls = L.classify(fit["TAXITIME_SEC_mvt"], fit["sd"]).values
    clean = cls == "clean"
    fit = fit.loc[clean].reset_index(drop=True)
    log.info("base fit rows (clean, fit months): %d", len(fit))

    vocab = build_vocab(fit)
    fit = _categorify(fit, vocab)

    feats = F.base_feature_list()
    stop_mask = fit["month"].isin(STOP_MONTHS)
    X_fit = fit.loc[~stop_mask, feats]
    y_fit = fit.loc[~stop_mask, "TAXITIME_SEC_mvt"].astype("float64").values
    X_stop = fit.loc[stop_mask, feats]
    y_stop = fit.loc[stop_mask, "TAXITIME_SEC_mvt"].astype("float64").values

    cat_idx = [X_fit.columns.get_loc(c) for c in F.CAT_COLS]

    boosters: list[lgb.Booster] = []
    best_iters: list[int] = []
    for seed in C.SEEDS_ENSEMBLE:
        params = dict(C.LGB_BASE, seed=seed, bagging_seed=seed,
                      feature_fraction_seed=seed)
        d_fit = lgb.Dataset(X_fit, y_fit, categorical_feature=cat_idx, free_raw_data=False)
        d_stop = lgb.Dataset(X_stop, y_stop, categorical_feature=cat_idx,
                             reference=d_fit, free_raw_data=False)
        booster = lgb.train(
            params, d_fit, num_boost_round=5000,
            valid_sets=[d_stop], valid_names=["stop"],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(200)],
        )
        boosters.append(booster)
        best_iters.append(booster.best_iteration)
        booster.save_model(str(C.MODELS / f"base_s{seed}.txt"))
        log.info("seed %d best iter %d (saved)", seed, booster.best_iteration)
    with open(C.MODELS / "base.features.json", "w") as f:
        json.dump({"features": feats, "cat": list(F.CAT_COLS),
                   "vocab": {c: [str(v) for v in vocab[c]] for c in vocab}}, f)
    with open(C.MODELS / "base.vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    return {"seeds": list(C.SEEDS_ENSEMBLE), "best_iters": best_iters}


def load() -> tuple[list[lgb.Booster], list[str], dict[str, list]]:
    with open(C.MODELS / "base.features.json") as f:
        meta = json.load(f)
    boosters = [lgb.Booster(model_file=str(C.MODELS / f"base_s{s}.txt"))
                for s in C.SEEDS_ENSEMBLE]
    with open(C.MODELS / "base.vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return boosters, meta["features"], vocab


def predict(dep: pd.DataFrame) -> np.ndarray:
    boosters, feats, vocab = load()
    dep = _categorify(dep.copy(), vocab)
    preds = np.mean([b.predict(dep[feats]) for b in boosters], axis=0)
    return preds
