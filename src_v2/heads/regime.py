"""C1. Per-airport fallback classifier.

For each airport in FALLBACK_AIRPORTS: train a binary LightGBM on
|y - sd| <= FB_TOL vs. clean, on fit months minus the stop months. Fit an
isotonic calibrator on the stop months. Serve calibrated probability.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src_v2 import config as C
from src_v2 import features as F
from src_v2 import labels as L
from src_v2.train.base import STOP_MONTHS, _categorify, build_vocab

log = logging.getLogger(__name__)


def _training_frame(dep: pd.DataFrame, airport: str) -> pd.DataFrame:
    d = dep[(dep["ADEP_mvt"] == airport) & dep["month"].isin(C.FIT_MONTHS)].copy()
    cls = L.classify(d["TAXITIME_SEC_mvt"], d["sd"]).values
    keep = np.isin(cls, ["clean", "fallback"])
    d = d.loc[keep].reset_index(drop=True)
    d["y_fb"] = (L.classify(d["TAXITIME_SEC_mvt"], d["sd"]).values == "fallback").astype("int8")
    return d


def train(dep: pd.DataFrame) -> dict:
    feats = F.regime_feature_list()
    cats = [c for c in ("ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "flt_prefix",
                        "stand_prefix", "AIRCRAFT_TYPE_mvt") if c in feats]
    vocab = build_vocab(dep)
    out_summary = {}
    for airport in C.FALLBACK_AIRPORTS:
        d = _training_frame(dep, airport)
        if len(d) < 500:
            log.warning("regime %s: not enough rows (%d), skipping", airport, len(d))
            continue
        d = _categorify(d, vocab)
        stop = d["month"].isin(STOP_MONTHS)
        X_fit, y_fit = d.loc[~stop, feats], d.loc[~stop, "y_fb"].values
        X_stop, y_stop = d.loc[stop, feats], d.loc[stop, "y_fb"].values
        cat_idx = [X_fit.columns.get_loc(c) for c in cats]

        params = dict(C.LGB_REGIME, seed=C.SEED, bagging_seed=C.SEED)
        d_fit = lgb.Dataset(X_fit, y_fit, categorical_feature=cat_idx, free_raw_data=False)
        d_stop = lgb.Dataset(X_stop, y_stop, categorical_feature=cat_idx,
                             reference=d_fit, free_raw_data=False)
        booster = lgb.train(params, d_fit, num_boost_round=1500,
                            valid_sets=[d_stop], valid_names=["stop"],
                            callbacks=[lgb.early_stopping(80, verbose=False),
                                       lgb.log_evaluation(0)])
        raw_stop = booster.predict(X_stop)
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_stop, y_stop)

        booster.save_model(str(C.MODELS / f"regime_{airport}.txt"))
        with open(C.MODELS / f"regime_{airport}.iso.pkl", "wb") as f:
            pickle.dump(iso, f)
        out_summary[airport] = {"rows": int(len(d)),
                                "best_iter": int(booster.best_iteration),
                                "pos_rate": float(y_fit.mean())}
        log.info("regime %s trained: rows=%d best_iter=%d pos=%.3f",
                 airport, len(d), booster.best_iteration, y_fit.mean())

    with open(C.MODELS / "regime.features.json", "w") as f:
        json.dump({"features": feats, "cat": cats, "airports": list(out_summary),
                   "vocab": {c: [str(v) for v in vocab[c]] for c in vocab}}, f)
    with open(C.MODELS / "regime.vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    return out_summary


def predict(dep: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (calibrated p_fb, mask_head_fired) per row."""
    with open(C.MODELS / "regime.features.json") as f:
        meta = json.load(f)
    with open(C.MODELS / "regime.vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    d = _categorify(dep.copy(), vocab)
    feats = meta["features"]
    p = np.zeros(len(dep))
    fired = np.zeros(len(dep), dtype=bool)
    for airport in meta["airports"]:
        mask = (dep["ADEP_mvt"].values == airport)
        if not mask.any():
            continue
        b = lgb.Booster(model_file=str(C.MODELS / f"regime_{airport}.txt"))
        with open(C.MODELS / f"regime_{airport}.iso.pkl", "rb") as f:
            iso = pickle.load(f)
        raw = b.predict(d.loc[mask, feats])
        p[mask] = iso.transform(raw)
        fired[mask] = True
    return p, fired
