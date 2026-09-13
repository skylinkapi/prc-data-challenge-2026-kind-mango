"""P1. Scoring on the 2025 hold-out (months 1 and 7).

score(pred, y, adep) -> {full, clean, per_class: {class: mse}, per_airport}
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src_v2 import config as C
from src_v2.labels import classify


def score(pred: np.ndarray, y: np.ndarray, sd: np.ndarray,
          adep: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    sd = np.asarray(sd, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err2 = (pred - y) ** 2

    cls = classify(pd.Series(y), pd.Series(sd)).values
    n = len(y)

    full_rmse = float(np.sqrt(err2.mean()))
    clean_mask = (cls == "clean")
    clean_rmse = float(np.sqrt(err2[clean_mask].mean())) if clean_mask.any() else float("nan")

    per_class = {c: float(err2[cls == c].sum()) for c in ("clean", "fallback", "tail", "24h", "invalid")}
    per_class_mse = {c: (v / n) for c, v in per_class.items()}

    per_airport = defaultdict(dict)
    for a in np.unique(adep):
        for c in ("clean", "fallback", "tail", "24h"):
            m = (adep == a) & (cls == c)
            per_airport[a][c] = float(err2[m].sum() / n) if m.any() else 0.0

    return {
        "n": int(n),
        "full_rmse": full_rmse,
        "clean_rmse": clean_rmse,
        "class_mse_share": per_class_mse,
        "per_airport": dict(per_airport),
    }


def format_report(res: dict) -> str:
    lines = [
        f"n={res['n']:,}  FULL={res['full_rmse']:.3f}  CLEAN={res['clean_rmse']:.3f}",
        "class MSE (per-row share of the FULL):",
    ]
    for c, v in res["class_mse_share"].items():
        lines.append(f"  {c:10s} {v:12.2f}")
    lines.append("per-airport class MSE share:")
    lines.append(f"  {'airport':6s} {'clean':>12s} {'fallback':>12s} {'tail':>12s} {'24h':>12s}")
    for a, d in sorted(res["per_airport"].items()):
        lines.append(f"  {a:6s} {d['clean']:12.2f} {d['fallback']:12.2f} {d['tail']:12.2f} {d['24h']:12.2f}")
    return "\n".join(lines)
