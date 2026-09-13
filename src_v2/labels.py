"""Label classes (A1). Read from src_v2/config.py in every script."""
import numpy as np
import pandas as pd

from src_v2 import config as C


def classify(y: pd.Series, sd: pd.Series) -> pd.Series:
    """Return one of clean / fallback / tail / 24h / invalid per row (A1)."""
    y = y.astype("float64")
    sd = sd.astype("float64")
    fb = (y - sd).abs() <= C.FB_TOL
    is_24h = (y > C.Y_24H_MIN) & ~fb
    is_tail = (y > C.Y_CLEAN_MAX) & (y <= C.Y_TAIL_MAX) & ~fb
    is_clean = (y >= C.Y_MIN) & (y <= C.Y_CLEAN_MAX) & ~fb
    out = pd.Series("invalid", index=y.index, dtype=object)
    out[is_clean] = "clean"
    out[fb & (y > 0)] = "fallback"
    out[is_tail] = "tail"
    out[is_24h] = "24h"
    return out


def is_fallback(y: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.abs(y - sd) <= C.FB_TOL
