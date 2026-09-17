"""MP1: month-blocked six-fold split.

Each fold holds out one Jan-side and one Jul-side month, trains on the other
ten, and early-stops on the two training months next to the hold-out pair.
Every artefact under MP4 (encoders, rate maps, tail model, calibrator,
route medians, normal means) is fitted on the fold's training months only.

The `(1, 7)` fold is the historical hold-out and is reported alone next to
the fold mean and standard error (MP1, MP2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src_v3 import config as C


@dataclass(frozen=True)
class Fold:
    hold: tuple[int, int]
    early_stop: tuple[int, int]
    train: tuple[int, ...]

    def train_mask(self, month: pd.Series) -> pd.Series:
        return month.isin(self.train)

    def stop_mask(self, month: pd.Series) -> pd.Series:
        return month.isin(self.early_stop)

    def hold_mask(self, month: pd.Series) -> pd.Series:
        return month.isin(self.hold)


def build_folds() -> list[Fold]:
    """Six folds. The month sets do not overlap between train and stop."""
    all_months = tuple(range(1, 13))
    folds: list[Fold] = []
    for hold in C.FOLDS:
        stop = C.EARLY_STOP_BY_FOLD[hold]
        train = tuple(m for m in all_months if m not in hold and m not in stop)
        folds.append(Fold(hold=hold, early_stop=stop, train=train))
    return folds


def paired_fold_delta(deltas: np.ndarray, alpha: float = C.FOLD_INTERVAL_ALPHA
                      ) -> dict:
    """Return the mean, standard error and (1 - alpha) interval of one lever's
    per-fold paired delta. MP2 accepts a lever when the interval lies below 0,
    rejects when it lies above, and records "inconclusive" otherwise."""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    # Two-sided normal interval, per MP2 (n = 6 is small; the pass says "90 %
    # interval", read as the normal approximation on the fold-mean estimator).
    from math import sqrt
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    lo, hi = mean - z * se, mean + z * se
    verdict = "inconclusive"
    if hi < 0:
        verdict = "accept"
    elif lo > 0:
        verdict = "reject"
    return {"mean": mean, "se": se, "lo": lo, "hi": hi,
            "alpha": alpha, "n": n, "verdict": verdict,
            "per_fold": d.tolist()}


def wrap_warning() -> dict[tuple[int, int], tuple[int, int]]:
    """The (6, 12) fold early-stops on (1, 7). Callers filtering that fold's
    stop rows out of a global evaluation on months 1 and 7 should call this
    function to check whether it applies to their protocol.
    """
    return dict(C.FOLD_WRAP_WARN)
