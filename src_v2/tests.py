"""F4. One assertion per contract, run against the cached frame.

    python -m src_v2.tests
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src_v2 import config as C
from src_v2 import frame as F
from src_v2 import labels as L


def label_identity(train: pd.DataFrame) -> None:
    """MVT - BLOCK == TAXITIME on 100% of DEP rows."""
    raw = pd.read_parquet(C.TRAIN_DIR / "training_2025-07-01_2025-08-01.parquet",
                          columns=["PHASE_mvt", "MVT_TIME_UTC_mvt",
                                   "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
    dep = raw[raw["PHASE_mvt"] == "DEP"].copy()
    diff = (F._norm_ts(dep["MVT_TIME_UTC_mvt"]) - F._norm_ts(dep["BLOCK_TIME_UTC_mvt"])).dt.total_seconds()
    mismatch = (diff - dep["TAXITIME_SEC_mvt"].astype("float64")).abs() > 1
    assert mismatch.mean() < 0.0001, f"label identity fails on {mismatch.sum()} rows"
    print("label identity OK")


def class_shares_reasonable(train: pd.DataFrame) -> None:
    """LIRF fallback share around 15% under the 5-second window (T11)."""
    lirf = train[train["ADEP_mvt"] == "LIRF"]
    cls = L.classify(lirf["TAXITIME_SEC_mvt"], lirf["sd"])
    shares = cls.value_counts(normalize=True)
    assert 0.12 <= shares.get("fallback", 0) <= 0.20, f"LIRF fallback share off: {shares}"
    print(f"LIRF class shares OK: {shares.to_dict()}")


def coverage_p4(rank: pd.DataFrame) -> None:
    """No unseen NaN spike in the 2026 anchors."""
    for c in ("sd", "mvt_eobt1"):
        nonnull = rank[c].notna().mean()
        assert nonnull > 0.95, f"{c} coverage on rank: {nonnull:.3f}"
    print("2026 anchor coverage OK")


def main():
    train = F.load_or_build("train")
    rank = F.load_or_build("rank")
    label_identity(train)
    class_shares_reasonable(train)
    coverage_p4(rank)


if __name__ == "__main__":
    main()
