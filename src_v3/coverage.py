"""MP8: strict coverage monitor.

The fourteenth-pass monitor compared 2025 with 2026 with a 20-point absolute
rule and skipped the columns that only the LIRF head reads. This module
compares against the same 2025 month, adds a relative rule (share halved),
covers every served column of every component including the head and gates
on failures. It writes one JSON per candidate ranking file next to the
submission.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src_v3 import config as C


# MP8 rules
ABS_DROP_POINTS = 20    # historical rule
RELATIVE_HALF = 0.5     # a share more than halved is a flag


@dataclass
class CoverageResult:
    passed: bool
    reasons: list[str]
    report: dict

    def as_json(self) -> str:
        return json.dumps({"passed": self.passed, "reasons": self.reasons,
                           "report": self.report}, indent=1)


def _non_null_share(s: pd.Series) -> float:
    return float(s.notna().mean())


def _non_zero_share(s: pd.Series) -> float:
    """Share of rows with a non-null and non-zero value."""
    v = pd.to_numeric(s, errors="coerce")
    return float(((v.notna()) & (v != 0)).mean())


def _fails(v_2025: float, v_2026: float, mode: str = "non_null") -> bool:
    """A cell fails when it drops by more than 20 points OR more than half.

    The 20-point rule catches EDDF July 92 to 74; the half rule catches
    LFPG arr_same_rwy 20 to 3.9 which was under 20 points.
    """
    abs_drop = v_2025 - v_2026
    rel_drop = (abs_drop / v_2025) if v_2025 > 1e-9 else 0.0
    return (abs_drop >= ABS_DROP_POINTS / 100.0) or (rel_drop >= RELATIVE_HALF and v_2025 >= 0.05)


def check_columns(train: pd.DataFrame, rank: pd.DataFrame,
                  columns: list[str],
                  train_month_of: dict[int, int],
                  mode: str = "non_null") -> CoverageResult:
    """Compare per-airport, per-month coverage on the given columns.

    `train_month_of` maps a ranking month to the same 2025 month; the
    default identity match {1: 1, 7: 7} keeps the pass's "same 2025 month"
    rule. Callers pick `mode` = "non_null" or "non_zero".
    """
    if mode not in ("non_null", "non_zero"):
        raise ValueError(mode)
    fn = _non_null_share if mode == "non_null" else _non_zero_share

    reasons: list[str] = []
    per_col: dict[str, dict] = {}
    for col in columns:
        if col not in train.columns or col not in rank.columns:
            per_col[col] = {"missing": True}
            reasons.append(f"{col}: absent from train or rank")
            continue
        cells: dict[str, dict] = {}
        for a in sorted(set(rank["ADEP_mvt"].astype(str).dropna())):
            for rm in sorted(set(rank["month"].dropna().astype(int))):
                tm = train_month_of.get(int(rm), int(rm))
                v25 = fn(train[(train["ADEP_mvt"].astype(str) == a)
                               & (train["month"] == tm)][col])
                v26 = fn(rank[(rank["ADEP_mvt"].astype(str) == a)
                              & (rank["month"] == rm)][col])
                cell = {"train_month": tm, "rank_month": int(rm),
                        "v25": v25, "v26": v26,
                        "abs_drop": v25 - v26,
                        "rel_drop": (v25 - v26) / v25 if v25 > 1e-9 else 0.0,
                        "fails": _fails(v25, v26, mode)}
                key = f"{a}/{int(rm)}"
                cells[key] = cell
                if cell["fails"]:
                    reasons.append(
                        f"{col} @ {a} month {int(rm)}: {mode} {v25:.3f} -> {v26:.3f}"
                    )
        per_col[col] = {"mode": mode, "cells": cells}
    return CoverageResult(passed=not reasons, reasons=reasons,
                          report={"columns": per_col})


def check_head_columns_present(feature_file: Path, must_include: list[str]) -> list[str]:
    """MP8/D1: the LIRF head's feature list must not read columns the
    monitor has flagged. Returns the columns still present."""
    text = feature_file.read_text().splitlines()
    return [c for c in must_include if c in text]
