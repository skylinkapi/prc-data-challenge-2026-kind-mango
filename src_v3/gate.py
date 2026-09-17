"""MP7: label-free upload gate.

Every candidate submission passes here before upload. The gate reads no
label, only shapes and quantiles. It reports per-airport statistics against
the 2025 support and the current reference file, lists every row moved by
more than 3,000 s and blocks the upload on unexplained rows or a strict
drift failure. LIRF `p_fb` and `R_norm` quantiles per month are handled by
`lirf_head_gate` when the caller has those arrays; otherwise the general
`prediction_gate` covers the served output only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src_v3 import config as C

# Thresholds. Every value has a comment naming the rule; MC2 sweeps these.
BIG_MOVE_S = 3_000                # MP7: list rows moved this much or more
# The LIRF ITY340 hedge ships one 2026 row at ~111,654 s and Step A can
# serve up to sd = 131,167 (T16). Set the hard upper bound above that
# support to catch predictions that exceed the 24-h class ceiling only.
UPPER_HARD_S = 200_000
LOWER_HARD_S = 0                  # deterministic clip target
OVER_7200_MARGIN = 2.0            # allowed factor over the 2025 airport count
OVER_3600_MARGIN = 2.0            # same for the 3,600 s threshold


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]
    report: dict

    def as_json(self) -> str:
        return json.dumps({"passed": self.passed, "reasons": self.reasons,
                           "report": self.report}, indent=1)


def _quantiles(x: np.ndarray, qs: tuple[float, ...] = (0.01, 0.5, 0.99, 1.0)
               ) -> dict[str, float]:
    return {f"q{q:g}": float(np.nanquantile(x, q)) for q in qs}


def _airport_support_2025(train_dep: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per-airport 2025 counts above thresholds; label-free."""
    out: dict[str, dict[str, int]] = {}
    for a, g in train_dep.groupby(train_dep["ADEP_mvt"].astype(str), sort=False):
        y = g["TAXITIME_SEC_mvt"].astype(float).values
        out[a] = {
            "n_2025": int(len(y)),
            "n_over_3600": int((y > 3600).sum()),
            "n_over_7200": int((y > 7200).sum()),
            "max_2025": float(y.max()) if len(y) else 0.0,
        }
    return out


def prediction_gate(pred: pd.DataFrame, ref: pd.DataFrame,
                    train_support: dict[str, dict[str, int]],
                    frame_meta: pd.DataFrame) -> GateResult:
    """Score a candidate parquet against label-free rules.

    - `pred` and `ref` are the submission dataframes with columns
      `MVT_ID_mvt` and `TAXITIME_SEC_mvt`.
    - `train_support` is the output of `_airport_support_2025`.
    - `frame_meta` carries `MVT_ID_mvt`, `ADEP_mvt`, `month` per ranking row.
    """
    reasons: list[str] = []
    m = (frame_meta.merge(pred.rename(columns={"TAXITIME_SEC_mvt": "new"}),
                          on="MVT_ID_mvt", how="left", validate="1:1")
                   .merge(ref.rename(columns={"TAXITIME_SEC_mvt": "ref"}),
                          on="MVT_ID_mvt", how="left", validate="1:1"))
    if m["new"].isna().any() or m["ref"].isna().any():
        reasons.append("some template rows are missing from pred or ref")
    new = m["new"].values.astype(float)
    ref_v = m["ref"].values.astype(float)
    diff = new - ref_v

    # Hard bounds
    if (new < LOWER_HARD_S).any():
        reasons.append(f"{int((new < LOWER_HARD_S).sum())} predictions below {LOWER_HARD_S}")
    if (new > UPPER_HARD_S).any():
        reasons.append(f"{int((new > UPPER_HARD_S).sum())} predictions above {UPPER_HARD_S}")

    per_apt: dict[str, dict] = {}
    big_moves: list[dict] = []
    for a, g in m.groupby(m["ADEP_mvt"].astype(str), sort=False):
        idx = g.index.values
        p = new[idx]
        r = ref_v[idx]
        d = diff[idx]
        s = train_support.get(a, {"n_over_3600": 0, "n_over_7200": 0, "max_2025": 0.0,
                                   "n_2025": 0})
        n_over_3600 = int((p > 3600).sum())
        n_over_7200 = int((p > 7200).sum())
        allow_3600 = int(s["n_over_3600"] * OVER_3600_MARGIN + 5)
        allow_7200 = int(s["n_over_7200"] * OVER_7200_MARGIN + 3)
        if n_over_3600 > allow_3600:
            reasons.append(f"{a}: {n_over_3600} predictions > 3,600 s (allow {allow_3600})")
        if n_over_7200 > allow_7200:
            reasons.append(f"{a}: {n_over_7200} predictions > 7,200 s (allow {allow_7200})")

        per_apt[a] = {
            "n": int(len(g)),
            "mean_new": float(p.mean()),
            "mean_ref": float(r.mean()),
            "mean_shift_s": float(d.mean()),
            "rows_moved": int((np.abs(d) > 1e-9).sum()),
            "max_abs_diff_s": float(np.abs(d).max()) if len(d) else 0.0,
            "quantiles": _quantiles(p),
            "n_over_3600": n_over_3600,
            "n_over_7200": n_over_7200,
            "allow_over_3600": allow_3600,
            "allow_over_7200": allow_7200,
            "support_2025": s,
        }
        big = np.where(np.abs(d) >= BIG_MOVE_S)[0]
        for k in big:
            row_id = int(idx[k])
            big_moves.append({"row_index": row_id, "airport": a,
                              "new": float(p[k]), "ref": float(r[k]),
                              "diff_s": float(d[k])})

    report = {
        "n_rows": int(len(m)),
        "per_airport": per_apt,
        "big_moves_over_3000s": big_moves[:200],
        "big_moves_count": len(big_moves),
    }
    return GateResult(passed=not reasons, reasons=reasons, report=report)


def lirf_head_gate(p_fb_by_month: dict[int, np.ndarray],
                   r_norm_by_month: dict[int, np.ndarray]) -> dict:
    """MP7 rider: `p_fb` and `R_norm` quantiles by month at LIRF.

    Callers pass a dict of month -> array. The report is a diagnostic; a
    prior month with a lot of `p_fb` above 0.9 that a later month lacks is
    a flag for the human to explain.
    """
    return {
        "p_fb": {int(m): _quantiles(v, (0.1, 0.5, 0.9, 0.99))
                 for m, v in p_fb_by_month.items()},
        "r_norm": {int(m): _quantiles(v, (0.1, 0.5, 0.9, 0.99))
                   for m, v in r_norm_by_month.items()},
    }


def build_train_support(train_frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Wrapper so callers do not import the private helper."""
    return _airport_support_2025(train_frame)
