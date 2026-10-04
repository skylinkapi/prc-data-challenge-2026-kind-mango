"""Third-edition N5 simulation: do pseudo-labelled stand-clock rows of the scored days help the base?

In 2026 the ranking rows with a clock at the assigned stand can carry a
pseudo-label: `adsb_own_move` minus the airport median of
`adsb_own_move - y` on clean training rows. In deployment the training
months 1 and 7 hold real 2025 labels next to the 2026 pseudo rows. The
simulation copies that on 2025 with a day split of January and July:

- both arms train the plain v97 LightGBM member on months 2 to 6 and 8 to
  10 plus the odd days of January and July with their real labels;
- the treatment adds the pseudo-labelled stand-clock rows of the even days;
- both arms score the even days against the true labels.

A first design put only pseudo rows in months 1 and 7; `month` is a member
feature, so the trees isolated those months and the member lost 17,952 MSE
on rows without a clock. The report gives the member and the blend change
(plain member swapped, other v97 hold-out members kept) on the even days,
on the hold-out scale.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3 import stack_pushback_v90 as sp
from src_v3.build_nm_clocks_v73 import load_dep
from src_v3.catboost_base import load_frame
from src_v3.fit_blend_v77 import blend
from src_v3.measure_adsb_base_v97 import LGB_EXTRA, MEMBERS, lgb_fit, out_path
from src_v3.measure_catboost_blend import FIT_MONTHS, holdout_rows
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.tune_stack_v93 import frame as stack_frame

PSEUDO_MONTHS = (1, 7)
MAX_OWN_DMIN_M, MAX_GAP_S = 50, 120     # stand-clock quality filter of the pseudo rows (WINNING_PLAN third edition 4.5)
# Parked-aircraft jitter makes `adsb_own_move` hours early: 298 Jan+Jul 2025 candidates had it above 3,600 s with a median
# label of 552 s. The first move must lie at most 600 s before the last point at the stand, and within 3,600 s of `MVT`.
MAX_MOVE_TO_LAST_S, MAX_PSEUDO_S = 600, 3600
REPORT = C.MODELS / "pseudo_n5.holdout.json"
log = logging.getLogger(__name__)


def even_day_ids() -> set:
    """Departure ids of the even UTC days of January and July 2025."""
    d = load_dep(sorted(C.TRAIN_DIR.glob("training_2025-0[17]-*.parquet")))
    return set(d.loc[d["MVT_TIME_UTC_mvt"].dt.day % 2 == 0, "MVT_ID_mvt"])


def pseudo_rows(rows: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Stand-clock rows of `rows` with the label replaced by the bias-corrected clock."""
    y = train["TAXITIME_SEC_mvt"]
    clean = train[(y >= C.Y_CLEAN_MIN) & (y <= C.Y_CLEAN_MAX) & train["adsb_own_move"].notna()]
    bias = (clean["adsb_own_move"] - clean["TAXITIME_SEC_mvt"]).groupby(clean["ADEP_mvt"].astype(str)).median()
    p = rows[rows["adsb_own_move"].notna() & (rows["adsb_own_dmin"] <= MAX_OWN_DMIN_M) & (rows["adsb_gap_max"] <= MAX_GAP_S)
             & (rows["adsb_own_move"] - rows["adsb_own_last"]).between(0, MAX_MOVE_TO_LAST_S)
             & (rows["adsb_own_move"] <= MAX_PSEUDO_S)].copy()
    p["TAXITIME_SEC_mvt"] = (p["adsb_own_move"] - p["ADEP_mvt"].astype(str).map(bias)).clip(lower=C.Y_CLEAN_MIN)
    log.info("pseudo rows %d; airport bias %s", len(p), bias.round(0).to_dict())
    return p.dropna(subset=["TAXITIME_SEC_mvt"])


def members(even: set) -> tuple[np.ndarray, np.ndarray, int]:
    """Hold-out predictions (row order of `PRED_OUT`) of the control and the treatment member."""
    dep, feat = load_frame(LGB_EXTRA)
    scored = dep["month"].isin(PSEUDO_MONTHS) & dep["MVT_ID_mvt"].isin(even)
    train = dep[dep["month"].isin(FIT_MONTHS) | (dep["month"].isin(PSEUDO_MONTHS) & ~scored)]
    pseudo = pseudo_rows(dep[scored], train)
    boosters = [lgb_fit(train, feat), lgb_fit(pd.concat([train, pseudo], ignore_index=True), feat)]
    del dep, train
    h, _ = holdout_rows()
    for path in LGB_EXTRA:
        h = h.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    control, treat = (np.clip(b.predict(h[feat]), 0, None) for b in boosters)
    return control, treat, len(pseudo)


def with_stack(path) -> pd.DataFrame:
    """Hold-out blend at `path` plus the MLP at 0.15, the cross-fitted v95 stack and the A5 bound, as v97 serves it."""
    s, _ = sp.crossfit(stack_frame(C.AUTO_MLP_WEIGHT, False, blend_path=path))
    full = pd.read_parquet(path).dropna(subset=["MVT_ID_mvt"]).set_index("MVT_ID_mvt")
    full.loc[s["MVT_ID_mvt"].values, "base"] = np.fmax(s["stack"].values, sp.stand_bound(s))
    return full


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ref = pd.read_parquet(PRED_OUT)
    even = even_day_ids()
    on = ref["MVT_ID_mvt"].isin(even).values
    y = ref["y"].values[on]
    control, treat, n_pseudo = members(even)
    m = {k: pd.read_parquet(out_path(k))["pred"].values for k in MEMBERS}
    a = pd.read_parquet(C.ADSB_DIR / "v97" / "adsb_clock_train.parquet")
    own = ref[["MVT_ID_mvt"]].merge(a, on="MVT_ID_mvt", how="left")[["adsb_own_move", "adsb_own_last"]].notna().any(axis=1).values[on]

    def mix(lgb):
        return blend({"lgb": 0.5 * lgb + 0.5 * m["lgb_per_airport"], "lgb_anchor": m["lgb_anchor"],
                      "cat": m["cat"], "cat_anchor": m["cat_anchor"]}, *C.BLEND_V77_SERVED)[on]

    report = {"design": "day split: odd days real labels in both arms, even days pseudo rows and scoring",
              "pseudo_rows": n_pseudo, "scored_rows": int(on.sum()),
              "member": mse_delta(treat[on], control[on], y),
              "member_stand_clock_rows": mse_delta(treat[on][own], control[on][own], y[own]),
              "member_other_rows": mse_delta(treat[on][~own], control[on][~own], y[~own]),
              "blend": mse_delta(mix(treat), mix(control), y)}
    report["blend_both_halves_estimate"] = 2 * report["blend"]
    REPORT.write_text(json.dumps(report, indent=1))
    log.info("%s", json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
