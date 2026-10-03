"""WINNING_PLAN second edition 4.5 step 1: a LightGBM specialist on the pushback clocks of the scored flight.

It learns `y` from the OPDI, ADS-B and NM clocks of departures with an own
clock outside LIRF, on the 2025 months outside the stack's hold-out months
1 and 7. Its prediction is therefore out of sample on the hold-out rows that
train the stack and on the 2026 ranking rows, and the stack reads it as one
more column (`stack_pushback_v90 --spec`). Rounds come from early stopping on
whole days of the training months.
"""
from __future__ import annotations

import argparse
import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from src_v3 import config as C
from src_v3.build_adsb_clock import OUT_RANK as ADSB_RANK, OUT_TRAIN as ADSB_TRAIN
from src_v3.build_nm_clocks_v73 import OUT_RANK as NM_RANK, OUT_TRAIN as NM_TRAIN, load_dep
from src_v3.build_opdi_own import OUT_RANK as OPDI_RANK, OUT_TRAIN as OPDI_TRAIN

log = logging.getLogger(__name__)


def paths(tag: str) -> dict:
    """Output files of one specialist fit; the tag freezes them per served version."""
    return {"holdout": C.MODELS / f"adsb_spec_holdout_{tag}.parquet", "rank": C.MODELS / f"adsb_spec_rank_{tag}.parquet",
            "report": C.MODELS / f"adsb_spec_{tag}.holdout.json"}


def frame(paths: list, tables: tuple, label: bool) -> pd.DataFrame:
    """Departures with every clock column, the airport and the UTC hour; only rows with an own clock outside LIRF."""
    dep = load_dep(paths, ["TAXITIME_SEC_mvt"] if label else [])
    f = dep[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", *(["TAXITIME_SEC_mvt"] if label else [])]]
    for t in tables:
        f = f.merge(pd.read_parquet(t), on="MVT_ID_mvt", how="left", validate="1:1")
    f["apt"] = pd.Categorical(f["ADEP_mvt"].astype(str), categories=list(C.TARGETS))
    f["hour"] = f["MVT_TIME_UTC_mvt"].dt.hour
    seen = f[["adsb_seg", "opdi_first"]].notna().any(axis=1)
    return f[seen & (f["apt"] != "LIRF")].reset_index(drop=True)


def features(f: pd.DataFrame) -> list[str]:
    drop = {"MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "adsb_day_loaded"}
    return [c for c in f.columns if c not in drop]


def fit(train: pd.DataFrame, feat: list[str]) -> tuple[lgb.Booster, int]:
    """Early-stop on every `SPEC_STOP_EVERY`-th day, then refit on all training rows with that round count."""
    day = train["MVT_TIME_UTC_mvt"].dt.dayofyear
    stop = (day % C.SPEC_STOP_EVERY == 0).values
    y = train["TAXITIME_SEC_mvt"].values
    probe = lgb.train(C.SPEC_PARAMS, lgb.Dataset(train.loc[~stop, feat], y[~stop]), C.SPEC_MAX_ROUNDS,
                      valid_sets=[lgb.Dataset(train.loc[stop, feat], y[stop])],
                      callbacks=[lgb.early_stopping(C.SPEC_PATIENCE, verbose=False)])
    rounds = probe.best_iteration
    return lgb.train(C.SPEC_PARAMS, lgb.Dataset(train[feat], y), rounds), rounds


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="Version tag of the output files (v96).")
    out = paths(ap.parse_args().tag)
    rows = frame(sorted(C.TRAIN_DIR.glob("training_*.parquet")), (NM_TRAIN, OPDI_TRAIN, ADSB_TRAIN), label=True)
    y = rows["TAXITIME_SEC_mvt"].astype(float)
    month = rows["MVT_TIME_UTC_mvt"].dt.month
    hold = month.isin((1, 7))
    train = rows[~hold & (y > 0) & (y <= C.MLP_Y_CLIP)]
    feat = features(rows)
    booster, rounds = fit(train, feat)
    h = rows[hold]
    pred = np.clip(booster.predict(h[feat]), 0, None)
    report = {"train_rows": int(len(train)), "train_days": int(train["MVT_TIME_UTC_mvt"].dt.floor("D").nunique()),
              "rounds": rounds, "holdout_rows": int(len(h)),
              "holdout_rmse": float(np.sqrt(np.mean((pred - h["TAXITIME_SEC_mvt"]) ** 2)))}
    log.info("%s", report)
    out["report"].write_text(json.dumps(report, indent=1))
    pd.DataFrame({"MVT_ID_mvt": h["MVT_ID_mvt"].values, "spec": pred}).to_parquet(out["holdout"])
    r = frame([C.RANK], (NM_RANK, OPDI_RANK, ADSB_RANK), label=False)
    pd.DataFrame({"MVT_ID_mvt": r["MVT_ID_mvt"].values,
                  "spec": np.clip(booster.predict(r[feat]), 0, None)}).to_parquet(out["rank"])


if __name__ == "__main__":
    main()
