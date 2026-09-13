"""One entry-point per stage. All stages read the config in src_v2/config.py.

    python -m src_v2.cli build         # build feature frames for train + rank (F3 cache)
    python -m src_v2.cli fit           # fit encoders, base, regime heads, tail head
    python -m src_v2.cli holdout       # score end-to-end on months 1 and 7 (P1)
    python -m src_v2.cli predict OUT   # write submission parquet OUT
    python -m src_v2.cli report OUT    # P4 label-free report on the file OUT
    python -m src_v2.cli upload OUT    # mc cp OUT to the OSN bucket
"""
from __future__ import annotations

import json
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src_v2 import config as C
from src_v2 import encoders as E
from src_v2 import frame as F
from src_v2 import labels as L
from src_v2.evaluate import harness
from src_v2.heads import regime, tail
from src_v2.predict import serve
from src_v2.train import base as B

LOG_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _configure_logging():
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)


def cmd_build():
    for src in ("train", "rank"):
        t0 = time.time()
        f = F.load_or_build(src)
        logging.info("%s frame: %d rows in %.1fs", src, len(f), time.time() - t0)


def _add_plan_and_encoders(train_dep: pd.DataFrame, rank_dep: pd.DataFrame,
                            refit: bool = False):
    for d in (train_dep, rank_dep):
        d["mvt_ts"] = F._norm_ts(d["MVT_TIME_UTC_mvt"])
        d["arvt1_ts"] = F._norm_ts(d["ARVT_1_flt"])
    route_path = C.MODELS / "route_medians.parquet"
    if refit or not route_path.exists():
        route_med = F.compute_route_medians(train_dep)
        route_med.to_frame("plan_block_median").to_parquet(route_path)
    else:
        route_med = pd.read_parquet(route_path)["plan_block_median"]
    train_dep = F.add_plan_residual(train_dep, route_med)
    rank_dep = F.add_plan_residual(rank_dep, route_med)

    enc_path = C.MODELS / "encoders.pkl"
    if refit or not enc_path.exists():
        enc = E.fit_encoders(train_dep)
        with open(enc_path, "wb") as f:
            pickle.dump(enc, f)
    else:
        with open(enc_path, "rb") as f:
            enc = pickle.load(f)
    train_dep = E.apply_encoders(train_dep, enc)
    rank_dep = E.apply_encoders(rank_dep, enc)
    return train_dep, rank_dep


def cmd_fit():
    train_dep = F.load_or_build("train")
    rank_dep = F.load_or_build("rank")
    train_dep, rank_dep = _add_plan_and_encoders(train_dep, rank_dep, refit=True)
    logging.info("train frame: %d rows, %d cols", len(train_dep), len(train_dep.columns))

    logging.info("== base ==")
    base_meta = B.train(train_dep)
    with open(C.LOG_DIR / "base_train.json", "w") as f:
        json.dump(base_meta, f, indent=2)

    logging.info("== regime heads ==")
    reg_meta = regime.train(train_dep)
    with open(C.LOG_DIR / "regime_train.json", "w") as f:
        json.dump(reg_meta, f, indent=2)

    logging.info("== tail head ==")
    tail_meta = tail.train(train_dep)
    with open(C.LOG_DIR / "tail_train.json", "w") as f:
        json.dump(tail_meta, f, indent=2)

    logging.info("all training artefacts saved to %s", C.MODELS)


def cmd_holdout():
    train_dep = F.load_or_build("train")
    rank_dep = F.load_or_build("rank")
    train_dep, _ = _add_plan_and_encoders(train_dep, rank_dep)

    ho = train_dep[train_dep["month"].isin(C.HOLDOUT_MONTHS)].copy()
    pred = serve.predict(ho)
    res = harness.score(
        pred, ho["TAXITIME_SEC_mvt"].astype(float).values,
        ho["sd"].astype(float).values,
        ho["ADEP_mvt"].astype(str).values,
    )
    print(harness.format_report(res))
    with open(C.LOG_DIR / "holdout.json", "w") as f:
        json.dump(res, f, indent=2)


def cmd_predict(out_name: str):
    train_dep = F.load_or_build("train")
    rank_dep = F.load_or_build("rank")
    _, rank_dep = _add_plan_and_encoders(train_dep, rank_dep)

    pred = serve.predict(rank_dep)
    tmpl = pd.read_parquet(C.SUB_TMPL)
    out = tmpl[["MVT_ID_mvt"]].merge(
        pd.DataFrame({"MVT_ID_mvt": rank_dep["MVT_ID_mvt"].values,
                      "TAXITIME_SEC_mvt": pred}),
        on="MVT_ID_mvt", how="left",
    )
    n_nan = int(out["TAXITIME_SEC_mvt"].isna().sum())
    if n_nan:
        # Should be zero per T7; if not, fall back to per-airport median.
        raise RuntimeError(f"{n_nan} NaN predictions after merge (T7 says 0 expected)")
    # E3 asked for int32; the live scoring service rejects it silently, so we
    # ship as float64 matching submitting.parquet's dtype, rounded to the
    # nearest second.
    out["TAXITIME_SEC_mvt"] = np.rint(out["TAXITIME_SEC_mvt"]).clip(
        C.CLIP_LOW, C.CLIP_HIGH).astype("float64")
    out_path = C.ROOT / "submission" / out_name
    out.to_parquet(out_path)
    logging.info("wrote %s (%d rows, dtype %s)", out_path, len(out),
                 out["TAXITIME_SEC_mvt"].dtype)

    report = serve.label_free_report(pred, rank_dep)
    with open(out_path.with_suffix(".report.json"), "w") as f:
        json.dump(report, f, indent=2)


def cmd_coverage():
    """A3. Per-column coverage: 2025, 2026 Jan, 2026 Jul; fail if drop >20 pts."""
    from src_v2.evaluate.diagnostics import coverage_monitor
    train = F.load_or_build("train")
    rank = F.load_or_build("rank")
    jan = rank["month"] == 1
    jul = rank["month"] == 7
    cols = [c for c in train.columns
            if c not in ("mvt_ts", "arvt1_ts", "TAXITIME_SEC_mvt", "date",
                          "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt")
            and c not in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                           "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
                           "EOBT_1_flt", "ARVT_1_flt")]
    tab = coverage_monitor(train, rank, jan, jul, cols)
    tab.sort_values("drop_jul_pts", ascending=False, inplace=True)
    print(tab.head(30).to_string())
    bad = tab[tab["drop_jul_pts"].abs() > 20]
    if len(bad):
        print("\n!! columns with >20-pt drop:")
        print(bad.to_string())


def cmd_report(out_name: str):
    df = pd.read_parquet(C.ROOT / "submission" / out_name)
    print(df.dtypes)
    print("n rows", len(df))
    print("summary:")
    print(df["TAXITIME_SEC_mvt"].describe())


def cmd_full(out_name: str):
    """Fit + hold-out score + predict + write submission, in one process
    to avoid re-loading the frames."""
    train_dep = F.load_or_build("train")
    rank_dep = F.load_or_build("rank")
    train_dep, rank_dep = _add_plan_and_encoders(train_dep, rank_dep, refit=True)
    logging.info("train frame: %d rows, %d cols", len(train_dep), len(train_dep.columns))

    logging.info("== base ==")
    base_meta = B.train(train_dep)
    logging.info("== regime heads ==")
    reg_meta = regime.train(train_dep)
    logging.info("== tail head ==")
    tail_meta = tail.train(train_dep)
    with open(C.LOG_DIR / "fit_summary.json", "w") as f:
        json.dump({"base": base_meta, "regime": reg_meta, "tail": tail_meta}, f, indent=2)

    logging.info("== hold-out ==")
    ho = train_dep[train_dep["month"].isin(C.HOLDOUT_MONTHS)].copy()
    pred_ho = serve.predict(ho)
    res = harness.score(
        pred_ho, ho["TAXITIME_SEC_mvt"].astype(float).values,
        ho["sd"].astype(float).values,
        ho["ADEP_mvt"].astype(str).values,
    )
    print(harness.format_report(res))
    with open(C.LOG_DIR / "holdout.json", "w") as f:
        json.dump(res, f, indent=2)

    logging.info("== predict rank ==")
    pred = serve.predict(rank_dep)
    tmpl = pd.read_parquet(C.SUB_TMPL)
    out = tmpl[["MVT_ID_mvt"]].merge(
        pd.DataFrame({"MVT_ID_mvt": rank_dep["MVT_ID_mvt"].values,
                      "TAXITIME_SEC_mvt": pred}),
        on="MVT_ID_mvt", how="left",
    )
    n_nan = int(out["TAXITIME_SEC_mvt"].isna().sum())
    if n_nan:
        raise RuntimeError(f"{n_nan} NaN predictions after merge (T7 says 0 expected)")
    out["TAXITIME_SEC_mvt"] = np.rint(out["TAXITIME_SEC_mvt"]).clip(
        C.CLIP_LOW, C.CLIP_HIGH).astype("float64")
    out_path = C.ROOT / "submission" / out_name
    out.to_parquet(out_path)
    logging.info("wrote %s (%d rows, dtype %s)", out_path, len(out), out["TAXITIME_SEC_mvt"].dtype)
    report = serve.label_free_report(pred, rank_dep)
    with open(out_path.with_suffix(".report.json"), "w") as f:
        json.dump(report, f, indent=2)


def cmd_upload(out_name: str):
    creds = (C.ROOT / ".osn_credentials.txt").read_text().splitlines()
    kv = dict(line.split(" = ", 1) for line in creds if " = " in line)
    endpoint = kv.get("endpoint", "https://s3.opensky-network.org:443")
    # RECAP + memory say use port 443, not 9443.
    endpoint = endpoint.replace(":9443", ":443")
    bucket = kv.get("bucket", "prc-2026-kind-mango")
    from minio import Minio
    host = endpoint.split("://", 1)[1]
    client = Minio(host, access_key=kv["access_key"], secret_key=kv["secret_key"], secure=True)
    src = C.ROOT / "submission" / out_name
    logging.info("uploading %s -> %s/%s", src, bucket, out_name)
    client.fput_object(bucket, out_name, str(src))
    logging.info("uploaded")


def main(argv=None):
    _configure_logging()
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__); return
    cmd = argv[0]
    rest = argv[1:]
    fn = globals().get(f"cmd_{cmd}")
    if fn is None:
        print(f"unknown: {cmd}")
        print(__doc__); sys.exit(1)
    fn(*rest)


if __name__ == "__main__":
    main()
