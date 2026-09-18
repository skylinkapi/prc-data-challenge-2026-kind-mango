"""v53 = v51 recipe (MB3 base + v41 LIRF head + MS1 + MS2) + MS3.

Runs the v51 predict path with dump_features to save the v51 served
feature frame, then scores each of the three v51 boosters individually.
Where the three-member spread exceeds 3,600 s outside LIRF, replaces the
ensemble mean with the row median (MS3). LIRF rows are untouched. MS1
and MS2 apply on top exactly as v51.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict_v30 import main as predict_v30_main, load_training_categories
from predict_v45 import build_extra
from train_lgbm_v21 import CAT_COLS as SERVED_CAT_COLS
from train_r_norm_lirf_seeds import SEEDS as LIRF_SEEDS

from src_v3 import config as C
from src_v3.postprocess import apply_ms1, apply_ms2, apply_ms3, assert_ms4
from src_v3.support import build_support

MODELS_OLD = C.ROOT / "models"
V51_SEEDS = (42, 43, 44)
V2_RANK = MODELS_OLD / "v2" / "cache" / "frame_rank.parquet"
log = logging.getLogger(__name__)
R_NORM_FILES = [f"lgbm_r_norm_lirf_v41_s{s}.txt" for s in LIRF_SEEDS]


def _load_train_min() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(C.TRAIN_DIR / "*.parquet"))):
        t = pd.read_parquet(f, columns=["ADEP_mvt", "PHASE_mvt",
                                        "MVT_TIME_UTC_mvt",
                                        "SCHED_TIME_UTC_mvt",
                                        "TAXITIME_SEC_mvt"])
        t = t[(t["PHASE_mvt"] == "DEP") & t["ADEP_mvt"].isin(C.TARGETS)]
        t = t[t["TAXITIME_SEC_mvt"].astype(float) > 0]
        t["mvt_ts"] = pd.to_datetime(t["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
        t["sched_ts"] = pd.to_datetime(t["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
        t["sched_delay"] = (t["mvt_ts"] - t["sched_ts"]).dt.total_seconds()
        parts.append(t[["ADEP_mvt", "TAXITIME_SEC_mvt", "sched_delay"]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre-ms", default="kind-mango_v53_pre_ms.parquet")
    ap.add_argument("--out", default="kind-mango_v53.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    dump_path = C.MODELS / "v51_rank_features.parquet"
    pre_ms_path = C.ROOT / "submission" / args.pre_ms

    log.info("=== predict_v53 step 1: v51 served path + dump ===")
    predict_v30_main(
        args.pre_ms,
        r_norm_files=R_NORM_FILES,
        base_model="lgbm_r_all_v51",
        extra_columns=build_extra(),
        r_norm_features="lirf_regime_v41.features.txt",
        r_norm_clip=None,
        dump_features=str(dump_path),
    )

    log.info("=== step 2: score 3 v51 members individually ===")
    with open(MODELS_OLD / "lgbm_r_all_v51.features.txt") as f:
        feat = f.read().splitlines()
    dump = pd.read_parquet(dump_path)
    dump = dump[["MVT_ID_mvt"] + [c for c in feat if c in dump.columns]]
    log.info("aligning categorical vocab with the training set")
    train_cats = load_training_categories()
    for c in SERVED_CAT_COLS:
        if c in dump.columns:
            dump[c] = pd.Categorical(dump[c].astype(str), categories=train_cats[c])

    members: list[np.ndarray] = []
    for s in V51_SEEDS:
        t0 = time.time()
        b = lgb.Booster(model_file=str(MODELS_OLD / f"lgbm_r_all_v51_s{s}.txt"))
        p = np.clip(b.predict(dump[feat]), 0, None)
        members.append(p)
        log.info("v51 seed %d scored in %.0fs", s, time.time() - t0)
    members_arr = np.array(members)
    mean_pred = members_arr.mean(axis=0)

    log.info("=== step 3: MS3 disagreement gate ===")
    adep_dump = dump["ADEP_mvt"].astype(str).values
    ms3_pred, ms3_moved = apply_ms3(members_arr, adep_dump)
    n_ms3 = int(sum(v["rows_moved"] for v in ms3_moved.values()))
    log.info("MS3 moved %d rows: %s", n_ms3,
             {k: v["rows_moved"] for k, v in ms3_moved.items()})

    log.info("=== step 4: apply the MS3 delta to the served pre-MS output ===")
    pre = pd.read_parquet(pre_ms_path).rename(columns={"TAXITIME_SEC_mvt": "pre"})
    align = dump[["MVT_ID_mvt", "ADEP_mvt"]].assign(base_mean=mean_pred,
                                                    base_ms3=ms3_pred)
    align["delta"] = align["base_ms3"] - align["base_mean"]
    m = pre.merge(align[["MVT_ID_mvt", "delta"]], on="MVT_ID_mvt", how="left",
                  validate="1:1")
    corrected = (m["pre"].values.astype(float)
                 + m["delta"].fillna(0).values.astype(float))

    log.info("=== step 5: MS1 + MS2 ===")
    v2 = pd.read_parquet(V2_RANK, columns=["MVT_ID_mvt", "ADEP_mvt",
                                            "flt_null", "mvt_eobt1"])
    template = pd.read_parquet(C.SUB_TMPL, columns=["MVT_ID_mvt"])
    aligned = template.merge(pd.DataFrame({"MVT_ID_mvt": m["MVT_ID_mvt"].values,
                                            "y": corrected}),
                             on="MVT_ID_mvt", how="left", validate="1:1"
                            ).merge(v2, on="MVT_ID_mvt", how="left",
                                     validate="1:1")
    train = _load_train_min()
    support = build_support(train)
    pred_vec = aligned["y"].values.astype(float)
    adep = aligned["ADEP_mvt"].astype(str).values
    flt_null = aligned["flt_null"].fillna(1).astype(int).values
    mvt_eobt1 = aligned["mvt_eobt1"].fillna(0).astype(float).values
    v53, _ = apply_ms1(pred_vec, adep, flt_null, mvt_eobt1, support)
    v53, _ = apply_ms2(v53, adep, support)

    v51_ref = pd.read_parquet(C.ROOT / "submission" / "kind-mango_v51.parquet")
    v51_map = dict(zip(v51_ref["MVT_ID_mvt"], v51_ref["TAXITIME_SEC_mvt"]))
    v51_vec = np.array([v51_map.get(i, np.nan) for i in aligned["MVT_ID_mvt"].values])
    n_moved = int((np.abs(v53 - v51_vec) > 1e-9).sum())
    max_move = float(np.max(np.abs(v53 - v51_vec)))
    log.info("v53 vs v51: %d rows moved, max |diff| %.0f s", n_moved, max_move)
    assert_ms4(v53, template["MVT_ID_mvt"].values, aligned["MVT_ID_mvt"].values)
    out = pd.DataFrame({"MVT_ID_mvt": template["MVT_ID_mvt"].values,
                        "TAXITIME_SEC_mvt": v53.astype(np.float64)})
    (C.ROOT / "submission" / args.out).write_bytes(b"")  # placeholder
    out.to_parquet(C.ROOT / "submission" / args.out)
    log.info("wrote submission/%s (%d rows)", args.out, len(out))
    (C.MODELS / "predict_v53.json").write_text(json.dumps({
        "ms3_rows_moved": {k: v["rows_moved"] for k, v in ms3_moved.items()},
        "ms3_total": n_ms3,
        "rows_diff_vs_v51": n_moved,
        "max_abs_diff_vs_v51_s": max_move,
    }, indent=1))


if __name__ == "__main__":
    main()
