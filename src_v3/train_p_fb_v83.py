"""v83: the LIRF gate with the arrival-link columns (`build_arr_link_v83.py`).

v56 recipe and rounds on all 12 months of LIRF rows, the 167 v23 columns
plus the four link columns, and the v64 out-of-fold isotonic map. Hold-out
price (`measure_gate_clocks.py --tag arr_link_v83`): the served mixture on
Jan+Jul improves by 277 MSE; log loss improves in every `sd` bin.
`--tag g1 --extra models/arr_art_train_g1.parquet` fits the third-edition G1
gate with the arrival artefact shares instead of the link columns. `--extra` takes
several files: `--tag g5` adds the NM clocks to the g3 columns (twenty-first pass).
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm_v21 import CAT_COLS

from src_v3 import config as C
from src_v3.build_arr_link_v83 import OUT_TRAIN as LINK_TRAIN
from src_v3.train_p_fb_v56 import GATE_PARAMS, load_gate_frame
from src_v3.train_p_fb_v64 import V56_BOOSTER, score_oof

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v83")
    ap.add_argument("--extra", nargs="+", default=[str(LINK_TRAIN.relative_to(C.ROOT))], help="Parquets of the added gate columns.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    n_iter = lgb.Booster(model_file=str(V56_BOOSTER)).num_trees()
    dep, feat = load_gate_frame()
    for path in args.extra:
        cols = pd.read_parquet(C.ROOT / path)
        dep = dep.merge(cols, on="MVT_ID_mvt", how="left", validate="m:1")
        feat = feat + [c for c in cols.columns if c != "MVT_ID_mvt"]
    y = dep["is_fb"].values
    dt = lgb.Dataset(dep[feat], label=y, categorical_feature=CAT_COLS, params={"feature_pre_filter": False})
    lgb.train(GATE_PARAMS, dt, num_boost_round=n_iter).save_model(str(C.ROOT / "models" / f"lgbm_p_fb_lirf_{args.tag}.txt"))
    iso = IsotonicRegression(out_of_bounds="clip").fit(score_oof(dep, feat, n_iter), y)
    with open(C.ROOT / "models" / f"lirf_regime_{args.tag}.isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    (C.ROOT / "models" / f"lirf_regime_{args.tag}.features.txt").write_text("\n".join(feat))
    log.info("wrote the %s gate: %d rounds, %d features", args.tag, n_iter, len(feat))


if __name__ == "__main__":
    main()
