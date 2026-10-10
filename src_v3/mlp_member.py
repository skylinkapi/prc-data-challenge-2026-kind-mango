"""Neural-net member for the base blend: an MLP with categorical embeddings (PyTorch).

The trees of the v89 blend share their errors. An MLP learns a smooth
function of the same v85 columns, so its errors differ. Numeric columns go
through a signed log and a z-score fitted on the training rows; a missing
value becomes 0 plus a flag. Each categorical column gets an embedding;
a category unseen in training maps to index 0. `measure` fits on the fit
months, early-stops on months 11 and 12, scores the hold-out months 1 and 7
and prices the member in the saved v89-style blend. `--columns v97` adds the
OPDI and ADS-B columns of the v97 members and prices against the v97 blend
(third-edition N4); its files carry the suffix `_v97`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from src_v3 import config as C
from src_v3.build_lobt_terms import OUT_RANK as LOBT_RANK
from src_v3.build_lobt_terms import OUT_TRAIN as LOBT_TRAIN
from src_v3.build_nm_clocks_v73 import OUT_TRAIN as CLOCKS_TRAIN
from src_v3.build_opdi_own import OUT_TRAIN as OPDI_TRAIN
from src_v3.build_struct_terms import OUT_RANK as STRUCT_RANK
from src_v3.build_struct_terms import OUT_TRAIN as STRUCT_TRAIN
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS, holdout_rows
from src_v3.measure_tail_rules import mse_delta
from train_lgbm_v21 import CAT_COLS

FORWARD_V85 = C.ROOT / "models" / "nm_forward_train_v85.parquet"
# K7 (v127): the ADS-B and OPDI tables with the departures without an NM record linked by flight number
ADSB_K7, OPDI_K7 = C.ADSB_DIR / "v127", C.ROOT / "models"
COLUMN_SETS = {   # extra column files, hold-out blend, ranking feature dump
    "v85": ((CLOCKS_TRAIN, FORWARD_V85), C.MODELS / "blend_v89_holdout.parquet",
            C.MODELS / "lgbm_r_all_v85_rank_features.parquet"),
    "v97": ((CLOCKS_TRAIN, FORWARD_V85, OPDI_TRAIN, C.ADSB_DIR / "v97" / "adsb_clock_train.parquet"),
            C.MODELS / "blend_v97_holdout.parquet", C.MODELS / "lgbm_r_all_v97_rank_features.parquet"),
    "v115": ((CLOCKS_TRAIN, FORWARD_V85, OPDI_TRAIN, C.ADSB_DIR / "v115" / "adsb_clock_train.parquet",
              C.ROOT / "models" / "atfm_slot_train.parquet"),
             C.MODELS / "blend_v97_holdout.parquet", C.MODELS / "lgbm_r_all_v97_rank_features.parquet"),
    "v116": ((CLOCKS_TRAIN, FORWARD_V85, OPDI_TRAIN, C.ADSB_DIR / "v116" / "adsb_clock_train.parquet",
              C.ROOT / "models" / "atfm_slot_train.parquet"),
             C.MODELS / "blend_v97_holdout.parquet", C.MODELS / "lgbm_r_all_v97_rank_features.parquet"),
    "v120": ((CLOCKS_TRAIN, FORWARD_V85, OPDI_TRAIN, C.ADSB_DIR / "v116" / "adsb_clock_train.parquet",
              C.ROOT / "models" / "atfm_slot_train.parquet", LOBT_TRAIN),
             C.MODELS / "blend_v97_holdout.parquet", C.MODELS / "lgbm_r_all_v97_rank_features.parquet"),
    "v126": ((CLOCKS_TRAIN, FORWARD_V85, OPDI_TRAIN, C.ADSB_DIR / "v116" / "adsb_clock_train.parquet",
              C.ROOT / "models" / "atfm_slot_train.parquet", LOBT_TRAIN, STRUCT_TRAIN),
             C.MODELS / "blend_v97_holdout.parquet", C.MODELS / "lgbm_r_all_v97_rank_features.parquet"),
    "v116k7": ((CLOCKS_TRAIN, FORWARD_V85, OPDI_K7 / "opdi_own_train_v127.parquet", ADSB_K7 / "adsb_clock_train.parquet",
                C.ROOT / "models" / "atfm_slot_train.parquet"),
               C.MODELS / "blend_v97_holdout.parquet", C.MODELS / "lgbm_r_all_v97_rank_features.parquet"),
    "v120k7": ((CLOCKS_TRAIN, FORWARD_V85, OPDI_K7 / "opdi_own_train_v127.parquet", ADSB_K7 / "adsb_clock_train.parquet",
                C.ROOT / "models" / "atfm_slot_train.parquet", LOBT_TRAIN),
               C.MODELS / "blend_v97_holdout.parquet", C.MODELS / "lgbm_r_all_v97_rank_features.parquet"),
    "v126k7": ((CLOCKS_TRAIN, FORWARD_V85, OPDI_K7 / "opdi_own_train_v127.parquet", ADSB_K7 / "adsb_clock_train.parquet",
                C.ROOT / "models" / "atfm_slot_train.parquet", LOBT_TRAIN, STRUCT_TRAIN),
               C.MODELS / "blend_v97_holdout.parquet", C.MODELS / "lgbm_r_all_v97_rank_features.parquet")}
RANK_OVERRIDES = {   # ranking tables whose columns replace or extend the feature dump of a column set
    "v115": (C.ADSB_DIR / "v115" / "adsb_clock_rank.parquet", C.ROOT / "models" / "atfm_slot_rank.parquet"),
    "v116": (C.ADSB_DIR / "v116" / "adsb_clock_rank.parquet", C.ROOT / "models" / "atfm_slot_rank.parquet"),
    "v120": (C.ADSB_DIR / "v116" / "adsb_clock_rank.parquet", C.ROOT / "models" / "atfm_slot_rank.parquet", LOBT_RANK),
    "v126": (C.ADSB_DIR / "v116" / "adsb_clock_rank.parquet", C.ROOT / "models" / "atfm_slot_rank.parquet", LOBT_RANK,
             STRUCT_RANK),
    "v116k7": (ADSB_K7 / "adsb_clock_rank.parquet", C.ROOT / "models" / "atfm_slot_rank.parquet",
               OPDI_K7 / "opdi_own_rank_v127.parquet"),
    "v120k7": (ADSB_K7 / "adsb_clock_rank.parquet", C.ROOT / "models" / "atfm_slot_rank.parquet",
               OPDI_K7 / "opdi_own_rank_v127.parquet", LOBT_RANK),
    "v126k7": (ADSB_K7 / "adsb_clock_rank.parquet", C.ROOT / "models" / "atfm_slot_rank.parquet",
               OPDI_K7 / "opdi_own_rank_v127.parquet", LOBT_RANK, STRUCT_RANK)}


def swap_adsb(paths: tuple, old: Path, new: Path) -> tuple:
    return tuple(new if q == old else q for q in paths)


# K8 (v130): the K7 sets with the ADS-B tables in which the departures that no callsign links take their lift-off aircraft
ADSB_K8 = C.ADSB_DIR / "v130"
for _name in ("v116", "v120", "v126"):
    _extra, _blend, _dump = COLUMN_SETS[f"{_name}k7"]
    COLUMN_SETS[f"{_name}k8"] = (swap_adsb(_extra, ADSB_K7 / "adsb_clock_train.parquet", ADSB_K8 / "adsb_clock_train.parquet"),
                                 _blend, _dump)
    RANK_OVERRIDES[f"{_name}k8"] = swap_adsb(RANK_OVERRIDES[f"{_name}k7"], ADSB_K7 / "adsb_clock_rank.parquet",
                                             ADSB_K8 / "adsb_clock_rank.parquet")
BLEND_HOLDOUT = COLUMN_SETS["v85"][1]
PRED_OUT = C.MODELS / "holdout_mlp.parquet"
RANK_OUT = C.MODELS / "rank_mlp.parquet"


def out_paths(columns: str) -> tuple[Path, Path, Path]:
    """Hold-out predictions, ranking predictions and price report of a column set; v85 keeps the v94 names."""
    sfx = "" if columns == "v85" else f"_{columns}"
    return (C.MODELS / f"holdout_mlp{sfx}.parquet", C.MODELS / f"rank_mlp{sfx}.parquet",
            C.MODELS / f"mlp_member{sfx}.holdout.json")
log = logging.getLogger(__name__)


class Prep:
    """Column transforms fitted on the training rows and reused on every other frame."""

    def __init__(self, frame: pd.DataFrame, feat: list[str]):
        self.num = [c for c in feat if c not in CAT_COLS]
        self.cat = [c for c in feat if c in CAT_COLS]
        x = self._signed_log(frame)
        self.mean, self.std = np.nanmean(x, axis=0), np.nanstd(x, axis=0) + 1e-6
        self.vocab = {c: {v: i + 1 for i, v in enumerate(frame[c].astype(str).unique())} for c in self.cat}

    def _signed_log(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.num].to_numpy(dtype=np.float64)
        return np.sign(x) * np.log1p(np.abs(x))

    def __call__(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x = (self._signed_log(frame) - self.mean) / self.std
        miss = np.isnan(x)
        x = np.clip(np.nan_to_num(x), -C.MLP_Z_CLIP, C.MLP_Z_CLIP)
        num = torch.tensor(np.hstack([x, miss]), dtype=torch.float32)
        cat = torch.tensor(np.stack([frame[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(dtype=np.int64)
                                     for c in self.cat], axis=1))
        return num, cat


class Net(nn.Module):
    def __init__(self, n_num: int, cards: list[int]):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n + 1, min(C.MLP_EMB_MAX, (n + 2) // 2)) for n in cards])
        width = n_num + sum(e.embedding_dim for e in self.emb)
        layers = []
        for h in C.MLP_HIDDEN:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(C.MLP_DROPOUT)]
            width = h
        self.body = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        e = [m(cat[:, i]) for i, m in enumerate(self.emb)]
        return self.body(torch.cat([num, *e], dim=1)).squeeze(1)


def predict(net: Net, num: torch.Tensor, cat: torch.Tensor) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        out = [net(num[i:i + C.MLP_BATCH * 4], cat[i:i + C.MLP_BATCH * 4]) for i in range(0, len(num), C.MLP_BATCH * 4)]
    return np.clip(torch.cat(out).numpy() * C.MLP_Y_SCALE, 0, None)


def train(prep: Prep, fit_rows: pd.DataFrame, stop_rows: pd.DataFrame | None, seed: int = C.GLOBAL_SEED) -> Net:
    """Fit with AdamW and a one-cycle schedule; keep the best stop-month epoch, or `MLP_SERVE_EPOCH` without stop rows."""
    torch.manual_seed(seed)
    torch.set_num_threads(C.MLP_THREADS)
    num, cat = prep(fit_rows)
    y = torch.tensor(fit_rows["TAXITIME_SEC_mvt"].clip(0, C.MLP_Y_CLIP).to_numpy() / C.MLP_Y_SCALE, dtype=torch.float32)
    if stop_rows is not None:
        s_num, s_cat = prep(stop_rows)
        s_y = stop_rows["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    net = Net(num.shape[1], [len(prep.vocab[c]) for c in prep.cat])
    opt = torch.optim.AdamW(net.parameters(), lr=C.MLP_LR, weight_decay=C.MLP_WEIGHT_DECAY)
    steps = C.MLP_EPOCHS * (len(y) // C.MLP_BATCH + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.MLP_LR, total_steps=steps)
    best, best_state = np.inf, None
    for epoch in range(C.MLP_EPOCHS):
        net.train()
        for idx in torch.randperm(len(y)).split(C.MLP_BATCH):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(num[idx], cat[idx]), y[idx])
            loss.backward()
            opt.step()
            sched.step()
        if stop_rows is None:
            log.info("seed %d epoch %d done", seed, epoch)
            if epoch == C.MLP_SERVE_EPOCH:
                return net
            continue
        rmse = float(np.sqrt(np.mean((predict(net, s_num, s_cat) - s_y) ** 2)))
        log.info("epoch %d: stop-month RMSE %.2f s", epoch, rmse)
        if rmse < best:
            best, best_state = rmse, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


def measure(columns: str) -> None:
    extra, blend_path, _ = COLUMN_SETS[columns]
    pred_out, _, report_out = out_paths(columns)
    dep, feat = load_frame(extra)
    prep = Prep(dep[dep["month"].isin(FIT_MONTHS)], feat)
    net = train(prep, dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)])
    del dep
    h, _ = holdout_rows()
    for path in extra:
        h = h.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    pred = predict(net, *prep(h))
    ref = pd.read_parquet(blend_path)
    if not np.array_equal(ref["y"].values, h["TAXITIME_SEC_mvt"].values.astype(float)):
        raise AssertionError("The hold-out frame and the saved blend are not in the same row order. "
                             "Rebuild the blend with stack_pushback_v90, then rerun.")
    y, base = ref["y"].values, ref["base"].values
    report = {"member_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
              "blend_rmse": float(np.sqrt(np.mean((base - y) ** 2))),
              "corr_residuals": float(np.corrcoef(pred - y, base - y)[0, 1]),
              "weights": {str(w): mse_delta((1 - w) * base + w * pred, base, y) for w in C.MLP_BLEND_WEIGHTS}}
    log.info("%s", json.dumps(report, indent=1))
    report_out.write_text(json.dumps(report, indent=1))
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "pred": pred}).to_parquet(pred_out)


def serve(columns: str) -> None:
    """Fit on every served training row with `MLP_SERVE_SEEDS` seeds; score the ranking feature dump of the column set."""
    extra, _, rank_dump = COLUMN_SETS[columns]
    dep, feat = load_frame(extra)
    rank = pd.read_parquet(rank_dump)
    missing = sorted(set(feat) - set(rank.columns))
    if missing:
        raise KeyError(f"The ranking dump has no column {missing[:5]}. Rerun predict_v67 for v89 to rebuild it.")
    prep = Prep(dep, feat)
    x = prep(rank)
    pred = np.mean([predict(train(prep, dep, None, seed), *x) for seed in C.MLP_SERVE_SEEDS], axis=0)
    pd.DataFrame({"MVT_ID_mvt": rank["MVT_ID_mvt"].values, "pred": pred}).to_parquet(out_paths(columns)[1])
    log.info("ranking rows %d, mean %.1f s", len(pred), pred.mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("measure", "serve"))
    ap.add_argument("--columns", choices=sorted(COLUMN_SETS), default="v85")
    args = ap.parse_args()
    measure(args.columns) if args.mode == "measure" else serve(args.columns)


if __name__ == "__main__":
    main()
