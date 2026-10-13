"""TabM member for the base blend: k MLPs with shared weights, on the v97 columns, trained on the GPU.

The v97-column MLP (`mlp_member.py`) gained -0.48 s live at share 0.2 (v105), about four times its
hold-out price. TabM (Gorishniy et al., ICLR 2025; `tabm`, Apache-2.0) trains k submodels at once that
share their weights. Piecewise-linear embeddings (`rtdl_num_embeddings`) encode each numeric column on
quantile bins of the training rows. A missing value takes the training median and sets a missing flag.
Each categorical column gets an embedding; a category unseen in training maps to index 0.

`measure` fits on the fit months, early-stops on months 11 and 12, scores the hold-out months 1 and 7 and
prices the member in the v97 hold-out blend. `serve` fits on every served training row up to the best
measured epoch and scores the ranking feature dump. Each member is a named `Spec` in `config.TABM_SPECS`;
both modes write the `mlp_member` file names with that tag. `combine` writes the neural term `nn`, the MLP
and the mean of the named TabM members, that `stack_pushback_v90 --mlp-columns nn` reads. Run with
`PYTHONPATH=.gpu_site` (`config.GPU_SITE`): the served CPU build of torch has no CUDA.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rtdl_num_embeddings as rne
import tabm
import torch
from torch import nn

from src_v3 import config as C
from src_v3.catboost_base import load_frame
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS, holdout_rows
from src_v3.measure_tail_rules import mse_delta
from src_v3.mlp_member import CAT_COLS, COLUMN_SETS, RANK_OVERRIDES, out_paths

COLUMNS = "v97"
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Spec:
    """Settings of one TabM member; `config.TABM_SPECS` holds the overrides of each named member."""
    k: int = C.TABM_K
    blocks: int = C.TABM_BLOCKS
    width: int = C.TABM_WIDTH
    dropout: float = C.TABM_DROPOUT
    bins: int = C.TABM_BINS
    d_num: int = C.TABM_D_NUM
    lr: float = C.TABM_LR
    weight_decay: float = C.TABM_WEIGHT_DECAY
    batch: int = C.TABM_BATCH
    epochs: int = C.TABM_EPOCHS
    serve_epoch: int = C.TABM_EPOCHS - 1
    anchor_col: str | None = None   # mvt_aobt3: learn y minus the NM clock (0 where missing), as the anchored trees
    seeds: tuple = C.TABM_SERVE_SEEDS
    columns: str = COLUMNS             # column set of `mlp_member.COLUMN_SETS`
    embed: str = "ple"                 # numeric embeddings: piecewise-linear ("ple") or periodic ("plr")


class Prep:
    """Column transforms fitted on the training rows and reused on every other frame."""

    def __init__(self, frame: pd.DataFrame, feat: list[str], spec: Spec):
        self.anchor_col = spec.anchor_col
        self.num = [c for c in feat if c not in CAT_COLS]
        self.cat = [c for c in feat if c in CAT_COLS]
        x = self._signed_log(frame)
        self.median = np.nanmedian(x, axis=0)
        self.mean, self.std = np.nanmean(x, axis=0), np.nanstd(x, axis=0) + 1e-6
        self.flagged = np.isnan(x).any(axis=0)
        self.keep = None
        rows = np.random.default_rng(C.GLOBAL_SEED).choice(len(x), min(len(x), C.TABM_BIN_ROWS), replace=False)
        sample = self._numeric(x[rows])
        self.keep = sample.min(axis=0) < sample.max(axis=0)   # a column constant in the sample has no bins
        self.bins = rne.compute_bins(torch.tensor(sample[:, self.keep]), n_bins=spec.bins)
        self.vocab = {c: {v: i + 1 for i, v in enumerate(frame[c].astype(str).unique())} for c in self.cat}

    def _signed_log(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.num].to_numpy(dtype=np.float64)
        return np.sign(x) * np.log1p(np.abs(x))

    def _numeric(self, x: np.ndarray) -> np.ndarray:
        """z-scored columns, missing at the median, plus the missing flags; the embedding's linear term needs a small scale under fp16."""
        miss = np.isnan(x)
        z = np.clip((np.where(miss, self.median, x) - self.mean) / self.std, -C.MLP_Z_CLIP, C.MLP_Z_CLIP)
        z = np.hstack([z, miss[:, self.flagged]]).astype(np.float32)
        return z if self.keep is None else z[:, self.keep]

    def __call__(self, frame: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Numeric and categorical tensors, and the anchor in seconds that the net predicts on top of."""
        num = torch.tensor(self._numeric(self._signed_log(frame)), device=device)
        cat = np.stack([frame[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(dtype=np.int64)
                        for c in self.cat], axis=1)
        col = self.anchor_col
        anchor = np.zeros(len(frame)) if col is None else frame[col].clip(lower=0).fillna(0).to_numpy(dtype=float)
        return num, torch.tensor(cat, device=device), anchor


class Net(nn.Module):
    """TabM: numeric embeddings (piecewise linear or periodic), categorical embeddings, a backbone with `k` heads."""
    def __init__(self, bins: list[torch.Tensor], cards: list[int], spec: Spec):
        super().__init__()
        self.num = (rne.PiecewiseLinearEmbeddings(bins, spec.d_num, activation=False, version="B") if spec.embed == "ple"
                    else rne.PeriodicEmbeddings(len(bins), spec.d_num, lite=False))
        self.emb = nn.ModuleList([nn.Embedding(n + 1, min(C.MLP_EMB_MAX, (n + 2) // 2)) for n in cards])
        chunks = [spec.d_num] * len(bins) + [e.embedding_dim for e in self.emb]
        self.view = tabm.EnsembleView(k=spec.k)
        self.body = tabm.make_tabm_backbone(d_in=sum(chunks), n_blocks=spec.blocks, d_block=spec.width,
                                            dropout=spec.dropout, k=spec.k, start_scaling_init="normal",
                                            start_scaling_init_chunks=chunks)
        self.head = tabm.LinearEnsemble(spec.width, 1, k=spec.k)

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        """Predictions of the k submodels, shape (batch, k)."""
        x = torch.cat([self.num(num).flatten(1), *[m(cat[:, i]) for i, m in enumerate(self.emb)]], dim=1)
        return self.head(self.body(self.view(x))).squeeze(-1)


def predict(net: Net, num: torch.Tensor, cat: torch.Tensor, anchor: np.ndarray) -> np.ndarray:
    """Anchor plus the mean of the k submodels, in seconds, clipped at 0."""
    net.eval()
    step = C.TABM_PREDICT_BATCH
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = [net(num[i:i + step], cat[i:i + step]).float().mean(1) for i in range(0, len(num), step)]
    return np.clip(anchor + torch.cat(out).cpu().numpy() * C.MLP_Y_SCALE, 0, None)


def train(prep: Prep, fit_rows: pd.DataFrame, stop_rows: pd.DataFrame | None, seed: int,
          spec: Spec) -> tuple[Net, list[float]]:
    """AdamW with a one-cycle schedule over `spec.epochs`; keep the best stop-month epoch, or stop after `spec.serve_epoch`."""
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    num, cat, anchor = prep(fit_rows, dev)
    y = torch.tensor((fit_rows["TAXITIME_SEC_mvt"].clip(0, C.MLP_Y_CLIP).to_numpy() - anchor) / C.MLP_Y_SCALE,
                     dtype=torch.float32, device=dev)
    net = Net(prep.bins, [len(prep.vocab[c]) for c in prep.cat], spec).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=spec.lr,
                                                total_steps=spec.epochs * (len(y) // spec.batch + 1))
    scaler = torch.amp.GradScaler()
    stop = None if stop_rows is None else (*prep(stop_rows, dev), stop_rows["TAXITIME_SEC_mvt"].to_numpy(dtype=float))
    best, state, curve = np.inf, None, []
    for epoch in range(spec.epochs):
        net.train()
        for idx in torch.randperm(len(y), device=dev).split(spec.batch):
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(num[idx], cat[idx])
            loss = ((out.float() - y[idx, None]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if stop is None:
            log.info("seed %d epoch %d done", seed, epoch)
            if epoch == spec.serve_epoch:
                return net, curve
            continue
        curve.append(float(np.sqrt(np.mean((predict(net, *stop[:3]) - stop[3]) ** 2))))
        log.info("epoch %d: stop-month RMSE %.2f s", epoch, curve[-1])
        if curve[-1] < best:
            best, state = curve[-1], {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net, curve


def price(pred: np.ndarray, ref: pd.DataFrame, mlp: np.ndarray) -> dict:
    """MSE deltas on the hold-out scale: TabM in the v97 blend, and on top of the served v105 base (MLP at 0.2)."""
    y, base = ref["y"].values, ref["base"].values
    v105 = (1 - C.MLP_BLEND_W) * base + C.MLP_BLEND_W * mlp
    return {"member_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "blend_rmse": float(np.sqrt(np.mean((base - y) ** 2))),
            "corr_residuals": float(np.corrcoef(pred - y, base - y)[0, 1]),
            "corr_residuals_mlp": float(np.corrcoef(pred - y, mlp - y)[0, 1]),
            "in_v97_blend": {str(w): mse_delta((1 - w) * base + w * pred, base, y) for w in C.TABM_BLEND_WEIGHTS},
            "on_v105_base": {str(w): mse_delta((1 - w) * v105 + w * pred, v105, y) for w in C.TABM_BLEND_WEIGHTS},
            "replace_mlp": {str(w): mse_delta((1 - w) * base + w * pred, v105, y) for w in C.TABM_BLEND_WEIGHTS}}


def measure(tag: str, spec: Spec) -> None:
    """Fit one seed on the fit months, score the 2025 hold-out, write the member and its price against the blend."""
    extra, blend_path, _ = COLUMN_SETS[spec.columns]
    pred_out, _, report_out = out_paths(tag)
    dep, feat = load_frame(extra)
    prep = Prep(dep[dep["month"].isin(FIT_MONTHS)], feat, spec)
    net, curve = train(prep, dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)], spec.seeds[0], spec)
    del dep
    h, _ = holdout_rows()
    for path in extra:
        h = h.merge(pd.read_parquet(path), on="MVT_ID_mvt", how="left", validate="m:1")
    pred = predict(net, *prep(h, torch.device("cuda")))
    ref = pd.read_parquet(blend_path)
    if not np.array_equal(ref["y"].values, h["TAXITIME_SEC_mvt"].values.astype(float)):
        raise AssertionError("The hold-out frame and the saved blend are not in the same row order. "
                             "Rebuild the blend with stack_pushback_v90, then rerun.")
    mlp = pd.read_parquet(out_paths(COLUMNS)[0])["pred"].values
    report = {"curve": curve, "best_epoch": int(np.argmin(curve)), **price(pred, ref, mlp)}
    log.info("%s", json.dumps(report, indent=1))
    report_out.write_text(json.dumps(report, indent=1))
    pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "pred": pred}).to_parquet(pred_out)


def saved_paths(tag: str, seed: int) -> tuple[Path, Path]:
    """Fitted column transforms of a member and the weights of one of its seeds (git ignores both)."""
    return C.MODELS / f"tabm_{tag}.prep.pkl", C.MODELS / f"tabm_{tag}_s{seed}.pt"


def rank_frame(dump: Path, overrides: tuple, feat: list[str]) -> pd.DataFrame:
    """The ranking feature dump with the columns of `overrides` replaced or added."""
    rank = pd.read_parquet(dump)
    for path in overrides:
        cols = pd.read_parquet(path)
        rank = rank.drop(columns=[c for c in cols if c != "MVT_ID_mvt" and c in rank]).merge(
            cols, on="MVT_ID_mvt", how="left", validate="1:1")
    missing = sorted(set(feat) - set(rank.columns))
    if missing:
        raise KeyError(f"The ranking dump has no column {missing[:5]}. Rerun predict_v67 on the v97 recipe to rebuild it.")
    return rank


def serve(tag: str, spec: Spec) -> None:
    """Fit on every served training row with `spec.seeds`, save the fits, and score the v97 ranking feature dump."""
    extra, _, rank_dump = COLUMN_SETS[spec.columns]
    dep, feat = load_frame(extra)
    rank = rank_frame(rank_dump, RANK_OVERRIDES.get(spec.columns, ()), feat)
    prep = Prep(dep, feat, spec)
    with open(saved_paths(tag, spec.seeds[0])[0], "wb") as f:
        pickle.dump(prep, f)
    x = prep(rank, torch.device("cuda"))
    preds = []
    for seed in spec.seeds:
        net = train(prep, dep, None, seed, spec)[0]
        torch.save(net.state_dict(), saved_paths(tag, seed)[1])
        preds.append(predict(net, *x))
    pred = np.mean(preds, axis=0)
    pd.DataFrame({"MVT_ID_mvt": rank["MVT_ID_mvt"].values, "pred": pred}).to_parquet(out_paths(tag)[1])
    log.info("ranking rows %d, mean %.1f s", len(pred), pred.mean())


def predict_saved(tag: str, spec: Spec, dump: Path, overrides: list[Path], out: Path) -> None:
    """Score another ranking feature dump with the saved fits of `serve`; no training."""
    with open(saved_paths(tag, spec.seeds[0])[0], "rb") as f:
        prep = pickle.load(f)
    rank = rank_frame(dump, tuple(overrides), prep.num + prep.cat)
    x = prep(rank, torch.device("cuda"))
    cards = [len(prep.vocab[c]) for c in prep.cat]
    preds = []
    for seed in spec.seeds:
        net = Net(prep.bins, cards, spec).to("cuda")
        net.load_state_dict(torch.load(saved_paths(tag, seed)[1], map_location="cuda"))
        preds.append(predict(net, *x))
    pred = np.mean(preds, axis=0)
    pd.DataFrame({"MVT_ID_mvt": rank["MVT_ID_mvt"].values, "pred": pred}).to_parquet(out)
    log.info("%s: %d rows from the saved fits, mean %.1f s -> %s", tag, len(pred), pred.mean(), out.name)


def combine(tags: list[str], share: float) -> float:
    """Write the neural term `nn`: the MLP and the mean of the TabM members `tags` at `share` on top of the v105 base.

    The stack mixes one neural file at one weight; the return value is that weight. `d_tabm_mlp` is the TabM mean
    minus the MLP, a member column of the stack (v137).
    """
    weight = C.MLP_BLEND_W * (1 - share) + share
    for i in (0, 1):
        mlp = pd.read_parquet(out_paths(COLUMNS)[i])
        members = [pd.read_parquet(out_paths(t)[i]) for t in tags]
        for t, m in zip(tags, members):
            if not np.array_equal(mlp["MVT_ID_mvt"].values, m["MVT_ID_mvt"].values, equal_nan=True):
                raise AssertionError(f"{out_paths(t)[i].name} and {out_paths(COLUMNS)[i].name} differ in row order. "
                                     "Rerun both members on the same frame, then retry.")
        tbm = np.mean([m["pred"].values for m in members], axis=0)
        nn_pred = (C.MLP_BLEND_W * (1 - share) * mlp["pred"].values + share * tbm) / weight
        mlp.assign(pred=nn_pred, d_tabm_mlp=tbm - mlp["pred"].values).to_parquet(out_paths("nn")[i])
    log.info("wrote the nn files; serve them with --mlp-columns nn --mlp-weight %.4f", weight)
    return weight


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("measure", "serve", "predict", "combine"))
    ap.add_argument("--tag", default="tabm", choices=sorted(C.TABM_SPECS), help="measure, serve: the member.")
    ap.add_argument("--tags", nargs="+", default=["tabm"], help="combine: TabM members to average.")
    ap.add_argument("--share", type=float, default=C.TABM_ON_V105, help="combine: TabM share on top of the v105 base.")
    ap.add_argument("--dump", type=Path, help="predict: ranking feature dump to score with the saved fits of `--tag`.")
    ap.add_argument("--overrides", nargs="*", type=Path, default=[], help="predict: tables that replace dump columns.")
    ap.add_argument("--out", type=Path, help="predict: output parquet of the ranking predictions.")
    args = ap.parse_args()
    if args.mode == "combine":
        combine(args.tags, args.share)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is visible. The TabM member trains on the GPU. "
                           "Run with PYTHONPATH=.gpu_site (the CUDA build of torch), then retry.")
    spec = Spec(**C.TABM_SPECS[args.tag])
    if args.mode == "predict":
        predict_saved(args.tag, spec, args.dump, args.overrides, args.out)
        return
    measure(args.tag, spec) if args.mode == "measure" else serve(args.tag, spec)


if __name__ == "__main__":
    main()
