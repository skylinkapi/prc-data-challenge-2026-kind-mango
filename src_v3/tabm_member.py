"""TabM member for the base blend: k MLPs with shared weights, on the v97 columns, trained on the GPU.

The v97-column MLP (`mlp_member.py`) gained -0.48 s live at share 0.2 (v105), about four times its
hold-out price. TabM (Gorishniy et al., ICLR 2025; `tabm`, Apache-2.0) trains k submodels at once that
share their weights. Piecewise-linear embeddings (`rtdl_num_embeddings`) encode each numeric column on
quantile bins of the training rows. A missing value takes the training median and sets a missing flag.
Each categorical column gets an embedding; a category unseen in training maps to index 0.

`measure` fits on the fit months, early-stops on months 11 and 12, scores the hold-out months 1 and 7 and
prices the member in the v97 hold-out blend. `serve` fits on every served training row up to the best
measured epoch and scores the ranking feature dump. Both write the `mlp_member` file names with the tag
`tabm`, so `stack_pushback_v90 --mlp-columns tabm` reads them. Run with `PYTHONPATH=.gpu_site`
(`config.GPU_SITE`): the served CPU build of torch has no CUDA.
"""
from __future__ import annotations

import argparse
import json
import logging

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
from src_v3.mlp_member import CAT_COLS, COLUMN_SETS, out_paths

COLUMNS = "v97"
TAG = "tabm"
log = logging.getLogger(__name__)


class Prep:
    """Column transforms fitted on the training rows and reused on every other frame."""

    def __init__(self, frame: pd.DataFrame, feat: list[str]):
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
        self.bins = rne.compute_bins(torch.tensor(sample[:, self.keep]), n_bins=C.TABM_BINS)
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

    def __call__(self, frame: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        num = torch.tensor(self._numeric(self._signed_log(frame)), device=device)
        cat = np.stack([frame[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(dtype=np.int64)
                        for c in self.cat], axis=1)
        return num, torch.tensor(cat, device=device)


class Net(nn.Module):
    def __init__(self, bins: list[torch.Tensor], cards: list[int]):
        super().__init__()
        self.num = rne.PiecewiseLinearEmbeddings(bins, C.TABM_D_NUM, activation=False, version="B")
        self.emb = nn.ModuleList([nn.Embedding(n + 1, min(C.MLP_EMB_MAX, (n + 2) // 2)) for n in cards])
        chunks = [C.TABM_D_NUM] * len(bins) + [e.embedding_dim for e in self.emb]
        self.view = tabm.EnsembleView(k=C.TABM_K)
        self.body = tabm.make_tabm_backbone(d_in=sum(chunks), n_blocks=C.TABM_BLOCKS, d_block=C.TABM_WIDTH,
                                            dropout=C.TABM_DROPOUT, k=C.TABM_K, start_scaling_init="normal",
                                            start_scaling_init_chunks=chunks)
        self.head = tabm.LinearEnsemble(C.TABM_WIDTH, 1, k=C.TABM_K)

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        """Predictions of the k submodels, shape (batch, k)."""
        x = torch.cat([self.num(num).flatten(1), *[m(cat[:, i]) for i, m in enumerate(self.emb)]], dim=1)
        return self.head(self.body(self.view(x))).squeeze(-1)


def predict(net: Net, num: torch.Tensor, cat: torch.Tensor) -> np.ndarray:
    """Mean of the k submodels in seconds, clipped at 0."""
    net.eval()
    step = C.TABM_BATCH * 4
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = [net(num[i:i + step], cat[i:i + step]).float().mean(1) for i in range(0, len(num), step)]
    return np.clip(torch.cat(out).cpu().numpy() * C.MLP_Y_SCALE, 0, None)


def train(prep: Prep, fit_rows: pd.DataFrame, stop_rows: pd.DataFrame | None, seed: int,
          serve_epoch: int | None = None) -> tuple[Net, list[float]]:
    """AdamW with a one-cycle schedule over `TABM_EPOCHS`; keep the best stop-month epoch, or stop after `serve_epoch`."""
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    num, cat = prep(fit_rows, dev)
    y = torch.tensor(fit_rows["TAXITIME_SEC_mvt"].clip(0, C.MLP_Y_CLIP).to_numpy() / C.MLP_Y_SCALE,
                     dtype=torch.float32, device=dev)
    net = Net(prep.bins, [len(prep.vocab[c]) for c in prep.cat]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=C.TABM_LR, weight_decay=C.TABM_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.TABM_LR,
                                                total_steps=C.TABM_EPOCHS * (len(y) // C.TABM_BATCH + 1))
    scaler = torch.amp.GradScaler()
    stop = None if stop_rows is None else (*prep(stop_rows, dev), stop_rows["TAXITIME_SEC_mvt"].to_numpy(dtype=float))
    best, state, curve = np.inf, None, []
    for epoch in range(C.TABM_EPOCHS):
        net.train()
        for idx in torch.randperm(len(y), device=dev).split(C.TABM_BATCH):
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
            if epoch == serve_epoch:
                return net, curve
            continue
        curve.append(float(np.sqrt(np.mean((predict(net, stop[0], stop[1]) - stop[2]) ** 2))))
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


def measure(tag: str = TAG) -> None:
    extra, blend_path, _ = COLUMN_SETS[COLUMNS]
    pred_out, _, report_out = out_paths(tag)
    dep, feat = load_frame(extra)
    prep = Prep(dep[dep["month"].isin(FIT_MONTHS)], feat)
    net, curve = train(prep, dep[dep["month"].isin(FIT_MONTHS)], dep[dep["month"].isin(STOP_MONTHS)], C.GLOBAL_SEED)
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


def serve(epoch: int) -> None:
    """Fit on every served training row with `TABM_SERVE_SEEDS`; score the v97 ranking feature dump."""
    extra, _, rank_dump = COLUMN_SETS[COLUMNS]
    dep, feat = load_frame(extra)
    rank = pd.read_parquet(rank_dump)
    missing = sorted(set(feat) - set(rank.columns))
    if missing:
        raise KeyError(f"The ranking dump has no column {missing[:5]}. Rerun predict_v67 on the v97 recipe to rebuild it.")
    prep = Prep(dep, feat)
    x = prep(rank, torch.device("cuda"))
    pred = np.mean([predict(train(prep, dep, None, seed, epoch)[0], *x) for seed in C.TABM_SERVE_SEEDS], axis=0)
    pd.DataFrame({"MVT_ID_mvt": rank["MVT_ID_mvt"].values, "pred": pred}).to_parquet(out_paths(TAG)[1])
    log.info("ranking rows %d, mean %.1f s", len(pred), pred.mean())


def combine(share: float) -> float:
    """Write the neural term `nn` = MLP and TabM in the proportions of TabM at `share` on top of the v105 base.

    The stack mixes one neural file at one weight; the return value is that weight.
    """
    weight = C.MLP_BLEND_W * (1 - share) + share
    for i in (0, 1):
        mlp, tbm = (pd.read_parquet(out_paths(t)[i]) for t in (COLUMNS, TAG))
        if not np.array_equal(mlp["MVT_ID_mvt"].values, tbm["MVT_ID_mvt"].values, equal_nan=True):
            raise AssertionError(f"{out_paths(TAG)[i].name} and {out_paths(COLUMNS)[i].name} differ in row order. "
                                 "Rerun both members on the same frame, then retry.")
        nn_pred = (C.MLP_BLEND_W * (1 - share) * mlp["pred"].values + share * tbm["pred"].values) / weight
        mlp.assign(pred=nn_pred).to_parquet(out_paths("nn")[i])
    log.info("wrote the nn files; serve them with --mlp-columns nn --mlp-weight %.4f", weight)
    return weight


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("measure", "serve", "combine"))
    ap.add_argument("--epoch", type=int, default=C.TABM_SERVE_EPOCH, help="serve: last epoch (0-based).")
    ap.add_argument("--share", type=float, default=C.TABM_ON_V105, help="combine: TabM share on top of the v105 base.")
    args = ap.parse_args()
    if args.mode == "combine":
        combine(args.share)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is visible. The TabM member trains on the GPU. "
                           "Run with PYTHONPATH=.gpu_site (the CUDA build of torch), then retry.")
    measure() if args.mode == "measure" else serve(args.epoch)


if __name__ == "__main__":
    main()
