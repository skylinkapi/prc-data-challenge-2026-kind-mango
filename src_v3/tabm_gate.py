"""A TabM member in the LIRF gate, next to the LightGBM g3 gate.

The CatBoost gate member (N7, v109) priced -484 MSE and lost 0.89 s live. Its ordered target statistics fix
the 2025 fallback rate of each category, and the 2026 rates moved. A TabM classifier has no target
statistics: it reads the 2026 arrival-share columns of g3 directly. Neural members of the base transferred
at 2 to 4 times their hold-out price (v94, v105, v111).

`price` sets the epoch count with an early stop on months 11 and 12 (fit months 2 to 6 and 8 to 10), scores
every month out of fold with the six month-pair folds, takes the out-of-fold g3 scores saved by
`catboost_gate price`, calibrates each class with a nested isotonic map, and prices the served mixture on
the LIRF rows of months 1 and 7 outside Step A. `fit` trains on all 12 months and saves the member with the
isotonic map of its out-of-fold scores. `predict_v30.predict_p_fb` serves a `.pt` member through `score`.
The module runs on the GPU when one is visible, else on the CPU; run it with `PYTHONPATH=.gpu_site`.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from src_v3 import config as C
from src_v3.catboost_gate import OOF as G3_OOF
from src_v3.catboost_gate import gate_frame, mixture
from src_v3.measure_catboost_blend import FIT_MONTHS, STOP_MONTHS
from src_v3.measure_gate_clocks import nested_map
from src_v3.tabm_member import Net, Prep, Spec
from src_v3.train_p_fb_v64 import EPS, SD_BINS

SPEC = Spec(**C.TABM_GATE_SPEC)
OOF = C.MODELS / "tabm_gate_oof.parquet"
REPORT = C.MODELS / "tabm_gate.holdout.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)


def probability(net: Net, num: torch.Tensor, cat: torch.Tensor) -> np.ndarray:
    """Mean fallback probability of the k submodels."""
    net.eval()
    with torch.no_grad(), torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
        out = [torch.sigmoid(net(num[i:i + C.TABM_PREDICT_BATCH], cat[i:i + C.TABM_PREDICT_BATCH]).float()).mean(1)
               for i in range(0, len(num), C.TABM_PREDICT_BATCH)]
    return torch.cat(out).cpu().numpy()


def fit(rows: pd.DataFrame, feat: list[str], last_epoch: int,
        stop: pd.DataFrame | None = None) -> tuple[Net, Prep, list[float]]:
    """Binary cross-entropy with a one-cycle schedule over `SPEC.epochs`; with `stop`, keep the best log-loss epoch."""
    torch.manual_seed(C.GLOBAL_SEED)
    prep = Prep(rows, feat, SPEC)
    num, cat, _ = prep(rows, DEV)
    y = torch.tensor(rows["is_fb"].to_numpy(dtype=np.float32), device=DEV)
    net = Net(prep.bins, [len(prep.vocab[c]) for c in prep.cat], SPEC).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=SPEC.lr, weight_decay=SPEC.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SPEC.lr, total_steps=SPEC.epochs * (len(y) // SPEC.batch + 1))
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    held = None if stop is None else (*prep(stop, DEV)[:2], stop["is_fb"].to_numpy())
    best, state, curve = np.inf, None, []
    for epoch in range(last_epoch + 1):
        net.train()
        for idx in torch.randperm(len(y), device=DEV).split(SPEC.batch):
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                out = net(num[idx], cat[idx])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y[idx, None].expand_as(out))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if held is not None:
            curve.append(float(log_loss(held[2], np.clip(probability(net, *held[:2]), EPS, 1 - EPS), labels=[0, 1])))
            log.info("epoch %d: stop-month log loss %.4f", epoch, curve[-1])
            if curve[-1] < best:
                best, state = curve[-1], {k: v.clone() for k, v in net.state_dict().items()}
    if state is not None:
        net.load_state_dict(state)
    return net, prep, curve


def score(model_file: str, frame: pd.DataFrame, feat: list[str]) -> np.ndarray:
    """Raw fallback probability of a saved TabM gate member."""
    saved = torch.load(model_file, map_location=DEV, weights_only=False)
    net = Net(saved["prep"].bins, [len(saved["prep"].vocab[c]) for c in saved["prep"].cat], SPEC).to(DEV)
    net.load_state_dict(saved["state"])
    return probability(net, *saved["prep"](frame, DEV)[:2])


def price() -> None:
    dep, feat = gate_frame()
    _, _, curve = fit(dep[dep["month"].isin(FIT_MONTHS)], feat, SPEC.epochs - 1, stop=dep[dep["month"].isin(STOP_MONTHS)])
    last = int(np.argmin(curve))
    oof = np.full(len(dep), np.nan)
    for hold in C.FOLDS:
        h = dep["month"].isin(hold).values
        net, prep, _ = fit(dep[~h], feat, last)
        oof[h] = probability(net, *prep(dep[h], DEV)[:2])
        log.info("fold %s scored %d rows", hold, h.sum())
    g3 = pd.read_parquet(G3_OOF)
    if not np.array_equal(g3["MVT_ID_mvt"].values, dep["MVT_ID_mvt"].values):
        raise AssertionError(f"{G3_OOF.name} is not in the row order of the gate frame. Rerun `catboost_gate price`.")
    pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].values, "tabm": oof}).to_parquet(OOF)
    y, month, sd = dep["is_fb"].values, dep["month"], dep["sched_delay"].astype(float).values
    cal = {"lgb": nested_map(g3["lgb"].values, y, month), "tabm": nested_map(oof, y, month)}
    arms = {"lgb": cal["lgb"], **{str(w): (1 - w) * cal["lgb"] + w * cal["tabm"] for w in C.CATBOOST_GATE_SHARES}}
    report = {"curve": curve, "last_epoch": last, "corr_oof": float(np.corrcoef(cal["lgb"], cal["tabm"])[0, 1]), "log_loss": {}}
    for label, lo, hi in zip(["sd<=3600", "3600<sd<=14400", "sd>14400"], SD_BINS, SD_BINS[1:]):
        m = (sd > lo) & (sd <= hi)
        report["log_loss"][label] = {arm: float(log_loss(y[m], np.clip(q[m], EPS, 1 - EPS), labels=[0, 1]))
                                     for arm, q in arms.items()}
    report["mixture_minus_lgb"] = mixture(dep, arms)
    log.info("%s", json.dumps(report, indent=1))
    REPORT.write_text(json.dumps(report, indent=1))


def fit_full(tag: str) -> None:
    """Full-year member and the isotonic map of the out-of-fold scores of `price`."""
    dep, feat = gate_frame()
    oof = pd.read_parquet(OOF)
    if not np.array_equal(oof["MVT_ID_mvt"].values, dep["MVT_ID_mvt"].values):
        raise AssertionError(f"{OOF.name} is not in the row order of the gate frame. Rerun `tabm_gate price`.")
    net, prep, _ = fit(dep, feat, json.loads(REPORT.read_text())["last_epoch"])
    torch.save({"prep": prep, "state": net.state_dict()}, C.ROOT / "models" / f"tabm_p_fb_lirf_{tag}.pt")
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof["tabm"].values, dep["is_fb"].values)
    with open(C.ROOT / "models" / f"lirf_regime_{tag}.isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    log.info("wrote the %s TabM gate member", tag)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("price", "fit"))
    ap.add_argument("--tag", default="g3t", help="fit: tag of the member files.")
    args = ap.parse_args()
    price() if args.mode == "price" else fit_full(args.tag)


if __name__ == "__main__":
    main()
