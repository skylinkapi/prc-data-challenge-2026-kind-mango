"""WINNING_PLAN A1 and A2: stack the pushback clocks of the scored flight on the served blend.

The stack is a LightGBM model on `y - base`. `base` is the v89-style blend:
out of sample on the 2025 hold-out months 1 and 7, and the served v89 file
on the ranking rows. Only rows outside LIRF with an OPDI or an ADS-B clock
take part; every other row keeps `base`. `price` cross-fits the stack on
the hold-out and reports the MSE change. `serve` fits on every hold-out
row, moves the v89 file and applies MS1 and MS2 again. With
`--mlp-weight w`, the base outside LIRF is `(1 - w) * base + w * MLP`
(`src_v3/mlp_member.py`) on both the hold-out and the ranking rows.
`--neighbours` adds the A4 columns of `build_pushback_neighbours.py`;
`--stand-bound` floors the served file at the A5 stand bound after MS1
and MS2; `--base` names the served file that the stack moves (v92: the
v89 recipe with three per-airport seeds, B2). `--stack-rows own` moves only
rows with a clock at the assigned stand: on the v97 base the stack gains
41 MSE there and loses 175 on segment rows (third-edition N1).
`--member-columns` gives the stack two more columns: the neural term minus
the tree blend, and the TabM mean minus the MLP (v137).
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
from src_v3.adsb_specialist import paths as spec_paths
from src_v3.build_pushback_neighbours import OUT_RANK as NB_RANK, OUT_TRAIN as NB_TRAIN
from src_v3.fit_blend_v77 import blend
from src_v3.measure_member_v77 import out_path
from src_v3.measure_nm_clocks import PRED_OUT
from src_v3.measure_tail_rules import mse_delta
from src_v3.mlp_member import PRED_OUT as MLP_HOLDOUT
from src_v3.mlp_member import out_paths as mlp_paths
from src_v3.predict_v57 import write_ms

BLEND_HOLDOUT = C.MODELS / "blend_v89_holdout.parquet"
REPORT = C.MODELS / "stack_pushback_v90.holdout.json"
CLOCKS = ("adsb_seg", "adsb_move", "adsb_push", "adsb_parked", "adsb_stop1_end", "adsb_stop2_end",
          "adsb_own_first", "adsb_own_last", "adsb_own_move", "adsb_own_parked",
          "opdi_first", "opdi_last", "opdi_stand", "mvt_aobt3", "spec")
log = logging.getLogger(__name__)


def holdout_blend() -> pd.DataFrame:
    """Out-of-sample v89-style blend on the hold-out rows outside LIRF (members of v85 and v89)."""
    ref = pd.read_parquet(PRED_OUT)
    per = pd.read_parquet(C.MODELS / "holdout_lgb_per_airport_v89.parquet")["pred"].values
    glob = pd.read_parquet(C.MODELS / "holdout_lgb_v85.parquet")["pred"].values
    members = {"lgb": 0.5 * (per + glob),
               "lgb_anchor": pd.read_parquet(C.MODELS / "holdout_lgb_anchor_mb5.parquet")["pred_plain_stop"].values,
               "cat": pd.read_parquet(out_path("cat_plain"))["pred"].values,
               "cat_anchor": pd.read_parquet(out_path("cat_anchor"))["pred"].values}
    return pd.DataFrame({"MVT_ID_mvt": ref["MVT_ID_mvt"].values, "y": ref["y"].values,
                         "base": blend(members, *C.BLEND_V77_SERVED)})


def frame(rows: pd.DataFrame, paths: tuple, dep: pd.DataFrame, nb_rows: bool = False) -> pd.DataFrame:
    """`rows` with airport, day and every clock column; the clock differences are relative to `base`.

    With `nb_rows`, rows without an own clock but with a neighbour clock (A4) join the stack.
    """
    f = rows.merge(dep[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]], on="MVT_ID_mvt", how="left", validate="1:1")
    for p in paths:
        f = f.merge(pd.read_parquet(p), on="MVT_ID_mvt", how="left", validate="1:1")
    for c in CLOCKS:
        if c in f:
            f[f"d_{c}"] = f[c] - f["base"]
    f["adsb_day_loaded"] = f.get("adsb_day_loaded", 1)  # without ADS-B, every day counts for the folds
    f["apt"] = pd.Categorical(f["ADEP_mvt"].astype(str), categories=list(C.TARGETS))
    seen = f[[c for c in ("adsb_seg", "opdi_first") if c in f]].notna().any(axis=1)
    if nb_rows:
        seen |= f[[c for c in f if c.startswith("pnb_") and c.endswith("_n")]].gt(0).any(axis=1)
    f["has_clock"] = seen & (f["apt"] != "LIRF")
    return f


def stand_bound(f: pd.DataFrame) -> np.ndarray:
    """A5: the taxi is at least `MVT` minus the last point at the assigned stand, or minus the move after a later
    stand stop (a tow), outside LIRF. On 2025 labels 3.3 % of rows fall below it, 0.17 % by more than 300 s.

    A segment that ends more than `ADSB_SEGMENT_GAP_S` before `MVT` (v133) gives no bound: the aircraft can stay at the
    stand after its last point. On those hold-out rows the bound costs 2,240 MSE."""
    b = np.fmin(f["adsb_own_last"].values, f["adsb_push"].values)
    return np.where((f["apt"] != "LIRF") & (f["adsb_end"] <= C.ADSB_SEGMENT_GAP_S), b, np.nan)


def features(f: pd.DataFrame) -> list[str]:
    """Stack columns: each column of the frame except the keys, the label, the flags, the weight and four NM terms."""
    drop = {"MVT_ID_mvt", "y", "ADEP_mvt", "MVT_TIME_UTC_mvt", "has_clock", "adsb_day_loaded", "w",
            "aobt3_eobt1", "aobt3_sched", "mvt_lobt", "lobt_aobt3"}
    return [c for c in f.columns if c not in drop]


def fit(f: pd.DataFrame, feat: list[str], params: dict = C.STACK_V90_PARAMS,
        rounds: int = C.STACK_V90_ROUNDS, res_clip: int = C.STACK_V90_RES_CLIP_S) -> lgb.Booster:
    """LightGBM on `y - base`, clipped at `res_clip`; it uses the row weights `w` if the frame has them."""
    d = lgb.Dataset(f[feat], (f["y"] - f["base"]).clip(-res_clip, res_clip), weight=f.get("w"))
    return lgb.train(params, d, rounds)


def coverage_class(f: pd.DataFrame) -> pd.Series:
    """Coverage class of each row: a clock at the assigned stand, an ADS-B ground segment only, or OPDI only."""
    own = f[["adsb_own_move", "adsb_own_last"]].notna().any(axis=1)
    return pd.Series(np.select([own, f["adsb_seg"].notna()], ["own", "seg"], "opdi"), index=f.index)


def coverage_weights(train: pd.DataFrame, rank: pd.DataFrame) -> pd.Series:
    """Weight of each clock row of `train`: the 2026 share of its airport and coverage class over its 2025 share.

    The hold-out holds the 2025 coverage mix (EGLL had a stand clock on 4 % of departures, 46 % in 2026), so the
    weights point the stack's loss at the mix it serves. They are capped at `STACK_COVER_W` and read no label."""
    def shares(f: pd.DataFrame) -> pd.Series:
        c = f[f["has_clock"]]
        return c.groupby([c["ADEP_mvt"].astype(str), coverage_class(c)]).size() / len(c)
    ratio = (shares(rank) / shares(train)).clip(*C.STACK_COVER_W)
    key = pd.MultiIndex.from_arrays([train["ADEP_mvt"].astype(str), coverage_class(train)])
    return pd.Series(ratio.reindex(key).fillna(C.STACK_COVER_W[0]).values, index=train.index).where(train["has_clock"])


def crossfit(f: pd.DataFrame, params: dict = C.STACK_V90_PARAMS, rounds: int = C.STACK_V90_ROUNDS,
             res_clip: int = C.STACK_V90_RES_CLIP_S) -> tuple[pd.DataFrame, str]:
    """Clock rows with the out-of-fold `stack` column: January against July once both months hold enough ADS-B days."""
    feat, s = features(f), f[f["has_clock"]].copy()
    days = s["MVT_TIME_UTC_mvt"].dt.floor("D")
    months = s["MVT_TIME_UTC_mvt"].dt.month
    loaded = s["adsb_day_loaded"] == 1
    use_month = min(days[loaded & (months == 1)].nunique(), days[loaded & (months == 7)].nunique()) >= C.STACK_V90_MIN_FOLD_DAYS
    group = (months == 1).values if use_month else (days.dt.day % 2 == 0).values
    pred = np.empty(len(s))
    for g in (True, False):
        booster = fit(s[group != g], feat, params, rounds, res_clip)
        pred[group == g] = s.loc[group == g, "base"].values + booster.predict(s.loc[group == g, feat])
    s["stack"] = np.clip(pred, 0, None)
    s["fold_jan"] = group
    return s, "month" if use_month else "day parity"


def price(f: pd.DataFrame, params: dict = C.STACK_V90_PARAMS, rounds: int = C.STACK_V90_ROUNDS) -> dict:
    """Cross-fit the stack over two day groups of the hold-out; MSE change on the hold-out scale."""
    s, fold = crossfit(f, params, rounds)
    use_month = fold == "month"
    y, new, old = s["y"].values, s["stack"].values, s["base"].values
    apt = s["ADEP_mvt"].astype(str).values
    bounded = np.fmax(new, stand_bound(s))
    return {"fold": "month" if use_month else "day parity", "rows": int(len(s)),
            "all": mse_delta(new, old, y),
            "stand_bound": {"rows_lifted": int((bounded > new).sum()), "mse": mse_delta(bounded, new, y)},
            "per_airport": {a: {"rows": int((apt == a).sum()), "mse": mse_delta(new[apt == a], old[apt == a], y[apt == a]),
                                "rmse_base": float(np.sqrt(np.mean((old[apt == a] - y[apt == a]) ** 2))),
                                "rmse_stack": float(np.sqrt(np.mean((new[apt == a] - y[apt == a]) ** 2)))}
                            for a in sorted(set(apt))}}


def price_loao(f: pd.DataFrame) -> dict:
    """Leave one airport out: the stack fits on the other airports, which prices airports without 2025 tracks."""
    feat = [c for c in features(f) if c != "apt"]
    s = f[f["has_clock"]]
    apt = s["ADEP_mvt"].astype(str).values
    out = {}
    for a in sorted(set(apt)):
        if (apt == a).sum() >= C.STACK_V90_MIN_LOAO_ROWS:
            t = s[apt == a]
            new = np.clip(t["base"].values + fit(s[apt != a], feat).predict(t[feat]), 0, None)
            out[a] = mse_delta(new, t["base"].values, t["y"].values)
    return out


def mixed(base: pd.Series, nn: pd.DataFrame, weight: float, member_columns: bool) -> dict:
    """Columns of the base with the neural term `nn` at `weight`; `member_columns` adds the member gaps (v137)."""
    out = {"base": (1 - weight) * base.values + weight * nn["pred"].values}
    if member_columns:
        out |= {"d_nn_tree": nn["pred"].values - base.values, "d_tabm_mlp": nn["d_tabm_mlp"].values}
    return out


def mix_mlp(base: pd.DataFrame, mlp_path, weight: float, is_lirf: np.ndarray, member_columns: bool = False) -> pd.DataFrame:
    """`base` with the MLP member blended in at `weight` outside LIRF."""
    if weight == 0:
        return base
    nn = base[["MVT_ID_mvt"]].merge(pd.read_parquet(mlp_path), on="MVT_ID_mvt", how="left", validate="1:1")
    if nn.loc[~is_lirf, "pred"].isna().any():
        raise ValueError(f"{mlp_path.name} misses rows outside LIRF. Rerun `python -m src_v3.mlp_member`, then retry.")
    cols = mixed(base["base"], nn, weight, member_columns)
    return base.assign(**{**cols, "base": np.where(is_lirf, base["base"], cols["base"])})


def serve(train: pd.DataFrame, rank_paths: tuple, args: argparse.Namespace) -> None:
    """Fit the stack on the hold-out rows with a clock, move the ranking rows with a clock, write the file."""
    served = pd.read_parquet(C.ROOT / "submission" / args.base).rename(columns={"TAXITIME_SEC_mvt": "base"})
    adep = served[["MVT_ID_mvt"]].merge(load_dep([C.RANK])[["MVT_ID_mvt", "ADEP_mvt"]], on="MVT_ID_mvt", how="left")
    served = mix_mlp(served, mlp_paths(args.mlp_columns)[1], args.mlp_weight, (adep["ADEP_mvt"].astype(str) == "LIRF").values,
                     args.member_columns)
    f = frame(served, rank_paths, load_dep([C.RANK], ["CALLSIGN_flt"]), args.neighbours == "rows")
    if args.coverage_weights:
        train = train.assign(w=coverage_weights(train, f))
    feat = [c for c in features(train) if c != "w"]
    params = {**C.STACK_V90_PARAMS, "num_leaves": args.leaves, "min_data_in_leaf": args.min_data, "lambda_l2": args.l2}
    booster = fit(train[train["has_clock"]], feat, params, rounds=args.rounds)
    rows = f["has_clock"] & (f[["adsb_own_move", "adsb_own_last"]].notna().any(axis=1) if args.stack_rows == "own" else True)
    move = np.where(rows, booster.predict(f[feat]), 0.0)
    log.info("moved %d rows; mean move %.1f s; p99 |move| %.0f s", int(rows.sum()),
             move[rows].mean(), np.quantile(np.abs(move[rows]), 0.99))
    lower = pd.DataFrame({"MVT_ID_mvt": f["MVT_ID_mvt"], "lower": stand_bound(f)}) if args.stand_bound else None
    write_ms(pd.DataFrame({"MVT_ID_mvt": f["MVT_ID_mvt"], "TAXITIME_SEC_mvt": np.clip(f["base"] + move, 0, None)}),
             args.out, lower)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("price", "serve"))
    ap.add_argument("--out", default="kind-mango_v90.parquet")
    ap.add_argument("--sources", choices=("opdi", "adsb", "both"), default="both")
    ap.add_argument("--mlp-weight", type=float, default=0.0, help="MLP share of the base outside LIRF.")
    ap.add_argument("--mlp-columns", default="v85", help="Column set of the MLP member files (N4: v97).")
    ap.add_argument("--neighbours", choices=("none", "cols", "rows"), default="none",
                    help="A4: neighbour pushback columns; `rows` also stacks rows with neighbours but no own clock.")
    ap.add_argument("--base", default="kind-mango_v89.parquet", help="Served file that the stack moves.")
    ap.add_argument("--stand-bound", action="store_true", help="A5: floor at the last time at the stand, after MS1.")
    ap.add_argument("--stack-rows", choices=("all", "own"), default="all",
                    help="Clock rows the stack moves; `own` keeps only rows with a clock at the assigned stand (N1).")
    ap.add_argument("--adsb-train", default=str(ADSB_TRAIN), help="ADS-B clock table of the 2025 training days.")
    ap.add_argument("--adsb-rank", default=str(ADSB_RANK), help="ADS-B clock table of the ranking days (v115: X-Plane stands).")
    ap.add_argument("--opdi-train", default=str(OPDI_TRAIN), help="OPDI clock table of the 2025 training days.")
    ap.add_argument("--opdi-rank", default=str(OPDI_RANK), help="OPDI clock table of the ranking days (v127: flight-number links).")
    ap.add_argument("--rounds", type=int, default=C.STACK_V90_ROUNDS, help="Boosting rounds of the stack.")
    ap.add_argument("--leaves", type=int, default=C.STACK_V90_PARAMS["num_leaves"], help="Leaves of the stack trees.")
    ap.add_argument("--min-data", type=int, default=C.STACK_V90_PARAMS["min_data_in_leaf"], help="Minimum rows per stack leaf.")
    ap.add_argument("--l2", type=float, default=C.STACK_V90_PARAMS["lambda_l2"], help="L2 penalty of the stack leaves.")
    ap.add_argument("--coverage-weights", action="store_true",
                    help="Weight the hold-out clock rows by the 2026 share of their airport and coverage class (v128).")
    ap.add_argument("--member-columns", action="store_true",
                    help="Add the neural term minus the tree blend and the TabM mean minus the MLP as stack columns (v137).")
    ap.add_argument("--spec", help="Tag of the ADS-B specialist files to add as a column (v96: v96).")
    ap.add_argument("--blend-holdout", default=str(BLEND_HOLDOUT),
                    help="Out-of-sample hold-out blend the stack learns from; it must match `--base` (v97: blend_v97_holdout).")
    args = ap.parse_args()
    use = {"opdi": (args.opdi_train, args.opdi_rank), "adsb": (args.adsb_train, args.adsb_rank)}
    use = [v for k, v in use.items() if args.sources in (k, "both")]
    if args.neighbours != "none":
        use.append((NB_TRAIN, NB_RANK))
    if args.spec:
        spec = spec_paths(args.spec)
        use.append((spec["holdout"], spec["rank"]))
    if not BLEND_HOLDOUT.exists():
        holdout_blend().to_parquet(BLEND_HOLDOUT)
    rows = pd.read_parquet(args.blend_holdout)
    if args.mlp_weight:  # the MLP hold-out file shares the row order of the blend (checked in mlp_member)
        rows = rows.assign(**mixed(rows["base"], pd.read_parquet(mlp_paths(args.mlp_columns)[0]), args.mlp_weight, args.member_columns))
    rows = rows.dropna(subset=["MVT_ID_mvt"])  # 19 hold-out rows carry no id
    train = frame(rows, (NM_TRAIN, *[t for t, _ in use]),
                  load_dep(sorted(C.TRAIN_DIR.glob("training_2025-0[17]-*.parquet")), ["CALLSIGN_flt"]),
                  args.neighbours == "rows")
    if args.mode == "price":
        report = {"sources": args.sources, "mlp_weight": args.mlp_weight, "neighbours": args.neighbours,
                  **price(train, rounds=args.rounds), "loao": price_loao(train)}
        log.info("%s", json.dumps(report, indent=1))
        tag = args.sources + ("" if args.neighbours == "none" else f".nb_{args.neighbours}")
        REPORT.with_suffix(f".{tag}.json").write_text(json.dumps(report, indent=1))
    else:
        serve(train, (NM_RANK, *[r for _, r in use]), args)


if __name__ == "__main__":
    main()
