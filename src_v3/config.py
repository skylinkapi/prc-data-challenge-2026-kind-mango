"""MC1: one config, every constant with its provenance.

The fifteenth pass identifies at least 34 hand constants in the served path
(appendix A5). This module holds them once, with a comment naming the
finding, the earlier pass or the file where each value came from. Any
sweep of a constant (MC2) reads from here.
"""
from __future__ import annotations

from pathlib import Path

# -- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "training"
RANK = ROOT / "submission" / "ranking.parquet"
SUB_TMPL = ROOT / "submission" / "submitting.parquet"
MODELS = ROOT / "models" / "v3"
CACHE = MODELS / "cache"
LOG_DIR = MODELS / "logs"
METAR_DIR = ROOT / "external" / "metar"

for d in (MODELS, CACHE, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -- airports ---------------------------------------------------------------
TARGETS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
           "LFPG", "LIRF", "LSZH", "LTFM")
# ICAO -> country code, MODEL_ANALYSIS X1 and MF3.
ICAO_COUNTRY = {"EDDF": "DE", "EDDM": "DE", "EGLL": "GB", "EHAM": "NL",
                "LEBL": "ES", "LEMD": "ES", "LFPG": "FR", "LIRF": "IT",
                "LSZH": "CH", "LTFM": "TR"}
# Airports where fallback |y - sd| is a real reporting spike (T8, D2).
# MD1 will redefine this per-airport from the residual grid.
FALLBACK_AIRPORTS_PROVISIONAL = ("EDDM", "EGLL", "LEBL", "LEMD", "LFPG",
                                 "LIRF", "LTFM")

# -- label classes (MP10) ---------------------------------------------------
Y_LOW_MAX = 30          # low class: 0 < y < 30
Y_CLEAN_MIN = 30
Y_CLEAN_MAX = 7200
Y_TAIL_MAX = 80000      # 7200 < y <= 80000 is tail
Y_24H_MIN = 80000       # y > 80000 is 24-h
FB_TOL_DEFAULT = 60     # deployed head uses 60, MD1 replaces this per airport

# -- fold definitions (MP1) -------------------------------------------------
# Six folds; each holds out one Jan-side month and one Jul-side month, and
# trains on the other ten. The (1, 7) fold reproduces the historical split
# and is reported alone next to the fold mean and standard error.
FOLDS = [
    (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12),
]
# For fold f = (m1, m2) the early-stop months are the two training months
# adjacent to m1 and m2 (MP1).
EARLY_STOP_BY_FOLD = {
    (1, 7): (2, 8), (2, 8): (3, 9), (3, 9): (4, 10),
    (4, 10): (5, 11), (5, 11): (6, 12), (6, 12): (1, 7),  # last fold wraps
}
# The last fold's early-stop set wraps to (1, 7); flag this and prefer a
# 5-fold protocol when the wrap creates a leak into the (1, 7) reference.
FOLD_WRAP_WARN = {(6, 12): (1, 7)}

# -- interval decision rule (MP2) -------------------------------------------
FOLD_INTERVAL_ALPHA = 0.10  # 90 % interval

# -- categorical vocabulary (train and serve, MC5) --------------------------
CAT_COLS = ("ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
            "flt_prefix", "stand_prefix", "hour_local", "dow", "doy_sin",
            "doy_cos")

# -- served hand constants (A5, MC2) ----------------------------------------
# LIRF head, deployed values. Each is a constant the served prediction reads;
# MC2 sweeps every one of them on the folds.
P24_ITY = 5.0 / 6.0                     # predict_v30.py:36
ITY_SD_THRESHOLD = 70_000               # predict_v30.py:37
NORMAL_MEAN_LIRF = 1_150                # predict_v30.py:39, seventh pass
STEP_A_SD_THRESHOLD = 14_400            # LIRF, null-record, sd > this
STEP_A_GATE = 14_400                    # gate threshold in the deployed table
K_SMOOTH = 30                           # LIRF rate-map smoothing
MIN_COUNT = 20                          # target-encoder floor
OPDI_LAG_S = 600                        # OPDI live tempo lag, features_opdi_live
METAR_TOL_MIN = 45                      # features_weather backward join
TURNAROUND_TOL_H = 12                   # features_turnaround backward link
STAND_GAP_MAX_S = 1500                  # features_turnaround stand-gap window
TEMPO_WINDOWS_MIN = (30, 60)            # src_v2/frame.py
ORDER_WINDOW_MIN = 30                   # src_v2/frame.py
CONGESTION_WINDOWS_MIN = (15, 30, 60)   # features_congestion_v2
PLAN_RES_CLIP_S = 3600                  # build_plan_taxi_res.py
STOP_FRAC_OLD = 0.12                    # historical random-row stop, P4
LOW_VIS_KM = 5.0
VERY_LOW_VIS_KM = 1.5
LOW_CEILING_FT = 500
CEILING_DEFAULT_FT = 25_000
DEICING_TMPC = 3.0
DEICING_DEWPT_SPREAD_C = 2.0            # MF1, freezing-fog trigger
NIGHT_HOURS_LOCAL = (23, 24, 0, 1, 2, 3, 4, 5)  # night-restriction sweep, MF3

# -- LightGBM defaults; MB5 retunes each model class -------------------------
NUM_THREADS = 4                         # MC4 pins this; -1 was non-deterministic
DETERMINISTIC = True                    # MC4
GLOBAL_SEED = 42
LGB_LINEAR_BASE = dict(
    objective="regression", metric="rmse",
    linear_tree=True, linear_lambda=1.0,   # retuned in MB5
    num_leaves=127, learning_rate=0.03,
    min_data_in_leaf=200, feature_fraction=0.85,
    bagging_fraction=0.9, bagging_freq=5, lambda_l2=1.0,
    verbose=-1, use_missing=True,
    deterministic=DETERMINISTIC, num_threads=NUM_THREADS,
    seed=GLOBAL_SEED, bagging_seed=GLOBAL_SEED,
    feature_fraction_seed=GLOBAL_SEED,
)

# -- CatBoost second model class, MB7 / WINNING_PLAN L3 ---------------------
# Depth 8 and RMSE follow the plan (L3). A fixed thread count keeps CPU fits
# deterministic; early stop reads the v46 stop months 11 and 12.
CATBOOST_PARAMS = dict(
    loss_function="RMSE", depth=8, learning_rate=0.08, l2_leaf_reg=3.0,
    iterations=5000, od_type="Iter", od_wait=200,
    random_seed=GLOBAL_SEED, thread_count=12, verbose=250,
    train_dir=str(CACHE / "catboost_info"),
)
CATBOOST_BLEND_WEIGHTS = (0.3, 0.5)     # pre-registered in WINNING_PLAN L3
