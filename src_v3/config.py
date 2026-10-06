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
STEP_A_24H_EXTRA_S = 1150.0             # taxi on top of 86,400 s where a band has no 24-h row (`apply_stepA_v22`)
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
# Tree count of lgbm_p_fb_lirf_v23.txt (lirf_regime_v56.meta.json, v23_iter); the v56
# gate scales it, so a clean clone needs no v23 booster file.
P_FB_V23_ROUNDS = 182
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
# CatBoost member of the LIRF gate (third-edition N7, 2026-10-05): ordered target statistics on the categorical
# columns next to the LightGBM g3 gate. Rounds come from an early stop on months 11 and 12.
CATBOOST_GATE_PARAMS = dict(
    loss_function="Logloss", depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
    iterations=5000, od_type="Iter", od_wait=200,
    random_seed=GLOBAL_SEED, thread_count=12, verbose=250,
    train_dir=str(CACHE / "catboost_info_gate"),
)
CATBOOST_GATE_SHARES = (0.25, 0.5, 0.75, 1.0)   # CatBoost share of the calibrated gate; 0.5 is pre-registered
CATBOOST_BLEND_WEIGHTS = (0.3, 0.4, 0.5, 0.6)   # 0.3 and 0.5 pre-registered in WINNING_PLAN L3; v98 grid adds 0.4, 0.6
# Variant B (option B, v70): deeper trees, another seed and half the columns
# per split round, for diversity against the v67 member.
CATBOOST_PARAMS_B = {**CATBOOST_PARAMS, "depth": 10, "random_seed": 43, "rsm": 0.5}
CATBOOST_BY_TAG = {"v67": CATBOOST_PARAMS, "v69": CATBOOST_PARAMS, "v70": CATBOOST_PARAMS_B,
                   "v73": CATBOOST_PARAMS, "v75a": CATBOOST_PARAMS,
                   "v86c": CATBOOST_PARAMS,
                   "v86ca": {**CATBOOST_PARAMS, "train_dir": str(CACHE / "catboost_info_v86ca")},
                   # v97: own train_dir so the full-year fits run next to the hold-out CatBoost fits
                   "v97c": {**CATBOOST_PARAMS, "train_dir": str(CACHE / "catboost_info_v97c")},
                   "v97ca": {**CATBOOST_PARAMS, "train_dir": str(CACHE / "catboost_info_v97ca")}}

# -- NM clocks, WINNING_PLAN L1 and L2 (track B, 2026-09-26) ----------------
# 99.9 % of 2025 `MVT - AOBT_3` values lie in 27 to 3,600 s; the clip keeps
# linear leaves inside the training support.
NM_CLOCK_CLIP_S = 7200
NM_NEIGHBOUR_WINDOWS_S = (1800, 3600)   # WINNING_PLAN R7: 30 and 60 min
# Clock bias, v76: median of `y - (MVT - AOBT_3)` per key on clean rows.
CLOCK_BIAS_MIN_COUNT = 50               # medians on fewer rows carry their own label
CLOCK_BIAS_CLIP_S = 1800                # 2025 key medians lie in -300 to +70 s (p10/p90)
# MB5 study on the clock-era base (v76). Trials and threads bound the run to about 3 h.
TUNE_V76_TRIALS = 24
TUNE_V76_MAX_ROUNDS = 4000
TUNE_V76_EARLY_STOP = 60
TUNE_V76_THREADS = 12
TUNE_V76_MIN_GAIN_MSE = 150             # hold-out bar before a refit and upload
BLEND_V77_GRID = tuple(round(0.1 * i, 1) for i in range(11))   # served blend weights searched for v77
BLEND_V77_SERVED = (0.5, 0.5, 0.5)      # v75: CatBoost share, anchored share in LightGBM, in CatBoost
TUNE_R_NORM_V78_TRIALS = 40
# CatBoost variants priced on the hold-out for v80 (MB5 for the CatBoost class, 2026-09-27).
CATBOOST_VARIANTS_V80 = {
    "cat_l2": {**CATBOOST_PARAMS, "l2_leaf_reg": 10.0, "thread_count": 8,
               "train_dir": str(CACHE / "catboost_info_l2")},
    "cat_lossguide": {**CATBOOST_PARAMS, "grow_policy": "Lossguide", "max_leaves": 613,
                      "min_data_in_leaf": 515, "depth": 16, "thread_count": 8,
                      "train_dir": str(CACHE / "catboost_info_lossguide")},
}
# XGBoost third learner class (v84): depth-wise histogram trees, native categoricals.
XGB_PARAMS = dict(objective="reg:squarederror", tree_method="hist", max_depth=8, eta=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=50, reg_lambda=1.0,
                  max_cat_to_onehot=1, max_bin=256, nthread=12, seed=GLOBAL_SEED)
XGB_MAX_ROUNDS = 6000
XGB_EARLY_STOP = 200
NM_FORWARD_WINDOWS_S = (900, 1800)     # WINNING_PLAN L8: (t, t + 15 min] and (t, t + 30 min]
NM_FORWARD_EXTRA_WINDOWS_S = (300, 3600)  # v88 candidate: 5 and 60 min
PER_AIRPORT_THREADS = 6                 # v89: two per-airport refits run side by side
PER_AIRPORT_SEEDS = (42, 43, 44)        # WINNING_PLAN second edition B2; seed 42 is the v89 member

# -- ADS-B pushback clock of the scored flight, WINNING_PLAN A1 and A2 (2026-09-29) --
ADSB_DIR = ROOT / "external" / "adsblol"
ADSB_URL = ("https://github.com/adsblol/globe_history_{year}/releases/download/"
            "v{day}-planes-readsb-{feed}/v{day}-planes-readsb-{feed}.tar.{part}")
ADSB_FEEDS = ("prod-0", "staging-0", "mlatonly-0")   # daily feeds of ADSB.lol; prod-0 trained v91 to v97 (third edition K1)
ADSB_REFS_URL = "https://github.com/adsblol/globe_history_{year}.git/info/refs?service=git-upload-pack"
ADSB_PARTS = ("aa", "ab", "ac", "ad")   # a day splits into 2 GB parts; the last one is short
ADSB_BOX_DEG = (0.08, 0.12)             # lat, lon half-width of the kept box: about 9 km
ADSB_MAX_ALT_FT = 5000                  # keeps the climb-out that marks the take-off
ADSB_WORKERS = 12
ADSB_GROUND = -1                        # altitude code of a ground point in the points table
ADSB_LOOKBACK_S = 6 * 3600              # 2025 labels above 6 h are 24-h reporting artefacts, not taxi
ADSB_AFTER_S = 180                      # ADS-B lift-off lags `MVT` by 10 to 60 s on sample tracks
ADSB_SEGMENT_GAP_S = 300                # a longer silence ends the taxi segment (parked, transponder off)
ADSB_MOVING_KT = 1.0
ADSB_STAND_M = 80                       # WINNING_PLAN 4.3: stationary within 80 m of an OSM stand
ADSB_GATE_STOP_S = 120                  # WINNING_PLAN 4.3: a stop of 120 s or more at a stand
ADSB_PARKED_S = 900                     # a stop this long is a parked aircraft, not a runway queue
OPDI_RWY_BEFORE_S = 900                 # OPDI runway entry at most 15 min before `MVT`
OPDI_RWY_AFTER_S = 120                  # and at most 2 min after it (5 s event grid, clock skew)
OPDI_PUSH_MAX_S = 3 * 3600              # stand exits more than 3 h before the runway entry belong to another leg
# Stack on the served blend, v90. Rounds and leaves stay small: the stack reads few rows per airport.
STACK_V90_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                        verbose=-1, deterministic=True, num_threads=NUM_THREADS, seed=GLOBAL_SEED)
STACK_V90_ROUNDS = 600
STACK_V90_RES_CLIP_S = 3600             # a 24-h artefact must not steer the stack
STACK_V90_MIN_FOLD_DAYS = 5             # cross-fit by month once each month has this many ADS-B days
ADSB_RETRIES = 8                        # a DNS failure stopped the first July 2026 run
ADSB_BACKOFF_S = 15
ADSB_MAX_BAD_TRACE_SHARE = 0.01        # more corrupt traces than this means a broken archive, not one bad member
ADSB_CONNECT_TIMEOUT_S, ADSB_READ_TIMEOUT_S = 30, 120   # without a read timeout a stalled stream hangs forever
WATCH_START, WATCH_END = "2025-01-01", "2025-12-31"     # `watch_adsb_fetch.py`: every released 2025 day
WATCH_INTERVAL_S = 3600                 # a day takes about 6 min, so one hour without a change is a stall
WATCH_MAX_HOURS = 48
ADSB_OWN_M = 100                        # a point this close to the assigned stand counts as at the stand
STACK_V90_MIN_LOAO_ROWS = 300           # airports with fewer covered hold-out rows get no leave-one-out price
# METAR fetch (MODEL_ANALYSIS MC6): the IEM ASOS archive, the columns and range of the files that trained the models
METAR_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
METAR_COLUMNS = ("tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "alti", "mslp", "vsby", "gust",
                 "skyc1", "skyc2", "skyc3", "skyl1", "skyl2", "skyl3", "wxcodes", "ice_accretion_1hr",
                 "peak_wind_gust", "peak_wind_drct", "feel")
METAR_STATIONS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
METAR_REPORT_TYPES = (3, 4)             # routine and special reports
METAR_START, METAR_END = "2025-01-01", "2026-08-01"
METAR_RETRIES, METAR_BACKOFF_S, METAR_TIMEOUT_S = 5, 10, 300
# A4: neighbour pushback clocks; the order puts the clocks nearest the label first (v90, v91 notes)
PUSH_NB_CLOCKS = ("adsb_own_move", "opdi_stand", "adsb_push", "adsb_move", "opdi_first")
PUSH_NB_BACK_S = (1800, 3600)
# Artefacts each served version reads (MC3 manifest, `artefact_manifest.py`); raw data and open data excluded
SERVED_ARTEFACTS = {92: (
    "models/lgbm_r_all_v85_s*.txt", "models/lgbm_r_all_v85a_s*.txt", "models/lgbm_r_all_v85*.encoders.pkl",
    "models/catboost_r_all_v73.cbm", "models/catboost_r_all_v75a.cbm",
    "models/lgbm_pa_v89_*.txt", "models/lgbm_pa_v89a_*.txt",
    "models/lgbm_r_norm_lirf_v55_s*.txt", "models/lgbm_p_fb_lirf_v56.txt", "models/lirf_regime_v64.isotonic.pkl",
    "models/lirf_regime.encoders.pkl", "models/lirf_regime_v23.rate_maps.pkl",
    "models/tempo_p2575_rank.parquet", "models/plan_taxi_res_rank_v57.parquet", "models/plan_nm_taxi_rank_v65.parquet",
    "models/nm_clocks_*_v73.parquet", "models/nm_forward_rank_v85.parquet", "models/opdi_own_*_v90.parquet",
    "models/v2/cache/frame_rank.parquet", "models/v3/blend_v89_holdout.parquet",
    "external/adsblol/adsb_clock_rank.parquet", "external/adsblol/v92/adsb_clock_train.parquet")}
SERVED_ARTEFACTS[94] = (*SERVED_ARTEFACTS[92][:-1], "external/adsblol/v94/adsb_clock_train.parquet",
                        "models/v3/holdout_mlp.parquet", "models/v3/rank_mlp.parquet")
SERVED_ARTEFACTS[95] = tuple(p.replace("/v94/", "/v95/") for p in SERVED_ARTEFACTS[94])
# v112: v97 members and the g3 gate (the v104 base), the v105 stack with the MLP and the 24-epoch TabM member
SERVED_ARTEFACTS[112] = (
    "models/lgbm_r_all_v97_s*.txt", "models/lgbm_r_all_v97a_s*.txt", "models/lgbm_r_all_v97*.encoders.pkl",
    "models/lgbm_r_all_v97*.features.txt", "models/catboost_r_all_v97c.cbm", "models/catboost_r_all_v97ca.cbm",
    "models/lgbm_pa_v97_*.txt", "models/lgbm_pa_v97a_*.txt", "models/lgbm_pa_v97*.features.txt",
    "models/lgbm_r_norm_lirf_v55_s*.txt", "models/lgbm_p_fb_lirf_g3.txt", "models/lirf_regime_g3.isotonic.pkl",
    "models/lirf_regime_g3.features.txt", "models/lirf_regime.encoders.pkl", "models/lirf_regime_v23.rate_maps.pkl",
    "models/lirf_band_table_v30.json", "models/arr_art_op_link_rank_g3.parquet",
    "models/tempo_p2575_rank.parquet", "models/plan_taxi_res_rank_v57.parquet", "models/plan_nm_taxi_rank_v65.parquet",
    "models/nm_clocks_*_v73.parquet", "models/nm_forward_rank_v85.parquet", "models/opdi_own_*_v90.parquet",
    "models/v2/cache/frame_rank.parquet", "models/v3/blend_v97_holdout.parquet",
    "external/adsblol/adsb_clock_rank.parquet", "external/adsblol/v97/adsb_clock_train.parquet",
    "models/v3/holdout_mlp_v97.parquet", "models/v3/rank_mlp_v97.parquet",
    "models/v3/holdout_mlp_tabm_e24.parquet", "models/v3/rank_mlp_tabm_e24.parquet")
SERVED_ARTEFACTS[114] = tuple(p.replace("tabm_e24", "tabm_e32") for p in SERVED_ARTEFACTS[112])   # v114: 32-epoch member, three seeds
SERVED_ARTEFACTS[115] = (*(p.replace("tabm_e24", "tabm_v115") for p in SERVED_ARTEFACTS[112]),   # v115: X-Plane stands, ATFM slots
                         "external/adsblol/v115/adsb_clock_*.parquet", "external/xplane/*",
                         "models/atfm_slot_*.parquet")
# v93 stack study (`tune_stack_v93.py`): more 2025 ADS-B days give the stack about 9 times more rows
STACK_V93_MLP_WEIGHTS = (0.0, 0.1, 0.2)   # MLP hold-out: -190 at 0.1, -213 at 0.2, -69 at 0.3
STACK_V93_LEAVES = (31, 63)
STACK_V93_ROUNDS = (600, 1200)
# ADS-B specialist (`adsb_specialist.py`): trained on the 2025 months outside the stack's hold-out months
SPEC_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=127, min_data_in_leaf=100,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                   verbose=-1, deterministic=True, num_threads=NUM_THREADS, seed=GLOBAL_SEED)
SPEC_MAX_ROUNDS, SPEC_PATIENCE = 5000, 100
SPEC_STOP_EVERY = 7                     # every seventh day of the year stops the probe fit
SPEC_TAG = "v96"                        # specialist files priced by `tune_stack_v96.py`
# v96 stack study (`tune_stack_v96.py`): the 2025 label sits a median 230 s above the stand bound (tug disconnect)
STACK_V96_BOUND_OFFSETS_S = (60, 120, 180)
STACK_V96_BOUND_QUANTILES = (0.05, 0.10, 0.20)
# Unattended release (`auto_release.py`), 2026-09-30
AUTO_POLL_S = 300                      # check the download process every 5 min
AUTO_ROUNDS = (400, 600, 900)          # stack rounds priced after the rebuild; 600 stays unless beaten by the bar
AUTO_BAR_MSE = 150
AUTO_MAX_STACK_PRICE = -1500           # sanity: the 21-day stack priced -1,575 MSE; more days must not be worse
AUTO_MLP_WEIGHT = 0.15                 # v94
AUTO_BIG_MOVE_S = 3000
AUTO_MAX_BIG_MOVES = 10                # v92 moved 2 rows by more than 3,000 s, v93 and v94 none
AUTO_MAX_NEW_7200 = 5
AUTO_RESULT_POLLS, AUTO_RESULT_WAIT_S = 60, 10
# v98: serve-time parameters of the v97 members, chosen on the hold-out (`release_v98.py`)
V98_DEFAULT = (0.5, 0.5, 0.5)           # v97: CatBoost share, anchored share, per-airport share
V98_CAT_SHARES = (0.3, 0.4, 0.5, 0.6)
V98_ANCHOR_SHARES = (0.3, 0.5, 0.7)
V98_PER_AIRPORT_SHARES = (0.3, 0.5, 0.7)
V98_MLP_WEIGHTS = (0.1, 0.15, 0.2)
V98_STACK_ROUNDS = (400, 600)
HOLDOUT_ROWS = 344336                   # hold-out scale of every price
# v92 recipe, read by the parity test (C1)
V92_PER_AIRPORT_TAGS = ("v89",)         # B2 seeds were not ready at upload; v92 stacks on the v89 base
V92_STACK_ARGS = ("--sources", "both", "--stand-bound",
                  "--adsb-train", "external/adsblol/v92/adsb_clock_train.parquet")  # the 7 training days of v92
# v94: 21 training days (2025-01-01 to 01-15, 07-01 to 07-05, 07-15) and the MLP member at 0.15, the
# optimum of the quadratic through its hold-out prices (-190, -213, -69 MSE at 0.1, 0.2, 0.3)
V94_STACK_ARGS = ("--sources", "both", "--stand-bound", "--mlp-weight", "0.15",
                  "--adsb-train", "external/adsblol/v94/adsb_clock_train.parquet")
# v95 (`auto_release.py`, 2026-10-01): all 62 January and July 2025 days, 600 rounds
V95_STACK_ARGS = ("--sources", "both", "--stand-bound", "--mlp-weight", "0.15",
                  "--adsb-train", "external/adsblol/v95/adsb_clock_train.parquet", "--rounds", "600")
# Neural-net member (MLP with embeddings), priced for v92 and later on the hold-out.
MLP_HIDDEN = (1024, 512, 256)
MLP_DROPOUT = 0.1
MLP_EMB_MAX = 16
MLP_Z_CLIP = 8.0
MLP_Y_SCALE = 1000.0                    # the net learns y in thousands of seconds
MLP_LR = 2e-3
MLP_WEIGHT_DECAY = 1e-5
MLP_BATCH = 2048
MLP_EPOCHS = 30
MLP_THREADS = 8
MLP_Y_CLIP = 80_000                     # the tree members train on y up to 80,000 s (MB3)
MLP_BLEND_WEIGHTS = (0.1, 0.2, 0.3)
MLP_SERVE_EPOCH = 18                    # best stop-month epoch of the 30-epoch measure run (194.81 s)
MLP_SERVE_SEEDS = (42, 43, 44)
MLP_BLEND_W = 0.2                       # hold-out -213 MSE, clean rows -240, 8 of 9 airports
# TabM member (`tabm_member.py`): k MLPs with shared weights and piecewise-linear numeric embeddings, on the GPU
GPU_SITE = ROOT / ".gpu_site"           # CUDA build of torch 2.11.0, tabm 0.0.3, rtdl_num_embeddings 0.0.12
TABM_K = 16                             # submodels; each step trains all of them on the same batch
TABM_BLOCKS = 2
TABM_WIDTH = 512
TABM_DROPOUT = 0.1
TABM_BINS = 32                          # quantile bins per numeric column
TABM_BIN_ROWS = 500_000                 # training rows that set the bins
TABM_D_NUM = 16                         # embedding size of one numeric column
TABM_LR = 1e-3
TABM_WEIGHT_DECAY = 3e-4
TABM_BATCH = 1024
TABM_EPOCHS = 16
TABM_SERVE_SEEDS = (42, 43)
TABM_BLEND_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)
TABM_PREDICT_BATCH = 4096
TABM_SPECS = {   # priced and served TabM members: overrides of `tabm_member.Spec`
    "tabm": {"serve_epoch": 13},                      # v111: best stop-month epoch of 16 (163.54 s)
    "tabm_e24": {"epochs": 24, "serve_epoch": 23},    # 162.68 s; on the v105 base -343 MSE at 0.3
    "tabm_k32": {"k": 32, "width": 768},                # 162.88 s; on the v105 base -203 MSE at 0.3 (closed)
    "tabm_e32": {"epochs": 32, "serve_epoch": 27, "seeds": (42, 43, 44)},    # 162.19 s at epoch 27; on the v105 base -366 / -386 MSE at 0.3 / 0.4
    "tabm_e24_s43": {"epochs": 24, "serve_epoch": 23, "seeds": (43,)},   # seed spread of the hold-out member
    "tabm_v115": {"epochs": 32, "serve_epoch": 21, "seeds": (42, 43, 44), "columns": "v115"},   # 161.6 s at epoch 21; -390 MSE at 0.4
    "tabm_v116": {"epochs": 32, "serve_epoch": 21, "seeds": (42, 43, 44), "columns": "v116"},   # X-Plane stands at five airports
    "tabm_anchor": {"anchor_col": "mvt_aobt3"},         # 163.99 s; on the v105 base -105 MSE at 0.2 (closed)
}
TABM_GATE_SPEC = {"width": 256, "epochs": 12, "batch": 512}   # TabM member of the LIRF gate (`tabm_gate.py`)
TABM_ON_V105 = 0.3                      # share on top of the v105 base; hold-out -214 / -252 / -244 MSE at 0.2 / 0.3 / 0.4
# Error budget of v97 and the LIRF gap (`measure_budget_v97.py`, `measure_lirf_gap.py`), WINNING_PLAN third edition
RANK_ROWS = 344841                      # live scale of every ranking-file MSE
BUDGET_CLOCKS = ("adsb_own_move", "adsb_own_last", "adsb_push", "adsb_move", "adsb_seg",
                 "opdi_stand", "opdi_first", "mvt_aobt3")   # candidate clocks in seconds before `MVT`
BUDGET_NORMAL_CLASSES = ("clean", "fallback", "low")   # projected onto 2026; tail and 24-h are the remainder
BUDGET_MIN_CELL_ROWS = 1000             # a smaller airport-coverage cell takes the pooled value of its coverage class
BUDGET_D0_EDGES_M = (0, 150, 300, 600, 1000, 2000, 4000, 1e9)   # first ADS-B point to the assigned stand
BUDGET_ERR_EDGES_S = (120, 300, 600, 1200, 1800)    # error concentration of the clean rows
BUDGET_LONG_SEGMENT_S = 21000           # a ground segment this long reaches the 6 h lookback (`ADSB_LOOKBACK_S`)
LIRF_SD_BANDS_S = (1800, 3600, 7200, 14400, 70000)  # schedule-delay bands of the LIRF gate cost
ARTEFACT_MIN_CELL_ROWS = 30             # airline-month cells with fewer departures or arrivals are left out
ARTEFACT_SMOOTH_K = 10                  # arrivals of prior weight toward the airport share, airline-day share
ARTEFACT_SHARE_EDGES = (0.05, 0.1, 0.2, 0.4)   # buckets of the airline-day arrival artefact share
LATE_SD_S = 1800                        # a departure this late against the schedule carries most of the gate cost
# EUROCONTROL ATFM slot adherence per airport and day (`build_atfm_slot.py`); take-off times only, not label-derived
EC_CSV_URL = "https://www.eurocontrol.int/performance/data/download/csv/{}_{}.csv"
ATFM_SLOT_YEARS = (2025, 2026)
ATFM_SLOT_DIR = ROOT / "external" / "eurocontrol"
# X-Plane Scenery Gateway stands (`build_xplane_stands.py`), GPL v2 or later: positions where OSM has no assigned stand
XPLANE_DIR = ROOT / "external" / "xplane"
XPLANE_API = "https://gateway.x-plane.com/apiv1/scenery/{}"
XPLANE_SCENERY = {"EDDM": 111124, "EHAM": 111943, "LEMD": 99851, "LEBL": 102255, "LFPG": 111022}   # recommended packs on 2026-10-06
STANDS_OSM_XPLANE = XPLANE_DIR / "stands_osm_xplane.parquet"
# Route-distance clock (`build_route_clock.py`, third-edition K3)
ROUTE_SNAP_M = 400                      # snap radius of `build_osm_taxi_paths.snap_to_graph`
ROUTE_CUTOFF_M = 12000                  # longer routes than the largest airport span are not taxi routes
ADSB_TAXI_SPEED_MS = 7.0                # about 14 kt, a typical taxi speed; the trees learn the airport scale
