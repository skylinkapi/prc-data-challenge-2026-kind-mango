"""Single config for the v2 stack. All hand-picks live here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "training"
RANK = ROOT / "submission" / "ranking.parquet"
SUB_TMPL = ROOT / "submission" / "submitting.parquet"
MODELS = ROOT / "models" / "v2"
CACHE = ROOT / "models" / "v2" / "cache"
METAR_DIR = ROOT / "external" / "metar"
LOG_DIR = ROOT / "models" / "v2" / "logs"

for d in (MODELS, CACHE, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGETS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
# Airports where |y - sd| <= 5 is a real reporting spike (D1, T8).
FALLBACK_AIRPORTS = ("EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM")
# Airports where the same window is only the punctuality background (A2).
NO_FALLBACK_AIRPORTS = ("EDDF", "EHAM", "LSZH")

# Label classes (A1). y in seconds.
FB_TOL = 5                   # |y - sd| <= FB_TOL -> fallback (D1, T8)
Y_MIN = 30
Y_CLEAN_MAX = 7200
Y_TAIL_MAX = 80000
Y_24H_MIN = 80000

# Post-processing bounds (E1).
CLIP_LOW = 30
CLIP_HIGH = 100000

# Hold-out policy (P3): months 1 and 7 are hold-out; everything else fits.
FIT_MONTHS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
HOLDOUT_MONTHS = (1, 7)

# Categorical vocabulary keys shared by train and score.
CAT_COLS = ("ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
            "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "flt_prefix",
            "stand_prefix", "hour", "dow", "month")

MIN_COUNT = 20                # A7, swept in C4.
SEED = 42
SEEDS_ENSEMBLE = (42, 43, 44)  # C5.
NUM_THREADS = 4               # F5.

# Base LightGBM defaults (C2, C4). Constant leaves; use_missing True.
LGB_BASE = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.85,
    bagging_fraction=0.9,
    bagging_freq=5,
    lambda_l2=1.0,
    verbose=-1,
    use_missing=True,
    num_threads=NUM_THREADS,
)

LGB_REGIME = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=100,
    feature_fraction=0.85,
    bagging_fraction=0.9,
    bagging_freq=5,
    lambda_l2=1.0,
    verbose=-1,
    use_missing=True,
    num_threads=NUM_THREADS,
)

# Dirichlet prior for the null-flight tail head (D1).
TAIL_PRIOR = 2.0
