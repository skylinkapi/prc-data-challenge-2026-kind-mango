# PRC Data Challenge 2026 — team `kind-mango`

Solo entry. Predicts **taxi-out time in seconds** (`TAXITIME_SEC_mvt`) for
departing flights at 10 major European airports. Metric is RMSE against
airport-reported truth.

Challenge home: https://ansperformance.eu/study/data-challenge/dc2026/

## License

**GNU GPLv3.** See `LICENSE`. Prize-eligible per the challenge rules: source
public, open-data only, documented reproduction path.

## Quick start

Python 3.13, one virtualenv, four commands.

```bash
pip install pandas pyarrow numpy scikit-learn lightgbm optuna openpyxl
python src/build_eurocontrol_daily.py     # consolidate ATFM Excel -> parquet
python src/build_osm_stands.py            # fetch OSM stand positions
python src/train_lgbm_v5.py               # train the Optuna-tuned model
python src/predict_ensemble.py <name>.parquet   # write submission
```

The four external data fetches (Iowa State METAR, OurAirports runways,
Eurocontrol daily delays, OSM stands) are automated in `src/build_*.py` and
`external/metar/` bootstrap. Run each once; outputs cache to `external/`.

## Layout

```
src/                # all code, one file per stage or feature module
  baseline.py                   # floor baselines (4 grouped-median models)
  features_weather.py           # METAR join + crosswind on active runway
  features_congestion.py        # rolling airport-load counts
  features_eurocontrol.py       # daily ATFM per-airport features
  features_operator.py          # operator historic taxi-time encoders
  features_taxi_distance.py     # OSM stand-to-runway haversine distance
  build_eurocontrol_daily.py    # 5 Eurocontrol Excel -> one parquet
  build_osm_stands.py           # Overpass query -> stand centroids
  train_lgbm_v5.py              # Optuna-tuned LightGBM
  train_lgbm_v6.py              # v5 + data-quality cleaning
  tune_lgbm.py                  # Optuna hyperparameter search
  ensemble.py                   # v5+v6 blend weight search
  predict.py                    # single-model submission pipeline
  predict_ensemble.py           # blended-model submission pipeline

external/           # third-party open data, one dir per source
  metar/            # Iowa State ASOS, per-ICAO CSV
  eurocontrol/      # 5 daily .xlsx + consolidated daily_features.parquet
  osm/              # OSM stand positions per airport
  airports/         # OurAirports runways.csv + airports.csv

training/           # organiser-provided 2025 monthly parquets (12 files)
submission/         # organiser-provided ranking.parquet + submitting.parquet
                    # + kind-mango_v<n>.parquet outputs + result JSONs

models/             # trained models + feature lists + encoders + weights
docs/               # PRC brief

.gitignore          # excludes credentials, source datasets, model artifacts
```

## Data sources (all open)

| Source | Data | License |
|---|---|---|
| PRC / OpenSky Network | Movements at 10 EU airports (2025 train, 2026 ranking) | Challenge data licence |
| Iowa State ASOS | METAR reports, hourly, per station | Public domain |
| OurAirports | Runway coordinates, headings, lengths | Public domain |
| OSM Overpass | `aeroway=parking_position` stand centroids | ODbL |
| Eurocontrol NM | Daily ATC pre-dep delay, ATFM slot adherence, airport traffic, arrival ATFM delay | Open (`www.eurocontrol.int/performance/data`) |

## Model

Gradient-boosted regression, LightGBM. Two models trained, blended at
inference.

**Features (97–98 total, per model)**

- Categorical: airport (ADEP), destination, runway, stand, aircraft type,
  wake, market segment, operator, flight rule, flight type
- Time: hour, day-of-week, month, scheduled-departure delay
  (`MVT − SCHED`, clipped)
- Weather (METAR, ≤ 45 min before movement): wind speed, crosswind on runway,
  headwind, gust, visibility (km), low-vis flag (< 5 km), very-low-vis flag
  (< 1.5 km), ceiling (ft), low-ceiling flag, precipitation flag, snow flag,
  thunderstorm flag, freezing flag
- Congestion (from movement data itself): rolling DEP and ARR counts in prev
  15 / 30 / 60 min at the airport, same-runway DEP counts, forward 10-min
  DEP queue proxy
- Eurocontrol daily (per airport): total pre-departure delay minutes, ATC
  pre-departure delay minutes, ATFM slot adherence counts, arrival ATFM
  delay by cause, airport IFR traffic totals, plus one-day-lag values
- Operator historic (target encoders fit on training-months only): median
  and std of taxi-out per (operator × airport), (operator × airport × runway),
  (operator × airport × hour-bin), (operator × airport × stand)
- Physical: haversine distance from stand centroid to runway threshold in
  meters (OSM + OurAirports)
- Data-quality: `flt_null` flag when the NM flight-side join failed

**Training**

- Corpus: 2,083,634 DEP rows across 10 airports, 2025
- Hold-out: Jan and Jul 2025 (mirrors 2026 ranking window)
- Target filter: `[30, 7200]` seconds
- Loss: MSE (Huber and log-target tested, both worse — see `train_lgbm_robust.py`)
- Hyperparameters: 60-trial Optuna TPE, cached feature dataframes

**Best hyperparameters (Optuna, trial 31)**

```
learning_rate:    0.0228
num_leaves:       440
min_data_in_leaf: 76
feature_fraction: 0.674
bagging_fraction: 0.937
lambda_l1:        0.089
lambda_l2:        0.342
min_gain_to_split: 1.83
best_iteration:   992
```

**Ensemble**

Blend weights from grid search on hold-out:

```
0.54 * lgbm_v5  +  0.46 * lgbm_v6
```

## Local hold-out results (Jan + Jul 2025)

| Model | RMSE (s) |
|---|---|
| Global median | 461.3 |
| Per-airport median | 420.8 |
| Grouped median (5 keys) | 390.4 |
| Grouped median + sched-delay slope | 365.1 |
| HistGradientBoosting (all features) | 313.8 |
| LightGBM v1 (base + weather + congestion) | 302.2 |
| LightGBM v2 (+ Eurocontrol) | 298.8 |
| LightGBM v3 (+ operator encoders) | 298.6 |
| LightGBM v4 (+ OSM distance) | 297.8 |
| LightGBM v5 (Optuna) | 295.49 |
| LightGBM v6 (v5 + data-quality) | 295.56 |
| **Ensemble 0.54·v5 + 0.46·v6** | **295.11** |

## Leaderboard result

`kind-mango_v1.parquet`, ensemble model, uploaded 2026-09-04:
**562.44 s RMSE**. Score gap of ≈ 267 s vs hold-out.

Likely drivers of the gap (documented, not yet mitigated):

1. **`sched_delay` drift**: Jan 2026 flights depart 130–360 s later off
   schedule than 2025 training baseline (LTFM +360 s, EDDM +230 s, LSZH +180
   s, EDDF +150 s, EHAM +130 s). `sched_delay` is our #2 feature by gain;
   the model was fit on 2025 distribution and reads 2026 values at
   out-of-training points.
2. **Real regime shifts**: EHAM Jan 2026 movement volume is 88 % of Jan 2025
   (winter cancellations per Schiphol H1 2026 report). `_flt` NULL rate at
   EHAM tripled from 1.65 % to 4.96 %.
3. **Truth-set tail**: The organiser's truth set retains rows where
   `BLOCK_TIME` fell back to `SCHED_TIME`, which encodes pushback delay as
   taxi time. Our training filter drops these; the scorer keeps them.
4. **Target regime**: The truth values include an unknown share of ~24-hour
   BLOCK_TIME offset rows (arnavhm13's Discord report). A single such row
   contributes ≈ 186 s to RMSE.

## Reproduction

Full end-to-end from a clean clone:

```bash
git clone <repo>
cd prc-data-challenge-2026-kind-mango

# 1. Environment
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install pandas pyarrow numpy scikit-learn lightgbm optuna openpyxl

# 2. Get the challenge data into training/ and submission/
#    (organiser bucket: prc-2026-datasets on s3.opensky-network.org)
#    Not scripted here; run the `mc mirror` from the challenge docs.

# 3. Fetch external open data (~40 min total, incl. Overpass rate limits)
python src/fetch_metar.py          # or the inline script in scratchpad
python src/build_eurocontrol_daily.py
python src/build_osm_stands.py
#   (OurAirports CSVs are two direct HTTP GETs; see external/airports/README)

# 4. Train (fits into ~2 hours end-to-end on a laptop CPU)
python src/train_lgbm_v5.py    # ~2 min feature build + Optuna best config refit
python src/train_lgbm_v6.py    # ~5 min

# 5. Predict + upload
python src/predict_ensemble.py kind-mango_v<n>.parquet
mc cp submission/kind-mango_v<n>.parquet osn/prc-2026-kind-mango/
```

## Submissions

Team bucket: `prc-2026-kind-mango`.

Cap: **3 submissions per day** (resets 00:00 UTC). Bucket cap: **1 GB**.

Team score is the best RMSE across all uploaded submissions.

| Upload | Model | Local RMSE | Leaderboard | Notes |
|---|---|---|---|---|
| kind-mango_v1.parquet | 0.54·v5 + 0.46·v6 ensemble | 295.11 | 562.44 | first valid, after July-fix template |

## Acknowledgements

- Eurocontrol Performance Review Commission, OpenSky Network — challenge data
  and infrastructure
- Iowa State University Environmental Mesonet — METAR archive
- David Megginson — OurAirports open data
- OpenStreetMap contributors — stand and runway geometry
