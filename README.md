# PRC Data Challenge 2026 — team `kind-mango`

Solo entry. Predicts **taxi-out time in seconds** (`TAXITIME_SEC_mvt`) for
departing flights at 10 major European airports. Metric is RMSE against
airport-reported truth.

- **Challenge home:** https://ansperformance.eu/study/data-challenge/dc2026/
- **This repo:** https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango
- **Current leaderboard best:** **561.04 s RMSE** (`kind-mango_v9.parquet`, 5-model ensemble)

## License

**GNU GPLv3.** See `LICENSE`. Prize-eligible per the challenge rules: source
public, open-data only, documented reproduction path.

## Ethics — features we deliberately did NOT use

The challenge brief explicitly forbids "exploiting the ranking process".
Two features that would have given a large leaderboard boost but that we
deliberately avoided:

1. **`AOBT_3_flt` on ranking DEP rows** — present on 98.5 % of ranking DEP
   rows; `(MVT_TIME − AOBT_3_flt)` reconstructs the hidden label to within
   the per-airport reporting-offset noise (150–250 s equivalent leaderboard
   RMSE). The brief warns "using them is probably exploiting the ranking
   process (explicitly forbidden)".
2. **Per-flight own-flight OPDI event reconstruction** — `exit-parking_position`
   + `entry-runway` from the current flight's own OPDI events would give
   its actual taxi-out to the second. Espinielli clarified on Discord that
   OPDI is open data BUT that OPDI has "no off-block milestone" — we read
   this as not endorsing per-flight AOBT reconstruction as intended use.

Both would place us in the top-3 by score. We chose eligibility over rank.

## Quick start

Python 3.13, one virtualenv.

```bash
pip install pandas pyarrow numpy scikit-learn lightgbm optuna openpyxl networkx
# Fetch open external data (~50 min incl. rate-limited APIs)
python src/build_eurocontrol_daily.py     # daily ATFM per airport
python src/build_osm_stands.py            # OSM stand centroids
python src/build_osm_taxi_paths.py        # OSM taxiway graph -> path lengths
python src/build_vrs_aircraft.py          # aircraft type metadata
python src/fetch_opdi_events.py           # OPDI ADS-B events (~50 min, 560 MB)
# Then run the METAR fetch (see scratchpad/fetch_metar.py or bootstrap script)
# Train the best single model
python src/train_lgbm_v15.py              # OPDI live-taxi + full stack
# Predict on ranking
python src/predict_ensemble_v15.py kind-mango_v<n>.parquet
mc cp submission/kind-mango_v<n>.parquet osn/prc-2026-<team>/
```

Full reproduction steps are in [`REPRODUCE.md`](REPRODUCE.md).

## Layout

```
src/                     # 30+ Python modules
  baseline.py            # 4 floor baselines
  features_weather.py    # METAR + crosswind
  features_congestion.py # rolling airport-load counts
  features_eurocontrol.py# daily ATFM per-airport features
  features_operator.py   # operator historic taxi encoders
  features_taxi_distance.py  # OSM stand->runway haversine
  features_advanced.py   # physical proxies (queue, runway diversity, ades delay)
  features_osm_path.py   # OSM taxiway graph shortest-path
  features_opdi.py       # OPDI rolling event counts (leak-free)
  features_opdi_live.py  # OPDI rolling live-taxi mean/median (OTHER flights)
  features_vrs.py        # VRS aircraft-type metadata (unused in final)
  features_ssl.py        # semi-supervised drift (unused in final)
  features_opdi_extended.py  # more OPDI event types (unused in final)
  build_eurocontrol_daily.py, build_osm_stands.py,
  build_osm_taxi_paths.py, build_vrs_aircraft.py,
  build_opdi_taxi.py, build_opdi_taxi_v2.py,
  fetch_opdi_events.py    # external-data ingestion scripts
  train_lgbm_v[1-17].py   # progressive model versions
  train_lgbm_lirf.py      # LIRF-only specialist (unused)
  train_lgbm_robust.py    # Huber / log-target comparison
  train_tail_classifier.py, train_tail_oof.py, train_tail_oof_lite.py
  train_dq_classifier.py  # BLOCK==SCHED detector (failed)
  train_lgbm_quantile.py  # quantile regression models (unused)
  tune_lgbm.py            # Optuna hyperparameter search
  ensemble*.py, predict*.py

external/                # open data caches (gitignored)
  metar/       Iowa State METAR CSVs per ICAO
  eurocontrol/ 5 NM performance Excels + consolidated daily parquet
  osm/         OSM stand positions + taxiway graphs
  airports/    OurAirports runways.csv + airports.csv
  vrs/         VRS StandingData aircraft types
  opdi/        OPDI event chunks + consolidated events_all.parquet
training/    organiser-provided 2025 monthly parquets (12 files, gitignored)
submission/  ranking.parquet, submitting.parquet, kind-mango_v*.parquet + result.json
models/      trained model + feature list + encoders + weights (gitignored)
docs/        PRC brief
```

## Data sources (all open)

| Source | Data | License |
|---|---|---|
| PRC / OpenSky Network | Movements at 10 EU airports (2025 train, 2026 ranking) | Challenge data licence |
| Iowa State ASOS | METAR reports, hourly, per station | Public domain |
| OurAirports | Runway coordinates, headings, lengths | Public domain |
| OSM Overpass | `aeroway=parking_position`, `taxiway`, `runway` | ODbL |
| Eurocontrol NM | Daily ATC pre-dep delay, ATFM slot adherence, airport traffic, arrival ATFM delay | Open (`www.eurocontrol.int/performance/data`) |
| **OPDI** (PRC + OSN) | Flight events (entry-runway, entry-taxiway, exit-parking_position, ...) | Open (`opdi.aero`), CC-BY 4.0, confirmed open for challenge by organizer |
| VRS StandingData | Aircraft-type → manufacturer/engines/model lookup | Open (`github.com/vradarserver/standing-data`) |

## Model — best submission

**Ensemble of 5 LightGBM models**, blended by grid-search on 2025 hold-out:

```
0.55 × lgbm_v15 + 0.15 × lgbm_v16 + 0.15 × lgbm_v7 + 0.10 × lgbm_v11 + 0.05 × lgbm_v10
```

All five models are gradient-boosted regressors with **MSE loss**, sharing the same
Optuna-tuned hyperparameters. They differ in feature set:

- **v15** (best solo): 124 features, adds OPDI live-taxi rolling mean/median from
  OTHER flights
- **v16**: 128 features (v15 + semi-supervised drift; solo worse, but blend-useful)
- **v7**: 97 features, winter-weighted training on v5 base
- **v11**: 115 features, v10 + OPDI rolling event counts
- **v10**: 106 features, v9 + OSM real taxi path length

## Feature families used in v15 (best single model)

**Categorical (10)**: airport, destination, runway, stand, aircraft type, wake,
market segment, operator, flight rule, flight type.

**Numeric (114)**:
- Time (4): hour, dow, month, sched_delay
- METAR weather (13): wind cross/head, gust, vis, low_vis flags, ceiling, precip / snow / thunder / freezing
- Rolling congestion (10): DEP/ARR counts in prev 15/30/60 min, same-runway counts, next-10-min queue proxy
- `flt_null` flag
- Eurocontrol daily (18): pre-dep delay, ATFM slot adherence, airport traffic — plus one-day lag
- Operator historic encoders (16): median + std per (op × airport / runway / hour-bin / stand)
- Physical: haversine + real OSM path stand→runway, turn count, path/haversine ratio
- Advanced physics (5): scheduled pushback load, runway diversity, secs since last DEP on runway, rwy bank intensity, ADES arrival ATFM delay
- OPDI raw event counts (9): entry-runway / entry-taxiway / exit-parking rolling counts
- OPDI live-taxi (9): rolling mean / median / count of ACTUAL observed taxi from OTHER completed flights

**Best hyperparameters (Optuna, trial 31 out of 60):**

```
learning_rate:      0.0228
num_leaves:         440
min_data_in_leaf:   76
feature_fraction:   0.674
bagging_fraction:   0.937
lambda_l1:          0.089
lambda_l2:          0.342
min_gain_to_split:  1.83
early stop / best_iter around 1000-1200
```

## Local hold-out progression (Jan + Jul 2025)

| Model | RMSE (s) | Notes |
|---|---|---|
| Global median | 461.3 | reference floor |
| Per-airport median | 420.8 | |
| Grouped median + sched-delay slope | 365.1 | strongest non-ML |
| HistGradientBoosting | 313.8 | first ML |
| LightGBM v1 | 302.2 | base + weather + congestion |
| LightGBM v2 (+ Eurocontrol) | 298.8 | |
| LightGBM v3 (+ operator encoders) | 298.6 | |
| LightGBM v4 (+ OSM haversine) | 297.8 | |
| LightGBM v5 (Optuna-tuned) | 295.49 | |
| LightGBM v6 (+ flt_null flag) | 295.56 | (same) |
| LightGBM v7 (winter-weighted) | 295.46 | |
| LightGBM v9 (+ advanced physical) | 294.58 | |
| LightGBM v10 (+ OSM real path) | 294.40 | |
| LightGBM v11 (+ OPDI raw counts) | 294.18 | |
| **LightGBM v15 (+ OPDI live-taxi)** | **293.09** | best solo |
| **Ensemble v9 (5-model)** | **292.77** | best overall |

## Leaderboard submissions

All submissions on the NEW-template scoring set (344,841 DEP rows across
10 airports × Jan+Jul 2026).

| Upload | Model | Local RMSE | Live RMSE |
|---|---|---|---|
| kind-mango_v1.parquet | v5+v6 (0.54/0.46) | 295.11 | 562.44 |
| kind-mango_v2.parquet | v5+v6+v7 | 294.90 | 562.28 |
| kind-mango_v3.parquet | v5+v7+v9+v10 | 293.82 | 561.52 |
| kind-mango_v4.parquet | v7+v9+v10+v11 | 293.64 | 561.64 |
| kind-mango_v5.parquet | (v3 rebroadcast) | — | 561.52 |
| kind-mango_v6.parquet | v15 solo | 293.09 | 561.19 |
| kind-mango_v7.parquet | v7+v10+v11+v15 | 292.81 | 561.06 |
| kind-mango_v8.parquet | v16 solo | 293.70 | 561.81 |
| **kind-mango_v9.parquet** | **v7+v10+v11+v15+v16** | **292.77** | **561.04** ← best |

## Approaches tried and RULED OUT

For transparency, and to save future replicators from the same dead-ends:

- **Huber loss + log-target regression**: much worse (408 s hold-out)
- **Tighter target range [30, 21600] to model longer taxis**: hurt RMSE 6+ s
- **LIRF specialist model**: converged in same range as general model
- **Tail-mixture (P(y > 3600s) classifier + blend)**: signal exists (AUC 0.97)
  but blending false positives hurts more than true positives help
- **P_tail as regressor feature (naive)**: target leakage
- **P_tail as regressor feature (5-fold OOF)**: feature distribution shift
- **Data-quality classifier (BLOCK==SCHED detector)**: AUC 0.51 (random)
- **Semi-supervised drift features from ranking rows**: worse on both local + live
- **Quantile regression (q=0.5/0.55/0.60) ensemble**: near-zero effect
- **Extended OPDI event counts (exit-taxiway, entry-threshold, etc.)**: mildly worse
- **VRS aircraft-type derived features (manufacturer, engines)**: overfit, worse
- **Post-hoc airport-level shift calibration**: near-zero effect (model already unbiased)

## Reproduction

See [`REPRODUCE.md`](REPRODUCE.md).

## Acknowledgements

- Eurocontrol Performance Review Commission (PRC), OpenSky Network — challenge data + infrastructure
- Iowa State University Environmental Mesonet — METAR archive
- David Megginson — OurAirports open data
- OpenStreetMap contributors — stand and runway geometry
- OPDI project (PRC + OSN) — ADS-B derived flight events
- Virtual Radar Server community — StandingData aircraft-type lookup
