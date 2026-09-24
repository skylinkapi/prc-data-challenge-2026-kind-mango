# Reproduction guide

End-to-end from a fresh clone to the v33 submission file (live 301.87 s, team best).

Estimated time: **~2 hours** on a modern laptop CPU (16 GB RAM recommended).
Bandwidth: **~1 GB** of external data downloads.

## 0. Prerequisites

- Python 3.13 (3.10+ likely works)
- ~10 GB free disk (organizer data + external + models)
- Optional: MinIO client `mc` for uploading submissions

## 1. Environment

```bash
git clone https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango.git
cd prc-data-challenge-2026-kind-mango

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install pandas pyarrow numpy scikit-learn lightgbm optuna openpyxl networkx minio catboost
```

## 2. Get organiser data

Get an OSN account, generate S3 access + secret key via the MinIO console
(port 443, not 9443 — see challenge docs).

```bash
mc alias set osn https://s3.opensky-network.org <ACCESS_KEY> <SECRET_KEY>
mc mirror osn/prc-2026-datasets/ training/         # ~285 MB, 12 monthly parquets
mc cp osn/prc-2026-datasets/ranking.parquet  submission/
mc cp osn/prc-2026-datasets/submitting.parquet submission/
```

## 3. Fetch external open data

Each fetches to the corresponding `external/*` folder.

```bash
# METAR — Iowa State ASOS archive, 11 airports × 2 years  (~32 MB, ~1 min)
# No committed fetch script exists. Download one CSV per ICAO to external/metar/.

# OurAirports airport + runway CSVs  (~15 MB, seconds)
mkdir -p external/airports
curl -o external/airports/runways.csv  https://davidmegginson.github.io/ourairports-data/runways.csv
curl -o external/airports/airports.csv https://davidmegginson.github.io/ourairports-data/airports.csv

# Eurocontrol NM performance (5 Excels + one consolidated parquet, ~500 MB, 5 min)
python src/build_eurocontrol_daily.py

# OSM stand positions via Overpass (~2 min, may retry on rate limits)
python src/build_osm_stands.py

# OSM taxiway graph + shortest paths (~5 min)
python src/build_osm_taxi_paths.py

# VRS aircraft-type metadata  (~1 min)
python src/build_vrs_aircraft.py

# OPDI ADS-B events for 10 airports Jan 2025 - Aug 2026 (~50 min, 14 GB raw,
# stream-filtered to ~560 MB retained)
python src/fetch_opdi_events.py
# Rebuild the consolidated events_all.parquet with all 10 event types:
python -c "
import pandas as pd, glob
files = sorted(glob.glob('external/opdi/filtered/*.parquet'))
ev = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
ev['event_time'] = pd.to_datetime(ev['event_time'], utc=True, errors='coerce').astype('datetime64[us, UTC]')
ev['osn_flight_id'] = ev['osn_flight_id'].astype(str).str.strip().str.upper()
ev['osm_airport']   = ev['osm_airport'].astype(str).str.upper().str.strip()
ev = ev.dropna(subset=['event_time']).sort_values(['osm_airport','event_time']).reset_index(drop=True)
ev.to_parquet('external/opdi/events_all.parquet')
print(f'wrote {len(ev):,} events')
"
# OPDI per-flight taxi records (uses entry-taxiway fallback for pushback)
python src/build_opdi_taxi_v2.py
```

## 4. Train the models

Each script writes its files to `models/`. Run the steps in this order.
`train_lirf_regime.py` writes the LIRF feature list and encoders that the
later LIRF scripts read.

```bash
python src/train_r_all_v26.py             # base R_all_v26, seeds 42-44, linear_tree
python src/train_lirf_regime.py           # LIRF feature list + LIRF-only encoders
python src/train_r_norm_lirf_seeds.py     # R_norm_LIRF members, seeds 42-46
python src/train_lirf_regime_v23.py       # p_fb gate, fallback-rate maps, isotonic calibrator
python src/build_lirf_band_table_v30.py   # Step A band table, exclusive classes
```

Multi-threaded `linear_tree` training is not bit-deterministic. One v32 seed
needed a retry (`RECAP.md`, v32 blueprint).

## 5. Predict + upload

```bash
python src/predict_v33.py kind-mango_v33.parquet    # writes submission/kind-mango_v33.parquet

mc cp submission/kind-mango_v33.parquet osn/prc-2026-kind-mango/kind-mango_v33.parquet
```

The limit is 5 uploads per UTC day. The bucket gets a result JSON with the
score after the upload.

## Expected hold-out RMSE (2025 Jan+Jul)

3-member `R_all_v26` mean (`models/lgbm_r_all_v38.holdout.json`, `full_old` and
`clean_old`):

- FULL: **392.33 s**
- CLEAN (`30 <= y <= 7,200`): **266.46 s**

## Expected leaderboard score

v33: **301.87 s**. The gap to the hold-out comes from the 2025 to
2026 shift and the LIRF fallback rows. See `docs/MODEL_ANALYSIS.md`.
