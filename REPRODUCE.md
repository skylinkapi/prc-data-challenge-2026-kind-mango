# Reproduction guide

End-to-end from a fresh clone to a valid submission file.

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

pip install pandas pyarrow numpy scikit-learn lightgbm optuna openpyxl networkx
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
# See src/fetch_metar.py or the equivalent bootstrap; loops over ICAOs.
python src/fetch_metar.py

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

Each script trains one LGBM and saves it to `models/`. Order matters only for
Optuna (v5) since later versions reuse its best params.

```bash
python src/tune_lgbm.py           # 60-trial Optuna, ~4 h (or skip and use v5's saved params)
python src/train_lgbm_v6.py       # v5 + flt_null + BLOCK==SCHED cleaning     (~6 min)
python src/train_lgbm_v7.py       # v5 params + winter-weighted training      (~5 min)
python src/train_lgbm_v9.py       # v6 + advanced physical features           (~6 min)
python src/train_lgbm_v10.py      # v9 + OSM real path                        (~5 min)
python src/train_lgbm_v11.py      # v10 + OPDI raw event counts               (~6 min)
python src/train_lgbm_v15.py      # v11 + OPDI live-taxi                      (~7 min)
python src/train_lgbm_v16.py      # v15 + semi-supervised drift               (~6 min)
```

Ensemble weight search:

```bash
python src/ensemble_v15.py  # searches over v7/v9/v10/v11/v15
# or the extended search including v16:
# (see src/ that produces ensemble_weights_v16.txt)
```

## 5. Predict + upload

```bash
# Best solo model:
python src/predict_v15.py kind-mango_v<n>.parquet
# Or the ensemble (best result to date):
python src/predict_ensemble_v16.py kind-mango_v<n>.parquet

mc cp submission/kind-mango_v<n>.parquet osn/prc-2026-kind-mango/kind-mango_v<n>.parquet
sleep 30
mc cp osn/prc-2026-kind-mango/kind-mango_v<n>.parquet_result.json .
cat kind-mango_v<n>.parquet_result.json    # will show score
```

## Expected hold-out RMSE (2025 Jan+Jul)

Reproduces (within ±0.2 s):

- Best solo model (v15): **293.09 s**
- Best ensemble: **292.77 s**

## Expected leaderboard score

Best submission (2026-09-06): **561.04 s** (2025→2026 shift + truth-set data
quality issues account for the ~268 s local→live gap).
