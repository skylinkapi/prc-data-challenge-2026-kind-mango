# Reproduction guide

This guide rebuilds `submission/kind-mango_v89.parquet`, the team best (live
264.34 s, 2026-09-28), and `submission/kind-mango_v67.parquet` (live 281.87 s,
2026-09-24). v89 keeps the LIRF head of v67 and replaces the base outside LIRF
(section 6). Sections 1 to 5 build the shared parts and v67. Two tiers exist:

- **Tier 1** serves a file from the trained artefacts in `models/`. It takes
  about 6 minutes for v67. A parity test checks the result bit for bit.
- **Tier 2** retrains every served artefact from the organiser data and the
  open data. It takes about 8 hours for v67 and about 10 more hours for the
  v89 base, on a 16-thread CPU with 32 GB RAM.

Run every command from the repository root. Run the long steps one at a
time: two full-frame fits in parallel exhaust 32 GB of RAM.

## 0. Prerequisites

- Python 3.13 (the pins in `requirements.txt` come from 3.13.3 on Windows 11)
- About 20 GB of free disk for the organiser data, the open data and the models
- Optional: the MinIO client `mc` to upload a submission

## 1. Environment

```bash
git clone https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango.git
cd prc-data-challenge-2026-kind-mango
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Organiser data

Get an OSN account. Generate an S3 access key and secret key in the MinIO
console. Use port 443, not 9443.

```bash
mc alias set osn https://s3.opensky-network.org <ACCESS_KEY> <SECRET_KEY>
mc mirror osn/prc-2026-datasets/ training/            # 12 monthly parquets, about 285 MB
mc cp osn/prc-2026-datasets/ranking.parquet  submission/
mc cp osn/prc-2026-datasets/submitting.parquet submission/
```

## 3. Open data

Each command writes to its folder under `external/`. The served feature
functions read all of them.

```bash
# METAR: Iowa State ASOS archive, one CSV per ICAO in external/metar/ (about 32 MB).
# No committed fetch script exists yet (MODEL_ANALYSIS MC6). Download the CSVs by hand.

# OurAirports airport and runway tables (about 15 MB)
mkdir -p external/airports
curl -o external/airports/runways.csv  https://davidmegginson.github.io/ourairports-data/runways.csv
curl -o external/airports/airports.csv https://davidmegginson.github.io/ourairports-data/airports.csv

python src/build_eurocontrol_daily.py     # EUROCONTROL daily features, about 5 min
python src/build_osm_stands.py            # OSM stands via Overpass, about 2 min
python src/build_osm_taxi_paths.py        # OSM taxiway graph and shortest paths, about 5 min
python src/fetch_opdi_events.py           # OPDI ADS-B events, about 50 min
python -c "
import pandas as pd, glob
files = sorted(glob.glob('external/opdi/filtered/*.parquet'))
ev = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
ev['event_time'] = pd.to_datetime(ev['event_time'], utc=True, errors='coerce').astype('datetime64[us, UTC]')
ev['osn_flight_id'] = ev['osn_flight_id'].astype(str).str.strip().str.upper()
ev['osm_airport'] = ev['osm_airport'].astype(str).str.upper().str.strip()
ev = ev.dropna(subset=['event_time']).sort_values(['osm_airport', 'event_time']).reset_index(drop=True)
ev.to_parquet('external/opdi/events_all.parquet')
"
python src/build_opdi_taxi_v2.py          # OPDI per-flight taxi records
```

## 4. Tier 1: serve v67 from the trained artefacts

The trained artefacts are not in git (`.gitignore`, 936 MB for the served set). Copy them
into `models/` from the release bundle, or build them with Tier 2.

```bash
python -m src_v2.cli build                           # v2 frames, about 5 min
python src/build_tempo_p2575.py                      # tempo quartile columns
python -m src_v3.build_plan_taxi_res_v57             # plan_taxi_res, 12-month route medians
python -m src_v3.build_plan_nm_taxi_v65              # plan_nm_taxi
python -m src_v3.predict_v67 --weight 0.5 --out kind-mango_v67.parquet
python -m src_v3.test_parity --tag v67              # rebuilds v67 and checks the recorded hash
```

The parity test compares the SHA-256 of the ids and the predictions with
`submission/kind-mango_v67.parity.json`. On 2026-09-26 the rebuild matched
bit for bit.

v67 reads these artefacts (audit of every file the serving run opens):

| artefact | producer (Tier 2) |
|---|---|
| `lgbm_r_all_v65_s{42,43,44}.txt`, `lgbm_r_all_v65.features.txt`, `lgbm_r_all_v65.encoders.pkl` | `src_v3/train_v65_base.py` (encoders copied from v26) |
| `catboost_r_all_v67.cbm`, `catboost_r_all_v67.meta.json` | `src_v3/catboost_base.py --tag v67` |
| `lgbm_r_norm_lirf_v55_s{42..46}.txt` | `src_v3/train_r_norm_v55.py` |
| `lgbm_p_fb_lirf_v56.txt` | `src_v3/train_p_fb_v56.py` |
| `lirf_regime_v64.isotonic.pkl` | `src_v3/train_p_fb_v64.py` |
| `lirf_regime.encoders.pkl`, `lirf_regime.features.txt` | `src/train_lirf_regime.py` |
| `lirf_regime_v23.rate_maps.pkl`, `lirf_regime_v23.features.txt` | `src/train_lirf_regime_v23.py` |
| `lirf_regime_v41.features.txt` | `src/train_r_norm_lirf_v41.py` |
| `lirf_band_table_v30.json` (tracked) | `src/build_lirf_band_table_v30.py` |
| `v2/cache/frame_rank.parquet`, `tempo_p2575_rank.parquet`, `plan_taxi_res_rank_v57.parquet`, `plan_nm_taxi_rank_v65.parquet` | section 4 commands |

## 5. Tier 2: retrain every served artefact

The trainers read their round counts and parameters from tracked JSON
records, so no step needs an older booster file:
`tune_lgbm_v43.tuned_params.json`, `lgbm_r_all_v46.holdout.json`,
`lirf_regime_v41.holdout.json`, `models/v3/catboost_blend.holdout.json` and
`P_FB_V23_ROUNDS` in `src_v3/config.py`.

1. Build the frames and caches.

   ```bash
   python -m src_v2.cli build
   cd src && python tune_lgbm_v36.py build && cd ..   # H1 frame and its id side file
   python src/build_tempo_p2575.py
   python -m src_v3.build_plan_taxi_res_v57
   python -m src_v3.build_plan_nm_taxi_v65
   ```

   `tune_lgbm_v36.py build` writes `v36_tune_cache.parquet`, its feature list
   and `v36_tune_cache.ids.parquet`. The id file keeps a null id on the 169
   rows in `models/v36_tune_cache.null_ids.json`, as the historical file did
   (MODEL_ANALYSIS R2). Every base since v40 trained with that file.

2. Train the older components that v67 still serves.

   ```bash
   python src/train_r_all_v26.py             # writes lgbm_r_all_v26.encoders.pkl
   python src/train_lirf_regime.py           # LIRF feature list and encoders
   python src/eval_v33_holdout.py --rebuild-lirf   # writes models/lirf_frame_cache.parquet
   python src/train_lirf_regime_v23.py       # fallback-rate maps
   python src/train_r_norm_lirf_v41.py       # lirf_regime_v41.features.txt
   python src/build_lirf_band_table_v30.py   # Step A band table
   ```

3. Train the served components.

   ```bash
   python -m src_v3.train_v65_base           # LightGBM base, 3 seeds, about 35 min
   python -m src_v3.train_r_norm_v55         # LIRF R_norm, 5 seeds
   python -m src_v3.train_p_fb_v56           # LIRF gate
   python -m src_v3.train_p_fb_v64           # out-of-fold isotonic map for the gate
   python -m src_v3.catboost_base --tag v67  # CatBoost base, 7,385 rounds, about 2.6 h
   ```

4. Serve and check as in section 4.

Linear-leaf LightGBM runs with `deterministic` and 4 threads
(`src_v3/config.py`). CatBoost runs with 12 threads and a fixed seed.

## 6. v89: the base outside LIRF on the NM clocks

v89 serves the LIRF head of section 4 unchanged. Outside LIRF the base is
`0.5 x LightGBM term + 0.5 x CatBoost term`. Each term is a 50/50 mix of a
plain member and a member on the residual from the `AOBT_3` anchor. Rows
without `AOBT_3` get the plain member only. Each LightGBM member is a 50/50
mix of the 3-seed global model and one model per airport.

### 6.1 Tier 1: serve v89

Run the section 4 frame builds first (`src_v2.cli build`, the tempo, plan and
`plan_nm_taxi` caches). Then:

```bash
python -m src_v3.build_nm_clocks_v73                 # clocks of the scored flight, train and rank
python -m src_v3.build_nm_forward_v85                # forward windows, train and rank (8 served columns)
python -m src_v3.predict_v67 --weight 0.5 --ms3 --base-model lgbm_r_all_v85 \
    --catboost-tag v73 --catboost-anchor-tag v75a --anchor-model lgbm_r_all_v85a --anchor-weight 0.5 \
    --per-airport-tag v89 --per-airport-weight 0.5 \
    --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet models/nm_forward_rank_v85.parquet \
    --pre-ms kind-mango_v89_pre_ms.parquet --out kind-mango_v89.parquet
python -m src_v3.test_parity --tag v89               # rebuilds v89 and checks the recorded hash
```

The v89 hash is not recorded yet. On the machine that holds the uploaded
file, run `python -m src_v3.test_parity --tag v89 --record` once, then
commit `submission/kind-mango_v89.parity.json`.

v89 reads the LIRF artefacts of section 4 and these files:

| artefact | producer (Tier 2) |
|---|---|
| `lgbm_r_all_v85_s{42,43,44}.txt`, `lgbm_r_all_v85.features.txt` (tracked), `lgbm_r_all_v85.encoders.pkl` | `python -m src_v3.train_v76_base plain --tag v85 --extra models/nm_forward_train_v85.parquet` |
| `lgbm_r_all_v85a_s{42,43,44}.txt`, `lgbm_r_all_v85a.features.txt` (tracked), `lgbm_r_all_v85a.encoders.pkl` | `python -m src_v3.train_v76_base anchored --tag v85 --extra models/nm_forward_train_v85.parquet` |
| `lgbm_pa_v89_<ICAO>.txt` (9 airports), `lgbm_pa_v89.features.txt` (tracked) | `python -m src_v3.train_per_airport_v89 plain` |
| `lgbm_pa_v89a_<ICAO>.txt` (9 airports), `lgbm_pa_v89a.features.txt` (tracked) | `python -m src_v3.train_per_airport_v89 anchored` |
| `catboost_r_all_v73.cbm`, `catboost_r_all_v73.meta.json` (tracked) | `python -m src_v3.catboost_base --tag v73 --rounds-from v67 --extra models/nm_clocks_train_v73.parquet` |
| `catboost_r_all_v75a.cbm`, `catboost_r_all_v75a.meta.json` (tracked) | `python -m src_v3.catboost_base --tag v75a --rounds-from v67 --extra models/nm_clocks_train_v73.parquet --anchor-col mvt_aobt3` |
| `nm_clocks_rank_v73.parquet`, `nm_forward_rank_v85.parquet` | section 6.1 commands |

### 6.2 Tier 2: retrain the v89 base

Run section 5 steps 1 and 2 and the three LIRF commands of step 3
(`train_r_norm_v55`, `train_p_fb_v56`, `train_p_fb_v64`). v89 does not
serve `lgbm_r_all_v65` or `catboost_r_all_v67`. Then run these steps one at a
time:

```bash
python -m src_v3.build_nm_clocks_v73
python -m src_v3.build_nm_forward_v85
python -m src_v3.train_v76_base plain --tag v85 --extra models/nm_forward_train_v85.parquet      # 3 seeds, 3,377 rounds, about 2.3 h
python -m src_v3.train_v76_base anchored --tag v85 --extra models/nm_forward_train_v85.parquet   # 3 seeds, 3,340 rounds, about 2.3 h
python -m src_v3.train_per_airport_v89 plain                                                     # 9 members, about 45 min
python -m src_v3.train_per_airport_v89 anchored                                                  # 9 members, about 45 min
python -m src_v3.catboost_base --tag v73 --rounds-from v67 --extra models/nm_clocks_train_v73.parquet                          # 7,385 rounds, about 2.6 h
python -m src_v3.catboost_base --tag v75a --rounds-from v67 --extra models/nm_clocks_train_v73.parquet --anchor-col mvt_aobt3  # 7,305 rounds, about 2.2 h
```

The LightGBM trainers read their parameters and round count from the tracked
study record `models/tune_lgbm_v76.tuned_params.json`, so the 3-hour study
(`python -m src_v3.tune_lgbm_v76`) is optional. The CatBoost trainers read the
round count from the tracked v67 hold-out report. The two per-airport fits
are the exception to the one-fit rule: they use 6 threads each and ran side
by side on 2026-09-28.

## 7. Known gaps

- The trained artefacts need a release bundle with hashes (MODEL_ANALYSIS
  MC3). Until it exists, Tier 1 needs Tier 2 or a copy from the team machine.
- No committed v89 parity record exists yet (section 6.1). Nobody has rerun
  the v89 Tier 2 on a clean clone.
- Up to 2026-09-30, `build_nm_forward_v85.py` wrote the v86 variant (the 8
  forward columns plus 4 queue columns) to `nm_forward_*_v86.parquet`, not the
  v85 files that v85 and v89 read. The default now writes the v85 files.
- No committed script fetches the METAR CSVs (MC6).
- `eval_v33_holdout.py --rebuild-lirf` writes the LIRF cache first, then scores the
  v33 stack, which needs boosters that Tier 2 does not build. On a clean clone the
  scoring step can fail after the cache exists; the cache is the only output v67 needs.
- `train_r_all_v26.py`, `train_lirf_regime.py`, `train_lirf_regime_v23.py` and
  `train_r_norm_lirf_v41.py` also train models that v67 no longer serves.
  Nobody has rerun Tier 2 end to end on a clean clone, so bit parity of a
  full retrain is not yet shown.

## 8. Upload

```bash
python -m src_v2.cli upload kind-mango_v67.parquet
```

The limit is 5 uploads per UTC day. The bucket gets a result JSON with the
score within about 30 seconds.
