# Reproduction guide

This guide rebuilds `submission/kind-mango_v92.parquet`, the current team
file (WINNING_PLAN second edition, C1). It also keeps the path to v67, the
last file before the NM clocks. Section 8 builds the file of the final phase,
`kind-mango_final.parquet` (the v132 recipe). Two tiers exist:

- **Tier 1** serves v92 from the trained artefacts in `models/`. It takes
  about 15 minutes after the feature builds. A parity test checks the
  result bit for bit.
- **Tier 2** retrains every served artefact from the organiser data and the
  open data. It takes about 20 hours on a 16-thread CPU with 32 GB RAM,
  plus about 9 hours of ADSB.lol downloads.

Run every command from the repository root. Run the long fits one at a
time unless a step says otherwise: two full-frame fits in parallel use
about 11 GB of RAM each.

## 0. Prerequisites

- Python 3.13 (the pins in `requirements.txt` come from 3.13.3 on Windows 11)
- About 20 GB of free disk for the organiser data, the open data and the
  models; one raw ADSB.lol day needs about 8 GB more while it extracts
- Optional: the MinIO client `mc` to fetch the organiser data

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

Each command writes to its folder under `external/`. The attribution and
the licence of each source are in the `README.md` data-source table.

```bash
python -m src_v3.fetch_metar              # IEM ASOS METAR, 10 CSVs in external/metar/, about 2 min

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

ADSB.lol `globe_history` days (ODbL 1.0). A day is about 4 GB and takes
about 8 minutes on an 8 MB/s link. The script keeps the points near the
10 airports in `external/adsblol/points/` and deletes the raw archive. A
stopped run resumes at the first missing day.

```bash
python -m src_v3.fetch_adsb_days --start 2026-01-01 --end 2026-01-31   # ranking days
python -m src_v3.fetch_adsb_days --start 2026-07-01 --end 2026-07-31
python -m src_v3.fetch_adsb_days --start 2025-01-15 --end 2025-01-15   # the 7 training days of v92
python -m src_v3.fetch_adsb_days --start 2025-07-01 --end 2025-07-05
python -m src_v3.fetch_adsb_days --start 2025-07-15 --end 2025-07-15
```

For a long download, `python -m src_v3.watch_adsb_fetch` checks the fetch
every hour and restarts it when the process died or made no progress; it
logs to `logs/adsb_watchdog.log` and ends when every released 2025 day is
on disk.

The v92 stack trains on exactly these 7 days of 2025. More 2025 days
change `adsb_clock_train.parquet` and the stack, so a rebuild with more
days is a new version, not v92. The v92 recipe reads a frozen copy of the
7-day table: after `build_adsb_clock` on these 7 days, copy
`external/adsblol/adsb_clock_train.parquet` to
`external/adsblol/v92/adsb_clock_train.parquet`.

## 4. Tier 1: serve v92 from the trained artefacts

The trained artefacts are not in git (`.gitignore`, 49 files and about
15 GB for the served set; linear-leaf boosters are about 850 MB each).
Copy them into `models/` from the team machine, or build them with Tier 2.
`python -m src_v3.artefact_manifest check --version 92` verifies the copy
against the tracked SHA-256 list in `models/v92.manifest.json`.

1. Build the feature files.

   ```bash
   python -m src_v2.cli build                      # v2 frames, about 5 min
   python src/build_tempo_p2575.py                 # tempo quartile columns
   python -m src_v3.build_plan_taxi_res_v57        # plan_taxi_res, 12-month route medians
   python -m src_v3.build_plan_nm_taxi_v65         # plan_nm_taxi
   python -m src_v3.build_nm_clocks_v73            # NM clocks of the scored flight
   python -m src_v3.build_nm_forward_v85           # NM clocks of the next departures
   python -m src_v3.build_opdi_own                 # OPDI pushback clocks of the scored flight
   python -m src_v3.build_adsb_clock               # ADSB.lol pushback clocks, about 20 min
   ```

2. Serve the base (the v89 recipe; this command rebuilds
   `kind-mango_v89.parquet`), then the stack on top of it.

   ```bash
   python -m src_v3.predict_v67 --weight 0.5 --ms3 --base-model lgbm_r_all_v85 \
       --catboost-tag v73 --catboost-anchor-tag v75a --anchor-model lgbm_r_all_v85a --anchor-weight 0.5 \
       --per-airport-tag v89 --per-airport-weight 0.5 \
       --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet models/nm_forward_rank_v85.parquet \
       --pre-ms kind-mango_v92_base_pre_ms.parquet --out kind-mango_v92_base.parquet
   python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound \
       --adsb-train external/adsblol/v92/adsb_clock_train.parquet \
       --base kind-mango_v92_base.parquet --out kind-mango_v92.parquet
   python -m src_v3.test_parity --version 92       # rebuilds v92 and checks the recorded hash
   ```

   **v94** (the current best) uses the same base. Its stack trains on 21
   days of 2025 (2025-01-01 to 01-15, 07-01 to 07-05, 07-15; frozen copy in
   `external/adsblol/v94/`) and blends the MLP member at 0.15:

   ```bash
   python -m src_v3.mlp_member measure              # hold-out predictions, models/v3/holdout_mlp.parquet
   python -m src_v3.mlp_member serve                # ranking predictions, models/v3/rank_mlp.parquet
   python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound --mlp-weight 0.15 \
       --adsb-train external/adsblol/v94/adsb_clock_train.parquet \
       --base kind-mango_v92_base.parquet --out kind-mango_v94.parquet
   python -m src_v3.test_parity --version 94
   python -m src_v3.artefact_manifest check --version 94
   ```

   **v95** (the current best) is the v94 command with
   `--adsb-train external/adsblol/v95/adsb_clock_train.parquet --rounds 600`:
   its stack trains on all 62 January and July 2025 days.
   `python -m src_v3.test_parity --version 95` and
   `python -m src_v3.artefact_manifest check --version 95` check it.

   **v111** (the current best, 245.23 s) serves the v104 base, then the v105 stack with a neural
   term that holds the MLP and the TabM member. Build the v104 base (v97 members, g3 LIRF gate):

   ```bash
   python -m src_v3.predict_v67 --weight 0.5 --ms3 --base-model lgbm_r_all_v97 --catboost-tag v97c \
       --catboost-anchor-tag v97ca --anchor-model lgbm_r_all_v97a --anchor-weight 0.5 \
       --per-airport-tag v97 --per-airport-weight 0.5 \
       --gate-booster lgbm_p_fb_lirf_g3.txt --gate-iso lirf_regime_g3.isotonic.pkl \
       --gate-features lirf_regime_g3.features.txt \
       --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet \
               models/nm_forward_rank_v85.parquet models/opdi_own_rank_v90.parquet \
               external/adsblol/adsb_clock_rank.parquet models/arr_art_op_link_rank_g3.parquet \
       --pre-ms kind-mango_v104_base_pre_ms.parquet --out kind-mango_v104_base.parquet
   ```

   The v105 file is the stack on that base with the v97-column MLP at 0.2 (rebuild at 1e-13 s):

   ```bash
   python -m src_v3.mlp_member serve --columns v97
   python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound --rounds 600 \
       --adsb-train external/adsblol/v97/adsb_clock_train.parquet \
       --blend-holdout models/v3/blend_v97_holdout.parquet --mlp-weight 0.2 --mlp-columns v97 \
       --base kind-mango_v104_base.parquet --out kind-mango_v105.parquet
   ```

   v111 adds the TabM member (`src_v3/tabm_member.py`). It trains on an NVIDIA GPU with the CUDA
   build of torch in `.gpu_site/` (install lines in `requirements.txt`). `measure` writes the
   hold-out member that the stack trains on; `combine` writes the neural term `nn`
   (0.318 MLP + 0.682 TabM, TabM at 0.3 on top of the v105 base):

   ```bash
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member measure --tag tabm    # hold-out member, about 15 min
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member serve --tag tabm      # two seeds on all rows, about 25 min
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member combine --tags tabm --share 0.3
   python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound --rounds 600 \
       --adsb-train external/adsblol/v97/adsb_clock_train.parquet \
       --blend-holdout models/v3/blend_v97_holdout.parquet --mlp-weight 0.44 --mlp-columns nn \
       --base kind-mango_v104_base.parquet --out kind-mango_v111.parquet
   ```

   The script does not force deterministic CUDA kernels, so a TabM refit can differ from the
   recorded one. The recorded `models/v3/rank_mlp_tabm.parquet` and
   `models/v3/holdout_mlp_tabm.parquet` rebuild v111 through `combine` and the stack.

   **v112** (244.29 s) serves the 24-epoch TabM member at 0.35 and then removes the 24-h term of
   Step A on rows without a local date crossing (`src_v3/step_a_date.py`, one LIRF row). From the
   artefacts listed in `models/v112.manifest.json` (`python -m src_v3.artefact_manifest check
   --version 112`), these commands rebuild the uploaded file at a maximum difference of 0.0 s:

   ```bash
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member combine --tags tabm_e24 --share 0.35
   python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound --rounds 600 \
       --adsb-train external/adsblol/v97/adsb_clock_train.parquet \
       --blend-holdout models/v3/blend_v97_holdout.parquet --mlp-weight 0.48 --mlp-columns nn \
       --base kind-mango_v104_base.parquet --out kind-mango_v112_stack.parquet
   python -m src_v3.step_a_date kind-mango_v112_stack.parquet kind-mango_v112.parquet
   ```

   The TabM files come from `PYTHONPATH=.gpu_site python -m src_v3.tabm_member measure --tag tabm_e24`
   (hold-out member) and `serve --tag tabm_e24` (two seeds on all rows, about 45 min).

   **v115** (243.69 s) adds X-Plane stand positions at EDDM and EHAM and ATFM slot columns. From the artefacts in
   `models/v115.manifest.json`, these commands rebuild the uploaded file at a maximum difference of 2e-13 s:

   ```bash
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member combine --tags tabm_v115 --share 0.4
   python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound --rounds 600 \
       --adsb-train external/adsblol/v115/adsb_clock_train.parquet \
       --adsb-rank external/adsblol/v115/adsb_clock_rank.parquet \
       --blend-holdout models/v3/blend_v97_holdout.parquet --mlp-weight 0.52 --mlp-columns nn \
       --base kind-mango_v104_base.parquet --out kind-mango_v115_stack.parquet
   python -m src_v3.step_a_date kind-mango_v115_stack.parquet kind-mango_v115.parquet
   ```

   The inputs come from `python -m src_v3.build_xplane_stands fetch` and `build`, `python -m src_v3.build_adsb_clock
   --stands external/xplane/stands_osm_xplane.parquet --out-dir external/adsblol/v115`, `python -m
   src_v3.build_atfm_slot`, and `PYTHONPATH=.gpu_site python -m src_v3.tabm_member measure --tag tabm_v115` and
   `serve --tag tabm_v115` (three seeds, about 60 min).

   **v118** (242.81 s) is the v116 recipe plus J1, the NM join window (`src_v3/lobt_window.py`). It reads the
   artefacts of `models/v118.manifest.json` (the same 71 files as v116) and the ranking file. The base command
   rebuilds the v104 base with `--copy-window`, which sets the LIRF gate to 0 where `|SCHED - LOBT| > 3,606` s; its
   rows outside LIRF match `kind-mango_v104_base.parquet` at a maximum difference of 0.0 s. The last command
   projects every row with `LOBT` onto `MVT - LOBT` plus or minus 3,606 s (about 30 min in total):

   ```bash
   python -m src_v3.predict_v67 --weight 0.5 --ms3 --base-model lgbm_r_all_v97 --catboost-tag v97c \
       --catboost-anchor-tag v97ca --anchor-model lgbm_r_all_v97a --anchor-weight 0.5 \
       --per-airport-tag v97 --per-airport-weight 0.5 \
       --gate-booster lgbm_p_fb_lirf_g3.txt --gate-iso lirf_regime_g3.isotonic.pkl \
       --gate-features lirf_regime_g3.features.txt --copy-window \
       --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet \
               models/nm_forward_rank_v85.parquet models/opdi_own_rank_v90.parquet \
               external/adsblol/adsb_clock_rank.parquet models/arr_art_op_link_rank_g3.parquet \
       --pre-ms kind-mango_v118_base_pre_ms.parquet --out kind-mango_v118_base.parquet
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member combine --tags tabm_v116 --share 0.4
   python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound --rounds 600 \
       --adsb-train external/adsblol/v116/adsb_clock_train.parquet \
       --adsb-rank external/adsblol/v116/adsb_clock_rank.parquet \
       --blend-holdout models/v3/blend_v97_holdout.parquet --mlp-weight 0.52 --mlp-columns nn \
       --base kind-mango_v118_base.parquet --out kind-mango_v118_stack.parquet
   python -m src_v3.step_a_date kind-mango_v118_stack.parquet kind-mango_v118_stepa.parquet
   python -m src_v3.lobt_window kind-mango_v118_stepa.parquet kind-mango_v118.parquet
   ```

   **v119** (242.51 s) adds J2 and the `tabm_plr` member (`models/v119.manifest.json`). Run the v118 base command with
   `--copy-linked` after `--copy-window` and the names `kind-mango_v119_base*.parquet`; outside the 20 J2 rows it
   matches the v118 base at 0.0 s. Then:

   ```bash
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member measure --tag tabm_plr   # hold-out member, about 30 min
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member serve --tag tabm_plr     # three seeds, about 100 min
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member combine --tags tabm_v116 tabm_plr --share 0.4
   ```

   The v118 stack command with `--base kind-mango_v119_base.parquet --out kind-mango_v119_stack.parquet`, then
   `step_a_date` and `lobt_window` as for v118, write `kind-mango_v119.parquet`. The TabM files are not bit-exact on a
   rerun (no deterministic CUDA kernels); the recorded `models/v3/*tabm_plr.parquet` files rebuild the upload.

   **v121** adds N10 seeds and P1 (twentieth pass of `docs/MODEL_ANALYSIS.md`). Two TabM members with seeds 45 to 47
   join the served pair, so each family averages six seeds. Then the v119 stack, `step_a_date` and `lobt_window` run
   on the new neural term, and `mvt_precision` raises the 19 LFPG rows whose `MVT` lies off the minute grid:

   ```bash
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member measure --tag tabm_plr_b     # hold-out member, seed 45
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member measure --tag tabm_v116_b
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member serve --tag tabm_plr_b       # three seeds, about 110 min
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member serve --tag tabm_v116_b
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member combine --tags tabm_v116 tabm_v116_b tabm_plr tabm_plr_b --share 0.4
   ```

   The v118 stack command with `--base kind-mango_v119_base.parquet --out kind-mango_v121_stack.parquet`, then
   `step_a_date` and `lobt_window` as for v118 (output `kind-mango_v121_window.parquet`), then
   `python -m src_v3.mvt_precision serve kind-mango_v121_window.parquet kind-mango_v121.parquet`.
   `python -m src_v3.mvt_precision measure` writes the hold-out price of P1 (`models/v3/mvt_precision.holdout.json`).

   **v122** serves the v121 chain with `tabm_j` as a third TabM family (the files of
   `tabm_member serve --tag tabm_j`, made for v120):

   ```bash
   PYTHONPATH=.gpu_site python -m src_v3.tabm_member combine        --tags tabm_v116 tabm_v116_b tabm_plr tabm_plr_b tabm_j tabm_j --share 0.5
   ```

   `tabm_j` appears twice so that each family takes one third. Then run the v121 stack command with
   `--mlp-weight 0.6` and the names `kind-mango_v122_*`, followed by `step_a_date`, `lobt_window` and `mvt_precision serve`.

   **v127** links the departures without an NM record to their ADS-B and OPDI tracks by flight number:

   ```bash
   python -m src_v3.build_adsb_clock --stands external/xplane/stands_osm_xplane.parquet \
       --out-dir external/adsblol/v127 --link-flight --keep external/adsblol/v116
   python -m src_v3.build_opdi_own --link-flight \
       --keep models/opdi_own_train_v90.parquet models/opdi_own_rank_v90.parquet \
       --out models/opdi_own_train_v127.parquet models/opdi_own_rank_v127.parquet
   ```

   Then the v126 chain (combine with the ten TabM tags of `submission/kind-mango_v127.recipe.json`, share 0.65)
   with the stack options `--adsb-train/--adsb-rank external/adsblol/v127/...` and
   `--opdi-train/--opdi-rank models/opdi_own_*_v127.parquet`. **v128** adds the stack options `--coverage-weights
   --leaves 15 --min-data 100 --l2 50 --rounds 400` (the 600-round default goes). **v129** refits the five TabM families
   on the K7 column sets (`tabm_member measure` then `serve` for `tabm_s_k7`, `tabm_j_k7`, `tabm_plr_j_k7`,
   `tabm_v116_k7` and `tabm_plr_k7`) and combines them with `--share 0.65`, one fifth per family.

   **v130** links the departures that no callsign links by their lift-off:

   ```bash
   python -m src_v3.build_adsb_clock --stands external/xplane/stands_osm_xplane.parquet \n       --out-dir external/adsblol/v130 --link-flight --link-takeoff --keep external/adsblol/v127 --keep-linked
   ```

   Then the v129 chain with the stack options `--adsb-train/--adsb-rank external/adsblol/v130/...`. **v131** refits the
   five TabM families on the K8 column sets (`tabm_member measure` then `serve` for `tabm_s_k8`, `tabm_j_k8`,
   `tabm_plr_j_k8`, `tabm_v116_k8` and `tabm_plr_k8`) and combines them with `--share 0.65`, one fifth per family.

   **v132** refits the three plain tree members on the v130 tables (about 6.4 h, two fits at a time). `A` is
   `external/adsblol/v130/adsb_clock_train.parquet`:

   ```bash
   python -m src_v3.catboost_base --tag v130c --rounds-from v67 --extra models/nm_clocks_train_v73.parquet models/opdi_own_train_v90.parquet A
   python -m src_v3.train_per_airport_v89 plain --tag v130 --extra models/nm_forward_train_v85.parquet models/opdi_own_train_v90.parquet A
   python -m src_v3.train_v76_base plain --tag v130 --extra models/nm_forward_train_v85.parquet models/opdi_own_train_v90.parquet A
   python -m src_v3.measure_adsb_base_v97 fit cat --tag v130 --adsb-train A              # also lgb_per_airport and lgb
   python -m src_v3.measure_adsb_base_v97 price --tag v130 --pa-guard --keep-v97 lgb_anchor cat_anchor
   ```

   The base is the v119 base command with `--base-model lgbm_r_all_v130 --catboost-tag v130c --per-airport-tag v130
   --per-airport-anchor-tag v97a --pa-guard --anchor-adsb external/adsblol/adsb_clock_rank.parquet`, with
   `external/adsblol/v130/adsb_clock_rank.parquet` as the ADS-B table in `--extra`, and the names
   `kind-mango_v132_base*.parquet`. The anchored members keep the v97 fits and read the v97 ADS-B table. Then the
   v131 chain with `--blend-holdout models/v3/blend_v130_holdout.parquet --base kind-mango_v132_base.parquet`.

   **v133** (candidate) keeps the ADS-B segments that end up to 3,600 s before `MVT` (`ADSB_FIRST_GAP_S`); only
   the stack reads the new tables:

   ```bash
   python -m src_v3.build_adsb_clock --stands external/xplane/stands_osm_xplane.parquet \n       --out-dir external/adsblol/v133 --link-flight --first-gap 3600 --keep external/adsblol/v130 --keep-segments
   cp submission/kind-mango_v132_base.parquet submission/kind-mango_v133_base.parquet
   python -m src_v3.release_final serve --stack-only --name kind-mango_v133 --stack-adsb external/adsblol/v133
   ```

   The same `serve --stack-only` command without `--stack-adsb` rebuilds v132 at 0.0 s. For the final files, add
   `--also submission/ranking_phase1.parquet` and use the name `kind-mango_final133`.

   **v134** (candidate) refits the five TabM families on the v133 tables. **v135** (candidate) refits the plain
   tree members on them too. Run one heavy fit at a time on a machine with 32 GB of RAM:

   ```bash
   for t in tabm_s_k9 tabm_j_k9 tabm_plr_j_k9 tabm_v116_k9 tabm_plr_k9; do
     python -m src_v3.tabm_member measure --tag $t && python -m src_v3.tabm_member serve --tag $t   # GPU, about 2 h each
   done
   cp submission/kind-mango_v132_base.parquet submission/kind-mango_v134_base.parquet
   python -m src_v3.release_final serve --stack-only --name kind-mango_v134 --stack-adsb external/adsblol/v133 \
       --tabm tabm_s_k9 tabm_j_k9 tabm_plr_j_k9 tabm_v116_k9 tabm_plr_k9
   ```

   For v135, run the v132 fit commands above with the tag `v133` (CatBoost tag `v133c`) and
   `external/adsblol/v133/adsb_clock_train.parquet` as the ADS-B table, then:

   ```bash
   python -m src_v3.measure_adsb_base_v97 price --tag v133 --pa-guard --keep-v97 lgb_anchor cat_anchor
   python -m src_v3.release_final serve --name kind-mango_v135 --stack-adsb external/adsblol/v133 --trees v133 \
       --tabm tabm_s_k9 tabm_j_k9 tabm_plr_j_k9 tabm_v116_k9 tabm_plr_k9
   ```

   `--trees` names the tag of the plain tree members, their ADS-B folder and the hold-out blend of the stack.
   `python -m src_v3.artefact_manifest check --version 135` checks the 80 trained artefacts of v135.

   **v137** (candidate) is v135 with two member columns in the stack. It reads the artefacts of v135:

   ```bash
   cp submission/kind-mango_v135_base.parquet submission/kind-mango_v137_base.parquet
   python -m src_v3.release_final serve --stack-only --name kind-mango_v137 --stack-adsb external/adsblol/v133 \
       --trees v133 --member-columns --tabm tabm_s_k9 tabm_j_k9 tabm_plr_j_k9 tabm_v116_k9 tabm_plr_k9
   ```

   The same command without `--member-columns` rebuilds v135 at 0.0 s.

   `python -m src_v3.test_parity --version 67` rebuilds v67 the same way.
   `src_v3/test_parity.py` holds the served commands of each recorded
   version. The hashes are in `submission/kind-mango_v<N>.parity.json`.

v92 reads these artefacts:

| artefact | producer (Tier 2) |
|---|---|
| `lgbm_r_all_v85_s{42,43,44}.txt`, `lgbm_r_all_v85a_s{42,43,44}.txt`, their `.features.txt`, `.meta.json`, `.encoders.pkl` | `src_v3/train_v76_base.py` |
| `catboost_r_all_v73.cbm`, `catboost_r_all_v75a.cbm`, their `.meta.json` | `src_v3/catboost_base.py` |
| `lgbm_pa_v89_<airport>.txt`, `lgbm_pa_v89a_<airport>.txt` | `src_v3/train_per_airport_v89.py` |
| `lgbm_r_norm_lirf_v55_s{42..46}.txt` | `src_v3/train_r_norm_v55.py` |
| `lgbm_p_fb_lirf_v56.txt`, `lirf_regime_v64.isotonic.pkl` | `src_v3/train_p_fb_v56.py`, `src_v3/train_p_fb_v64.py` |
| `lirf_regime*.pkl`, `lirf_regime*.features.txt`, `lirf_band_table_v30.json` | the `src/` scripts of Tier 2 step 2 |
| `models/v3/blend_v89_holdout.parquet` (the stack's training base) | the hold-out members of Tier 2 step 5 |
| `nm_clocks_*_v73`, `nm_forward_*_v85`, `plan_nm_taxi_rank_v65`, `opdi_own_*_v90`, `external/adsblol/adsb_clock_*` | step 1 above |

## 5. Tier 2: retrain every served artefact

The trainers read their round counts and parameters from tracked JSON
records: `tune_lgbm_v43.tuned_params.json`, `tune_lgbm_v76.tuned_params.json`,
`lgbm_r_all_v46.holdout.json`, `lirf_regime_v41.holdout.json`,
`models/v3/catboost_blend.holdout.json` and `P_FB_V23_ROUNDS` in
`src_v3/config.py`.

1. Build the frames and caches.

   ```bash
   python -m src_v2.cli build
   cd src && python tune_lgbm_v36.py build && cd ..   # H1 frame and its id side file
   ```

   `tune_lgbm_v36.py build` writes `v36_tune_cache.parquet`, its feature list
   and `v36_tune_cache.ids.parquet`. The id file keeps a null id on the 169
   rows in `models/v36_tune_cache.null_ids.json`, as the historical file did
   (MODEL_ANALYSIS R2). Then run the feature builds of Tier 1 step 1.

2. Train the older components that v92 still serves (LIRF head and tables).

   ```bash
   python src/train_r_all_v26.py             # writes lgbm_r_all_v26.encoders.pkl
   python src/train_lirf_regime.py           # LIRF feature list and encoders
   python src/eval_v33_holdout.py --rebuild-lirf   # writes models/lirf_frame_cache.parquet
   python src/train_lirf_regime_v23.py       # fallback-rate maps
   python src/train_r_norm_lirf_v41.py       # lirf_regime_v41.features.txt
   python src/build_lirf_band_table_v30.py   # Step A band table
   python -m src_v3.train_r_norm_v55         # LIRF R_norm, 5 seeds
   python -m src_v3.train_p_fb_v56           # LIRF gate
   python -m src_v3.train_p_fb_v64           # out-of-fold isotonic map for the gate
   ```

3. Train the base members of the v89 recipe.

   ```bash
   python -m src_v3.train_v76_base plain    --tag v85 --extra models/nm_forward_train_v85.parquet   # about 2.5 h
   python -m src_v3.train_v76_base anchored --tag v85 --extra models/nm_forward_train_v85.parquet   # about 2.5 h
   python -m src_v3.catboost_base --tag v73  --rounds-from v67 --extra models/nm_clocks_train_v73.parquet                           # about 3 h
   python -m src_v3.catboost_base --tag v75a --rounds-from v67 --extra models/nm_clocks_train_v73.parquet --anchor-col mvt_aobt3    # about 2.2 h
   ```

4. Train the per-airport members. The two fits can run side by side
   (6 threads each, about 45 minutes). `--seed` fits more seeds (B2,
   priced after v92).

   ```bash
   python -m src_v3.train_per_airport_v89 plain
   python -m src_v3.train_per_airport_v89 anchored
   ```

5. Build the out-of-sample hold-out members (months 1 and 7 of 2025). The
   stack learns `y - base` from their blend.

   ```bash
   python -m src_v3.measure_nm_clocks                 # writes models/v3/nm_clocks_holdout.parquet
   python -m src_v3.measure_member_v77 cat_plain      # CatBoost, about 2 h
   python -m src_v3.measure_member_v77 cat_anchor     # anchored CatBoost, about 2 h
   python -m src_v3.measure_anchor_rounds             # anchored LightGBM
   python -m src_v3.measure_extra_v81 v85 models/nm_forward_train_v85.parquet
   python -m src_v3.measure_per_airport_v89           # seed 42, the stack's per-airport half
   ```

   The first run of `python -m src_v3.stack_pushback_v90 price` writes
   `models/v3/blend_v89_holdout.parquet` from these files.

6. Serve and check as in section 4.

Linear-leaf LightGBM runs with `deterministic` and a fixed thread count
(`src_v3/config.py`). CatBoost runs with 12 threads and a fixed seed. The
stack runs with `deterministic` and 4 threads.

## 6. Known gaps

- The trained artefacts need a release bundle with hashes (MODEL_ANALYSIS
  MC3). Until it exists, Tier 1 needs Tier 2 or a copy from the team machine.
- Nobody has rerun Tier 2 end to end on a clean clone, so bit parity of a
  full retrain is not yet shown. Tier 1 parity is recorded for v67 and v92.
- `eval_v33_holdout.py --rebuild-lirf` writes the LIRF cache first, then scores the
  v33 stack, which needs boosters that Tier 2 does not build. On a clean clone the
  scoring step can fail after the cache exists; the cache is the only output v92 needs.
- The ADSB.lol archive is a third-party release. If a day disappears from
  it, the clock table of that day cannot be rebuilt.
- v90 and v91 are not reproducible bit for bit: v91 read the clock table of
  24 July days, and the table now holds 62 days.
- The recorded ADS-B tables of v127 to v132 predate the ADS-B day 2026-06-30. A rebuild differs on 116 departures
  of 2026-07-01 by at most 38.9 s (section 8).

## 7. Upload

```bash
python -m src_v2.cli upload kind-mango_v92.parquet
```

The limit is 5 uploads per UTC day. The bucket gets a result JSON with the
score within about 30 seconds.

## 8. Final phase: `kind-mango_final.parquet`

The final file is the v132 recipe on the organiser files of the final phase (670,790 departures of January,
February, June and July 2026). `src_v3/release_final.py` holds every command. No model trains in this path.

1. Copy the repository with the trained artefacts of `models/v132.manifest.json` (80 files, 14.3 GB). Run
   `python -m src_v3.artefact_manifest check --version 132`. `python -m src_v3.mlp_member serve --columns v97`
   (CPU, about 2 h, the same weights on each run) and `tabm_member serve` for the five `tabm_*_k8` tags write the
   saved fits if they are missing.
2. Get the open data for the two new months (section 3): `python -m src_v3.fetch_adsb_days --start 2026-02-01 --end
   2026-02-28` and `--start 2026-05-31 --end 2026-06-30`, the OPDI events, the METAR files and the EUROCONTROL files.
3. Put the organiser files in place:

   ```bash
   cp final_ranking.parquet    submission/ranking.parquet        # 1,341,197 movements
   cp final_submitting.parquet submission/submitting.parquet     # 670,790 departures
   cp ranking.parquet          submission/ranking_phase1.parquet # ranking file of the first phase
   ```

4. Build and serve:

   ```bash
   python -m src_v3.release_final tables                                          # 19 ranking-side tables, about 1.8 h
   python -m src_v3.release_final serve --also submission/ranking_phase1.parquet  # about 1 h, TabM on the GPU
   ```

   `tables` deletes each ranking-side table before it rebuilds it, so no table of another file can serve as a
   cache. `serve` writes `submission/kind-mango_final.parquet` and runs the format checks of the scorer.
   `--also` gives the timestamp-precision rule the second release of the January and July movements: a take-off
   time that lies off the minute grid in one release is off the grid (2 more LFPG rows).

Evidence, 2026-10-11 and 2026-10-12, in a copy of the repository:

- With the files of the ranking phase in `submission/`, `tables` rebuilds 14 of the 18 tables equal to the
  recorded ones. The four ADS-B tables differ on 114 departures of 2026-07-01, because the ADS-B day 2026-06-30
  is on disk now and the builder reads the evening before. Every training-side file is equal byte for byte.
- `serve --name kind-mango_v132r` on those tables rebuilds `kind-mango_v132.parquet` except on 116 departures of
  2026-07-01 (maximum 38.9 s, mean absolute difference 0.002 s). With the recorded tables the difference is 0.0 s.
- The saved MLP and TabM fits score the recorded feature dump with a difference of 0.0 s from the recorded
  member files.
- On the 344,841 departures that both phases share, the final file differs from v132 by 6.6 s on average
  (median 4.4 s, 99th percentile 38 s; 35 rows above 300 s, 26 of them at LIRF). The organisers drew new noise of
  up to 6 s on `MVT_TIME_UTC_mvt` and `BLOCK_TIME_UTC_mvt`, so the same flight moves by up to 12 s.
- No feature column has a higher null share in February and June than in January and July.

The stand table `external/xplane/stands_osm_xplane.parquet` stays as recorded: 39 departures of February and June
use a stand name that January and July do not have.

### Candidates of the final file

`kind-mango_final.parquet` is the v132 recipe. Four candidates add the first-gap levers on the same tables
(`tables` builds `external/adsblol/v133` as its step 12). None of the four final files is uploaded. The ranking-phase twins v135 and v137 scored 232.45 s and 232.20 s on 2026-10-13 (v132: 235.54 s).

| File | Recipe | Command after `tables` |
|---|---|---|
| `kind-mango_final133.parquet` | v133: the stack reads the v133 tables | `serve --stack-only --name kind-mango_final133 --stack-adsb external/adsblol/v133 --also ...` on a copy of `kind-mango_final_base.parquet` |
| `kind-mango_final134.parquet` | v134: v133 with the K9 TabM families | `serve --name kind-mango_final134 --stack-adsb external/adsblol/v133 --tabm tabm_*_k9 --also ...` |
| `kind-mango_final135.parquet` | v135: v134 with the plain tree members of tag `v133` | `serve --name kind-mango_final135 --stack-adsb external/adsblol/v133 --tabm tabm_*_k9 --trees v133 --also ...` |
| `kind-mango_final137.parquet` | v137: v135 with two member columns in the stack | `serve --stack-only --name kind-mango_final137 --stack-adsb external/adsblol/v133 --tabm tabm_*_k9 --trees v133 --member-columns --also ...` on a copy of `kind-mango_final135_base.parquet` |

`--tabm` takes the five names `tabm_s_k9 tabm_j_k9 tabm_plr_j_k9 tabm_v116_k9 tabm_plr_k9`; `--also` takes
`submission/ranking_phase1.parquet`. The artefacts of the last file are in `models/v135.manifest.json`.

Evidence, 2026-10-13: on the 344,841 shared departures each final-frame file differs from its ranking-phase twin
(v133, v134, v135) by 6.6 to 6.8 s on average, the size of the new timestamp noise. LIRF is equal in the four
files. The moves of `kind-mango_final135` against `kind-mango_final` correlate 0.990 with `v135 - v132`. The
first-gap rows are 27,282 (January 7,562, February 7,281, June 5,243, July 7,196).

A second run of the full `serve` command for `kind-mango_final135` (base, MLP, TabM, stack and rules from the
saved fits, 53 min) wrote a file with the same SHA-256, 7fdd4b6a....
