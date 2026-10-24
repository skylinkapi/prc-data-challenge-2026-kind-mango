# PRC Data Challenge 2026 — team `kind-mango`

Entry of a team of two. The model predicts the **taxi-out time in seconds** (`TAXITIME_SEC_mvt`) of departures at 10
European airports. The metric is the RMSE against the times that the airports report.

- **Challenge home:** https://ansperformance.eu/study/data-challenge/dc2026/
- **This repo:** https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango
- **Licence:** GNU GPLv3, see `LICENSE`.

## Status

- **Final file:** `kind-mango_final.parquet`, uploaded on 2026-10-13 at 09:54 UTC. It holds the 670,790 departures
  of January, February, June and July 2026. Its recipe is v139 (next section). The record is
  `submission/kind-mango_final139.recipe.json`.
- **Ranking phase:** the best score is **231.90 s RMSE** (`kind-mango_v139.parquet`, 2026-10-13), team rank 21 of
  246, on the 344,841 departures of January and July 2026.
- **Path:** 430.45 s (v13) to 231.90 s (v139) in 139 versions. The Results section lists the steps.
- **Paper:** `paper/main.tex` is a draft for the Journal of Open Aviation Science (JOAS). The author names, the CRediT
  roles and the score of the final phase are open; `TODO(authors)` marks each open item.

## Where to read

| File | Content |
|---|---|
| `README.md` | This summary: the model, the results, the data, the layout |
| `docs/MODEL.md` | The model description: each part, its data, its fit and the reason for each design decision |
| `paper/main.tex` | The JOAS paper draft: the data, the model, the results and the limits |
| `REPRODUCE.md` | The commands that rebuild the served files. Section 8 builds the final file |
| `RECAP.md` | One row per upload: the mechanism, the hold-out price and the live score |
| `docs/MODEL_ANALYSIS.md` | 23 audit passes: where the error sits, the findings, the closed levers |
| `docs/WINNING_PLAN.md` | The ranked levers and the final state of each lever |
| `docs/HISTORY.md` | Earlier README content: the status log, the full table of uploads, the v21 model card |
| `AGENTS.md` | The rules for the AI coding tools that worked in this repo |

## How the model works

Seven steps build the final file. `python -m src_v3.release_final` runs them (`REPRODUCE.md` section 8).
`docs/MODEL.md` describes each step in full: the columns, the parameters, the blend and the design decisions.

| Step | What it does | Code |
|---|---|---|
| 1. Tables | Builds 19 column tables from the organiser files and the open data: schedule and plan terms, the NM clocks of the scored flight (`MVT - AOBT_3`, `LOBT` terms), the clocks of the departures that take off in the next 30 min, weather, stand and taxi-path geometry, and the pushback clocks from OPDI and ADSB.lol | `release_final tables`, `src_v2/frame.py`, `src_v3/build_*.py` |
| 2. Tree members | Outside LIRF: LightGBM with linear leaves (three seeds), one LightGBM per airport and CatBoost. Each class also has an anchored member that learns the label minus the `AOBT_3` clock. The fits use the 12 months of 2025 without LIRF and without labels above 80,000 s | `src_v3/predict_v67.py` |
| 3. LIRF head | At Rome Fiumicino a part of the departures reports the schedule as the off-block, so the label equals the schedule delay. A gate (LightGBM and an isotonic map) gives the probability `p` of that regime. The value is `p * delay + (1 - p) * normal model`. A band table serves the rows with a delay above 4 h and no flight record (Step A) | `src_v3/predict_v67.py`, `src_v3/build_gate_g3.py` |
| 4. Neural term | An MLP and five TabM families (three seeds each) on the same columns. Outside LIRF the neural term has the weight 0.72 and the tree blend 0.28 | `src_v3/mlp_member.py`, `src_v3/tabm_member.py` |
| 5. Pushback stack | A LightGBM on `label - base` for the 216,792 ranking departures outside LIRF that have an own ground track. It reads the pushback clocks, the NM clock and two member gaps. It learns on the January and July 2025 hold-out, with weights that match the 2026 coverage. A lower bound follows: the time since the last ADS-B point at the assigned stand | `src_v3/stack_pushback_v90.py`, `src_v3/build_adsb_clock.py`, `src_v3/build_opdi_own.py` |
| 6. Month level | The June and July values of seven airports move by one shift per airport and month to the mean taxi-out that EUROCONTROL publishes | `src_v3/month_level.py` |
| 7. Row rules | Three rules set single rows: the Step A date fix (1 row), the window of 3,606 s around `MVT - LOBT` (about 100 rows) and the LFPG take-off times that lie off the minute grid (37 rows) | `src_v3/step_a_date.py`, `src_v3/lobt_window.py`, `src_v3/mvt_precision.py` |

Most changes were priced on the January and July 2025 hold-out before the upload (344,336 rows; `AGENTS.md`
section 11). From 2026-10-08 the live score judged some uploads without a price (`docs/MODEL_ANALYSIS.md`
twenty-first pass). `RECAP.md` gives the price, where one exists, and the live score of each upload.

### Where the error sits

The live MSE of v135 is 54,034 (232.45 s). `docs/MODEL_ANALYSIS.md` section 23.2 splits it:

| Block | Ranking rows | Share of the MSE | RMSE on the 2025 hold-out, s |
|---|---|---|---|
| Outside LIRF, clock at the assigned stand | 96,221 | 5 % | 93 |
| Outside LIRF, ADS-B ground segment | 118,756 | 18 % | 165 to 170 |
| Outside LIRF, no clock (LTFM, LFPG and others) | 101,150 | 28 % | 217 |
| LIRF (26,899 rows), and the labels above 7,200 s at the other airports | | 48 % | |

The coverage of the ground tracks sets the score. Istanbul (LTFM) has no open ground track, and Paris (LFPG) has
one on 18 % of its departures.

### What a reviewer should know

1. **The model works after the flight.** It reads the take-off time, the NM actual off-block `AOBT_3_flt`, the
   clocks of later departures and the ADS-B track up to the take-off. It is not a forecast before the off-block.
2. **`AOBT_3_flt` and `LOBT_flt`** are organiser columns. The organisers allow them (section "Stance"). The NM
   clock is not the label: `MVT - AOBT_3` alone scores 376 s RMSE on the 2025 labels.
3. **The month level reads an aggregate of the labels.** The EUROCONTROL file gives the mean taxi-out per airport
   and month, also for the scored months. The organisers confirmed its use (Discord, 2026-10-08). It moves 14
   airport months and is worth 0.30 s on the ranking board (v138, v139). `release_final serve` without
   `--month-level` builds the model without it (v137, 232.20 s).
4. **Three parts model the reports, not the taxi.** The LIRF head, Step A and the LFPG rule follow the way an
   airport fills its off-block time. They are airport-specific and do not transfer to another airport.
5. **Known gaps.** EDDF has no ground track from 2026-01-29 to 2026-02-28 in ADSB.lol and OPDI (10,367 February
   departures; `docs/MODEL_ANALYSIS.md` F23.5). The stack learns on two months of 2025 only.
6. **No row value comes from a leaderboard score.** The leaderboard judged whole recipes (`AGENTS.md` section 11,
   rule 12).

## Results

Live RMSE on the ranking file, in seconds. The table shows the uploads that carried a new mechanism;
`RECAP.md` has every upload, the rejected ones included (`docs/HISTORY.md` has the long table).

| Upload | RMSE | Mechanism |
|---|---|---|
| v13 | 430.45 | Unfiltered labels; schedule and plan deltas (`EOBT_1`, `IOBT`) |
| v21 | 330.12 | LightGBM with linear leaves; first LIRF rules |
| v33 | 301.87 | LIRF regime head: gate, normal model, band table |
| v47 | 296.57 | Fit on the 12 months of 2025 |
| v51 | 288.25 | Base fit on the served rows only (no LIRF, no label above 80,000 s) |
| v67 | 281.87 | CatBoost as a second model class |
| v73 | 270.58 | NM clocks of the scored flight |
| v76 | 265.27 | Anchored members; new parameter study |
| v89 | 264.34 | Clocks of later departures; one LightGBM per airport |
| v92 | 253.82 | Pushback clocks from OPDI and ADSB.lol in a stack; stand bound |
| v97 | 247.89 | Every 2025 ADS-B day in the stack and in each base member |
| v105 | 246.48 | LIRF gate with arrival columns of the airline; MLP member on the new columns |
| v116 | 243.65 | TabM member; stand positions from X-Plane where OSM has none |
| v121 | 242.03 | `LOBT` window rules; take-off times off the minute grid at LFPG |
| v126 | 241.66 | Five TabM families |
| v129 | 239.72 | Ground tracks for the departures without an NM record (flight-number link) |
| v132 | 235.54 | Lift-off link for the departures that no callsign links; members refit |
| v135 | 232.45 | ADS-B segments that end before the take-off |
| v137 | 232.20 | Two member gaps as stack columns |
| v139 | 231.90 | Month level from the EUROCONTROL monthly means |

The pattern over the season: a new measurement of the off-block paid each time (v73, v92, v129, v132, v135);
new weights, new parameters and each change of the LIRF head after v105 did not
(section "Approaches tried and RULED OUT").

## Setup and run

```bash
pip install -r requirements.txt                 # Python 3.13, pinned versions
python -m src_v3.fetch_external                 # the open data except the ADS-B days, about 1 h
python -m src_v3.fetch_external adsb            # the ADS-B days, about 4 GB a day (see below)
python -m src_v3.release_final tables           # the 19 ranking-side tables, about 1.5 h
python -m src_v3.release_final serve --name kind-mango_final139 --stack-adsb external/adsblol/v133 \
    --trees v133 --member-columns --month-level kept \
    --tabm tabm_s_k9 tabm_j_k9 tabm_plr_j_k9 tabm_v116_k9 tabm_plr_k9
```

`fetch_external` gets each open source of the data-source table. It skips a file that is on disk, so a
stopped run resumes. `python -m src_v3.fetch_external --list` shows each step and its state.

The two `release_final` commands need three things on disk. None of them is in git:

- the organiser files in `training/` and `submission/` (`REPRODUCE.md` section 2);
- the open data in `external/` (`python -m src_v3.fetch_external`; `REPRODUCE.md` section 3). The final recipe
  reads 471 ADSB.lol days. A day is a download of about 4 GB and takes 6 to 8 min; about 15 MB of it stay on
  disk. The 471 days take about 2 days;
- the 80 trained artefacts of `models/v135.manifest.json`, 14 GB. `REPRODUCE.md` gives the fit command of each
  artefact; the full retrain takes about 20 h on a 16-thread CPU plus about 10 h on a GPU for the TabM families.
  This repo has no download bundle of the artefacts.

| Stage | Command |
|---|---|
| Fetch the organiser data | `REPRODUCE.md` section 2 (needs an OSN account) |
| Fetch the open data | `python -m src_v3.fetch_external`, then `python -m src_v3.fetch_external adsb` |
| Fit the members | `REPRODUCE.md` Tier 2 and the v130 to v135 paragraphs |
| Build the tables and serve | `python -m src_v3.release_final tables`, then `serve` |
| Price a change on the hold-out | `python -m src_v3.stack_pushback_v90 price ...`, `python -m src_v3.month_level measure FILE` |
| Check a file before an upload | `python src/check_submission_2026.py FILE --ref BEST` (label-free gates) |
| Check the rebuild of an old file | `python -m src_v3.test_parity --version 92` (bit for bit) |
| Build the paper | `python paper/make_figures.py`, then `latexmk -pdf -cd paper/main.tex` (TeX Live with biber) |

## Stance on the NM clocks, later departures, trajectory data and the monthly means

From v73 on, the base reads the NM clocks of the scored flight. Up to v72,
no served file read `AOBT_3_flt` or `LOBT_flt` (v39 read `LOBT_flt`; see
WINNING_PLAN 3.1). The team lifted that rule on 2026-09-26 and accepts the
risk of a later organiser ruling.

Facts that support the change:

1. The data page blanks only `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`
   on ranking DEP rows. `AOBT_3_flt` and `LOBT_flt` are listed flight
   columns with no restriction.
2. Organiser rulings on Discord: "any provided value is fair to use as a
   proxy (or as itself) if it helps" (2026-09-08). For off-block times,
   "there are no such restrictions" (2026-09-18, MX1).
3. `AOBT_3_flt` does not reconstruct the label. On 2025 departures,
   `MVT - AOBT_3` has a median absolute error of 175 s against the label,
   21 % of rows lie within 60 s, and the raw clock scores RMSE 376 s. On
   the 584 rows with a label above 7,200 s, the clock median is 1,082 s:
   it does not see long holds.
4. The quoted warning in earlier versions of this section came from the
   team's own brief (`docs/PRC_Data_Challenge_2026_BRIEF.md`), not from an
   organiser page.

Forward windows (changed 2026-09-28): from v85 the base reads the NM
clocks of departures that take off after the row (WINNING_PLAN L8). The
team stance MF6 kept such columns out. The organiser allows features from
other movements ("what is occupied, what is coming", 2026-09-09).

Pushback clocks from open trajectory data (changed 2026-09-29): from v90
the served file reads the pushback time of the scored flight from OPDI
flight events, and from v91 also from ADSB.lol traces. The organiser
allowed this on 2026-09-18: "if you can find open trajectory data with good
ground coverage and extract off-block times, we are ok" (`RECAP.md`
Discord confirmations). Raw OSN state vectors and Trino stay out.
The clocks are not the label: after a bias, 57 to 86 % of the linked
rows lie within 60 s, and a stack model learns when to trust them.

Monthly means (changed 2026-10-13): from v138 the served file reads the mean taxi-out per airport and month
that EUROCONTROL publishes (`taxi_out_additional_time`). The file is an aggregate of the labels of the scored
months. A team asked the organisers on 2026-10-08 if the step is still correct for the final phase; the answer
was "no change. still ok". The step is one shift per airport and month, in June and July only, and
`src_v3/month_level.py` states the rule that selects the airports. `RECAP.md` rows v138 and v139 give the
evidence. The daily EUROCONTROL delay file, which is also label-derived, is not used.

Risk: on 2026-09-11 the organiser said that reverse engineering of the
ranking is a practice "we despise" and kept the right to add a phase 2. A
phase 2 without NM actuals cancels this gain. No submission sets a row
value from a leaderboard score.

## Data sources (all open)

`python -m src_v3.fetch_external` downloads each source of this table into `external/` (`REPRODUCE.md`
section 3). The organiser data need an OSN account (`REPRODUCE.md` section 2).

| Source | Data | Licence |
|---|---|---|
| PRC / OpenSky Network | Movements at 10 EU airports (2025 train, 2026 ranking) | Challenge data licence |
| Iowa State ASOS | METAR reports, routine and special, per station; `python -m src_v3.fetch_metar` | Public domain |
| OurAirports | Runway coordinates, headings, lengths | Public domain |
| OSM Overpass | `aeroway=parking_position`, `taxiway`, `runway` | ODbL |
| Eurocontrol NM | Daily ATC pre-dep delay, ATFM slot adherence, airport traffic | Open (non-commercial-use clause; challenge use confirmed by organiser, 2026-09-17) |
| OPDI (PRC + OSN) | Flight events (entry-runway, entry-taxiway, exit-parking, …) | Open (CC-BY 4.0, confirmed by organiser) |
| VRS StandingData | Aircraft-type metadata | Open |
| X-Plane Scenery Gateway | `apt.dat` of EDDM (scenery 111124), EHAM (111943), LEMD (99851), LEBL (102255) and LFPG (111022): stand positions (row 1300) where OSM has no position for the assigned stand; `python -m src_v3.build_xplane_stands fetch` and `build` (v115 on). The `apt.dat` files are kept unchanged with their copyright header in `external/xplane/`. | GPL v2 or later |
| EUROCONTROL `atfm_slot_adherence` (ansperformance.eu CSV) | Departures, regulated departures and departures before, inside and after the slot window, per airport and UTC day, 2025 and 2026; `python -m src_v3.build_atfm_slot` (v115 on). Take-off times only, no off-block time. (c) EUROCONTROL. | Open, with attribution; same family as the daily files confirmed by the organiser, 2026-09-17 |
| EUROCONTROL `taxi_out_additional_time` (ansperformance.eu CSV) | Reference and additional taxi-out time per airport and month, 2025 and 2026; `python -m src_v3.month_level` (v138, v139) downloads the two files. The sum per departure is the monthly mean taxi-out, an aggregate of the labels. (c) EUROCONTROL. | Open, with attribution; the organisers confirmed the use for the scored months on Discord, 2026-10-08 |
| ADSB.lol `globe_history` | Daily ADS-B traces; `src_v3/fetch_adsb_days.py` keeps the low points near the 10 airports and deletes the raw archive. Used for the pushback clock of the scored flight (v91 on). Data (c) ADSB.lol contributors. | ODbL 1.0. The derived tables stay out of git (share-alike). |

## Layout

| Directory | Content |
|---|---|
| `src_v3/` | The current pipeline: table builders, members, stack, rules, `release_final.py`, price and audit scripts |
| `src_v2/` | The frame builder of the base columns (`python -m src_v2.cli build`); the v39 rewrite |
| `src/` | The first pipeline (v1 to v47): feature modules and LIRF fits that the served chain still calls |
| `models/` | Small records of the fits: JSON reports, feature lists, manifests. The trained artefacts are not in git |
| `submission/` | Per upload: the score (`.result.json`), the gate report (`.check.json`) and the recipe (`.recipe.json`). The parquet files are not in git |
| `docs/` | The model description, the analysis, the plan, the history and the challenge brief |
| `paper/` | The JOAS paper: `main.tex`, `reference.bib`, the figures, the figure data and the JOAS class |
| `training/`, `external/`, `logs/` | Organiser data, open data and run logs; not in git |

The block below lists the files of each directory.

```
src/
  # v33 pipeline (first generation)
  features_weather.py           # METAR + crosswind + Step 7 temperature block
  features_congestion_v2.py     # arrivals keyed on ADES_mvt, same-runway ARR/DEP counts
  features_congestion.py        # v1 keyed on ADEP_mvt (needed by v21_old booster)
  features_eurocontrol.py       # daily ATFM per-airport features
  features_operator.py          # operator historic taxi encoders
  features_taxi_distance.py     # OSM stand->runway haversine
  features_advanced.py          # physical proxies (queue, runway diversity, ades delay)
  features_osm_path.py          # OSM taxiway graph shortest-path
  features_opdi.py              # OPDI rolling event counts
  features_opdi_live.py         # OPDI rolling live-taxi mean/median (600 s lag fix)
  features_turnaround.py        # aircraft-on-stand link, uses BLOCK_TIME_UTC_mvt
  features_disruption.py        # 60-min disruption meters
  train_r_all_v26.py            # base regressor, 3 members
  train_lirf_regime.py          # LIRF feature list and LIRF-only encoders
  train_r_norm_lirf_seeds.py    # 5 R_norm_LIRF members
  train_lirf_regime_v23.py      # LIRF p_fb gate, fallback-rate maps, calibrator
  build_lirf_band_table_v30.py  # Step A band table, exclusive classes
  predict_v30.py                # shared prediction pipeline
  predict_v33.py                # ranking submission (current best)
  price_ensemble.py             # label-free ambiguity price on the ranking set
  eval_v33_holdout.py           # v33 stack scored end to end on the 2025 hold-out (P1)
  train_r_all_v40.py            # tempo columns on the v33 base, paired against a control
  predict_v40.py                # v33 stack with the v40 base members
  train_r_norm_lirf_v41.py      # LIRF regressor members with the tempo columns, paired
  train_p_fb_lirf_v41.py        # LIRF gate with the tempo columns, paired (rejected)
  features_order_next.py        # forward take-off order column (v42, set aside)
  train_r_all_v42.py            # forward order on the base, paired (set aside)
  predict_v41.py                # current best

  # fourteenth-pass harness (H1 to H5)
  build_h1_frame_cache.py       # cached frame with the movement id inside (h1_frame_cache.parquet)
  eval_v33_holdout.py           # scores any stack per class per airport on months 1 and 7
  check_submission_2026.py      # label-free 2026 checks against the best file
  check_coverage.py             # per-column coverage 2025 vs Jan/Jul 2026 + OPDI counts
  test_v33_parity.py            # rebuilds v33 to 0.0 s; writes the 2026 feature dump

  # rejected ships v34-v38, kept for the paper trail (see RECAP)
  build_lirf_band_table_v34.py, train_lirf_regime_v34.py,   # v34/v35 fit-month refit
  tune_lgbm_v36.py,                                          # purified tuner
  train_p_fb_lirf_seeds.py, price_p_fb_seeds.py,             # v37 p_fb seed mean
  features_plan.py,                                          # v38 ARVT_1 planned-time features
  predict_v34.py .. predict_v38.py

  # older versions kept for the paper trail
  train_lgbm_v{1..20}.py, train_r_all_v20.py, features_*_v2.py,
  ensemble_*.py, predict_v*.py, mixture_*.py

  # diagnostics used by MODEL_ANALYSIS.md
  eval_full.py, eval_v21_stepA.py, eval_v21_stepAC.py, eval_v20_final.py,
  eval_v19.py, eval_v20_combined.py, train_drift_meter.py,
  train_fallback_detector.py, train_v18_detectors.py

src_v3/                         # current pipeline (v48 on), constants in config.py
  predict_v67.py                # base outside LIRF and the LIRF head; release_v97.py builds the best file
  fetch_external.py             # one command for the open data: OurAirports, METAR, EUROCONTROL, OSM, VRS, X-Plane, OPDI and the ADS-B days
  release_final.py              # final phase: `tables` rebuilds the ranking-side tables, `serve` writes the final file (v132 recipe by default; the flags of v139 in REPRODUCE section 8)
  build_gate_g3.py              # column table of the LIRF gate g3: G1 shares, v83 arrival link, operator shares
  stack_pushback_v90.py         # OPDI and ADS-B pushback stack and the A5 stand bound
  build_adsb_clock.py, build_opdi_own.py   # pushback clocks of the scored flight
  measure_budget_v97.py         # v97 error budget per class, airport and ADS-B coverage; 2026 projection
  measure_lirf_gap.py           # LIRF gate ceiling and the arrival-artefact signal (WINNING_PLAN third edition)
  measure_nm_join.py            # J1, J2: hold-out price of the LOBT window rules on the v115 path (nineteenth pass)
  lobt_window.py                # J1 to J3 serve steps: LIRF copy tests for the gate, J3 anchor, LOBT window last (v118 on)
  mvt_precision.py              # P1: LFPG departures with MVT off the minute grid, raised to MVT - LOBT minus 242 s (v121)
  month_level.py                # June and July values of seven airports (six in v138) shifted to the EUROCONTROL monthly mean (v138, v139)
  build_lobt_terms.py           # unclipped MVT - LOBT and SCHED - LOBT for the TabM member tabm_j (v120)
  build_arr_artefact.py         # G1: airline arrival artefact shares for the LIRF gate (v101 on)
  train_p_fb_v83.py             # LIRF gate refit with added columns, `--tag g3 --extra ...` (v104, v105)
  catboost_gate.py              # N7: CatBoost member of the LIRF gate, `price` and `fit --tag g3c` (v109)
  blend_nnls.py                 # N3: NNLS member weights per coverage class, `price` and `serve` (v110)
  tabm_member.py                # TabM member on the GPU, `measure`, `serve` and `combine` (v111)
  step_a_date.py                # Step A without the 24-h term where the local dates do not cross
  measure_pseudo_n5.py, measure_member_swap.py, probe_adsb_feeds.py, build_route_clock.py   # closed levers N5, K3, K1

external/                       # open-data caches (gitignored)
  metar/       Iowa State ASOS CSVs per ICAO
  eurocontrol/ 5 NM performance Excels + consolidated daily parquet
  osm/         OSM stand positions + taxiway graphs
  airports/    OurAirports runways.csv + airports.csv
  vrs/         VRS StandingData aircraft types
  opdi/        OPDI event chunks + consolidated events_all.parquet + taxi_out_v2.parquet
training/                       # organiser-provided 2025 monthly parquets (gitignored)
submission/                     # ranking.parquet, submitting.parquet, kind-mango_v*.parquet
models/                         # trained boosters + encoders + JSON config
.gpu_site/                      # CUDA build of torch and the TabM packages for tabm_member.py (gitignored)
docs/                           # MODEL.md (the model description), MODEL_ANALYSIS.md (23 audit passes), WINNING_PLAN.md (levers and their final state), HISTORY.md (earlier README text), the challenge brief, the audit prompt
paper/                          # JOAS paper draft; the JOAS template asks for these file names
  main.tex, reference.bib       # the manuscript and its references
  make_figures.py               # draws figures/*.pdf from data/*.csv
  data/                         # live score of each upload; ADS-B coverage per airport (MODEL_ANALYSIS 22.1)
  extra/                        # joas.cls (LPPL 1.3c) and joas-logo.pdf, unchanged from github.com/open-aviation/joas-template
RECAP.md                        # session log for every attempt, submissions and scores
requirements.txt                # pinned library versions
```

## Approaches tried and RULED OUT

For transparency, and so the next replicator does not waste days on the same
dead-ends:

- **Last probes on the v135 path, 2026-10-13** (`docs/MODEL_ANALYSIS.md` twenty-third pass). OPDI taxiway events as
  clocks of the departures without a clock: +5 MSE on the 2025 hold-out. An ADS-B lookback of 7,200 s: -1,053
  against -1,081 MSE at 3,600 s. Neural weight, TabM share and stack size: -20 to +118 MSE. A second stack on the
  rows without a clock: +187 MSE. The month level in winter: +528 MSE, because the EUROCONTROL file drops the long
  departures of the de-icing months.
- **ADS-B specialist as a stack column (v96)** — -302 MSE on the 2025 hold-out,
  +0.53 s live. Derived columns did not transfer this season (v66, v71, v83, v96);
  new row-level observables did (v73, v90 to v95).
- **Lower bound from the first ADS-B ground point (WINNING_PLAN A3)** — the rule
  `max(pred, MVT - first moving point + k)` costs +4,875 to +7,954 MSE on the
  2025 hold-out for every `k`. The first point often moves before the reported
  off-block (41 % of EHAM rows), and where the stack predicts below the bound
  the label also sits below it on 69 % of rows. The stack already reads the
  first-point columns. The stand bound of A5 replaces it.
- **Neighbour pushback clocks (A4)** — the mean and count of other departures'
  OPDI and ADS-B pushback taxi over the v85 windows: +2 MSE as stack columns,
  +62 MSE when rows with neighbours but no own clock join the stack
  (`python -m src_v3.stack_pushback_v90 price --neighbours cols|rows`).
- **Three seeds for the per-airport members (B2)** — -5.2 MSE inside the v89-style
  blend on the 2025 hold-out, airports split both ways
  (`python -m src_v3.measure_per_airport_v89 --seed 43`, then `--seed 44`).
- **Local hour, airport-scoped stand and runway, drift flags (MF3, MD5, MD6)** — as
  columns of a residual stack on all hold-out rows, cross-fitted January against
  July: +723 to +1,267 MSE against +535 for the same stack without them.
- **Step A analogue outside LIRF (B1)** — outside LIRF, 0 of 412 rows of 2025
  with a null flight record and `sd > 14,400` sit within 60 s of `sd`
  (1 of 4,313 above 7,200 s). A rule would move 278 rows of 2026 with no support.
- **Fallback heads outside LIRF (MH2)** — mixing a calibrated P(|y - sd| < 60) toward `sd`
  costs +292 MSE on the 2025 hold-out and is worse at all 9 airports
  (`python -m src_v3.measure_mh2`).
- **Tail rules that pull predicted outliers toward the 2025 label tail** (Discord
  suggestion, 2026-09-23) — +1,639 MSE on the 2025 hold-out for a 50/50 blend
  with the 2025 curve of y on `mvt_eobt1`; a cap on normal rows moves 3 rows
  (`python -m src_v3.measure_tail_rules`).
- **Huber loss and log-target regression** — 408 s hold-out, much worse than MSE.
- **Tighter target range `[30, 21600]`** — hurt RMSE 6+ s.
- **LIRF specialist model** — converged in the same range as the general model.
- **Old tail-mixture (P(y > 3600) classifier + hard swap)** — v22/v27; live went
  from 372 to 625. Hard swaps have unbounded downside.
- **`P_tail` as a regressor feature** — target leakage (in-fold) or distribution
  shift (out-of-fold).
- **Data-quality classifier (BLOCK==SCHED detector)** — AUC 0.51 on the wrong
  target. The right target is `|y - sd| <= 1` per airport, deployed at LIRF only.
- **Semi-supervised drift features from ranking rows** — hurt both hold-out and
  live.
- **Quantile regression (q = 0.5 / 0.55 / 0.60) ensemble** — near-zero effect.
- **Extended OPDI event counts (exit-taxiway, entry-threshold)** — the counts
  collapse between 2025 and 2026 at 5 airports (see MODEL_ANALYSIS 7); use
  the raw counts at EDDF and LSZH only, with the OPDI-live 600 s lag fix.
- **VRS aircraft-type derived features (manufacturer, engines)** — overfit,
  duplicates the aircraft-type categorical.
- **Post-hoc airport-level shift calibration** — near-zero effect.
- **12-month refit + 3-seed seed/ff ensemble (v25)** — NNLS collapsed to one
  member, live +1.5 s inside noise.
- **Runway-configuration string categorical (v26)** — hundreds of levels,
  overfit, +7.6 s full hold-out.
- **Step A extended to `sd > 3,600`** — v21 already holds the conditional mean
  in that band; the extended rule cost 9.8 s on the hold-out.
- **Per-airport fallback detectors at EGLL, LEBL, LTFM (v18/v19 mix)** — with
  the corrected fallback definition (`|y - sd| <= 1` at 9 airports and `< 60`
  only at LIRF) these detectors model punctuality, not reporting failure. The
  R_all regressor already holds that signal.
- **Hourly / stand-occupancy / EOBT-copy detector families** — measured and
  worth 48 to 229 MSE at stake per airport, not worth a component.
- **7-seed base mean (v32)** — priced -1,204 MSE; live +0.13 s. The base
  expansion drove the regression.
- **Fit-month refit of the LIRF head statistics (v34, v35)** — live +0.21 and
  +0.22 s vs v33.
- **Zero-clip repair on the base mean (v36)** — live +0.18 s vs v33.
- **5-seed `p_fb` mean (v37)** — priced -151 MSE; live +0.65 s vs v33.
- **`R_norm_LIRF` retrain without the columns that lose 2026 coverage** —
  priced at 103 MSE, under the 268 MSE lottery; no upload.
- **`ARVT_1_flt` planned-time features in the base, raw form (v38)** — seed 43
  stops at iteration 6; 3-member hold-out CLEAN +10.8 s worse; no upload.
- **Runway configuration or 3-lane submission variants that differ on a handful
  of rows** — the brief forbids learning from the ranking process.

## Reproduction

See [`REPRODUCE.md`](REPRODUCE.md). Section 8 builds the final file with two commands
(`python -m src_v3.release_final tables`, then `serve` with the flags of v139) from the artefacts of
`models/v135.manifest.json`. A second run of the full `serve` command rebuilt `kind-mango_final135.parquet`
with the same SHA-256. Tier 1 serves v92 from the trained artefacts; Tier 2 retrains every served artefact, and
step 3 of the guide fetches the open data, ADSB.lol days included. `python -m src_v3.test_parity --version 92`
rebuilds v92 and checks it against the recorded hash in `submission/kind-mango_v92.parity.json`;
`--version 67` does the same for v67.

## License

**GNU GPLv3.** See `LICENSE`. Prize-eligible per the challenge rules: source
public, open-data only, documented reproduction path.

## Acknowledgements

- Eurocontrol Performance Review Commission (PRC), OpenSky Network — challenge
  data and infrastructure.
- Iowa State University Environmental Mesonet — METAR archive.
- David Megginson — OurAirports open data.
- OpenStreetMap contributors — stand and runway geometry.
- OPDI project (PRC + OSN) — ADS-B derived flight events.
- Virtual Radar Server community — StandingData aircraft-type lookup.
- The idea to use `MVT - AOBT_3_flt` as a taxi-out clock is public in the
  GPLv3 repos of other teams (for example `Phoenix-Ops-LTD/prc2026-taxiout`
  and `ChandanHegde07/OpenAir`). v73 reimplements the idea in this
  pipeline and copies no code.
