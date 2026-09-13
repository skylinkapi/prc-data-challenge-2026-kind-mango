# PRC Data Challenge 2026 — team `kind-mango`

Solo entry. Predicts **taxi-out time in seconds** (`TAXITIME_SEC_mvt`) for
departing flights at 10 major European airports. Metric is RMSE against
airport-reported truth.

- **Challenge home:** https://ansperformance.eu/study/data-challenge/dc2026/
- **This repo:** https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango
- **Current leaderboard best:** **301.87 s RMSE** (`kind-mango_v33.parquet`),
  rank 44 of 111 teams. v33 landed at 301.87 vs the priced 301.86 (off by
  0.01 s), validating the 2026 ambiguity framework from the seventh audit.
- **After v33 (status 2026-09-13):** v34 to v37 scored 302.05 to 302.52 and
  are rejected. v38 failed its hold-out gate. v39 (twelfth-audit rewrite
  under `src_v2/`, sections P–F in order) scored 352.19 s live from a cold
  start; it improved CLEAN by 2 s on hold-out but the 60-column base
  regressed FULL because it dropped 37 v33 columns the audit did not
  account for. v33 stays best. See `docs/MODEL_ANALYSIS.md` section 4.1.

## License

**GNU GPLv3.** See `LICENSE`. Prize-eligible per the challenge rules: source
public, open-data only, documented reproduction path.

## Ethics — features we deliberately did NOT use

The challenge brief forbids "exploiting the ranking process". We stay off two
inputs that would place us in the top 3 but that we read as excluded:

1. **`AOBT_3_flt` on ranking DEP rows** — present on 98.9 % of ranking rows;
   `MVT_TIME - AOBT_3_flt` reconstructs the hidden off-block time to within
   the reporting-offset noise. The brief warns "using them is probably
   exploiting the ranking process".
2. **`LOBT_flt`** — equals `AOBT_3_flt` on 4.3 % of rows and is otherwise a
   planned time, not an actual off-block. Excluded for the same reason.

Both fields are visible in the training and ranking files. Our submissions never
read them. The exclusion sets the floor near 300 s on the leaderboard, per the
analysis in `docs/MODEL_ANALYSIS.md`.

## What the model is

The scoring stack (v33, current best) is documented in `docs/MODEL_ANALYSIS.md`.
In one line:

**Three-seed LightGBM regressor with `linear_tree` (`R_all_v26`), plus a LIRF
regime head (5-member `R_norm_LIRF` mean + calibrated `p_fb` mixture), plus
the Step A LIRF band lookup, plus the ITY340 offset rule.**

Per-airport delegation. `src/predict_v33.py` calls `src/predict_v30.py:main`
with the 5 `R_norm_LIRF` members:

| airport | scoring component |
|---|---|
| LIRF, `sd > 14,400`, null flight | Step A band table (`p_fb * sd + p_24h * (86,400 + mean_24h_extra) + p_norm * mixture`), exclusive classes since v30 |
| LIRF, `sd > 70,000`, outside Step A | ITY340 constant `(5/6) * (86,400 + 1,150) + (1/6) * sd` |
| LIRF, all other rows | `mixture = p_fb_cal * sd + (1 - p_fb_cal) * min(R_norm_LIRF, 4,431)` |
| every other airport | 3-seed mean of `lgbm_r_all_v26_s{42,43,44}` |

Both LIRF-head boosters read the LIRF-only encoders in
`models/lirf_regime.encoders.pkl`. Serving them the all-airport encoders was
the v30 defect fix. The v33 build swaps `R_norm_LIRF` for a 5-member mean
(seeds 42-46). The price was 74 MSE of variance reduction. The live score
confirmed it. `mean_24h_extra` is a per-band value in
`models/lirf_band_table_v30.json`; the fallback is 1,150 s.

## Live leaderboard progression

Only `kind-mango_v*.parquet` uploads are shown. All are scored on the same
344,841-row 2026 Jan+Jul ranking set. RMSE decreases are the improvement.

| upload | live RMSE | change | headline |
|---|---|---|---|
| kind-mango_v1 to v10 | 560-562 | — | Old metric era; label filter hid the tail |
| kind-mango_v13 | 430.45 | -130 | Unfiltered labels + unclipped `sd` + `EOBT_1_flt`, `IOBT_flt` deltas |
| kind-mango_v16 | 372.40 | -58 | + LIRF band-lookup rule for `sd > 14,400` null-flight rows |
| kind-mango_v18 | 370.63 | -1.8 | + Section 6.3 LIRF null-flight `3,600 < sd <= 14,400` classifier |
| kind-mango_v19 | 359.46 | -11.2 | + per-airport R_norm switch on the wins |
| kind-mango_v20 | 345.70 | -13.8 | + turnaround family + disruption meters + ITY340 rule |
| kind-mango_v21 | 330.12 | -15.6 | + linear_tree + signed-log copies + OPDI-live leak fix + temperature |
| kind-mango_v22 | 320.31 | -9.8 | + LIRF regime head (`R_norm_LIRF` + calibrated `p_fb`) + refined Step A band table |
| kind-mango_v23 | 317.50 | -2.8 | + fallback-rate encodings on `p_fb` (7 keys × 2 = 14 features) |
| kind-mango_v24 | 316.85 | -0.65 | + 3-seed base with honest 12 % random stop split |
| kind-mango_v26 | 303.71 | -13.1 | + drop 38 `ec_*` and 18 `opdi_*` features (0 % 2026 coverage) |
| kind-mango_v29 | 303.26 | -0.45 | + `R_norm_LIRF` clip at 4,431 s + ITY340 constant formula |
| kind-mango_v30 | 301.98 | -1.28 | + LIRF-only encoders for LIRF models + Step A band table with exclusive classes |
| kind-mango_v32 | 302.11 | +0.13 | + 7-seed base + 5-member `R_norm_LIRF` — base expansion regressed (see MODEL_ANALYSIS 10) |
| **kind-mango_v33** | **301.87** | **-0.11** | + 5-member `R_norm_LIRF` only, 3-seed base kept; priced 301.86, live 301.87 |
| kind-mango_v34 to v36 | 302.05-302.09 | +0.18 to +0.22 | hold-out-leak refits and the zero-clip repair; rejected (see RECAP) |
| kind-mango_v37 | 302.52 | +0.65 | + 5-seed `p_fb` mean; priced 301.62, regressed; rejected (see MODEL_ANALYSIS 4.1) |
| kind-mango_v39 | 352.19 | +50.32 | twelfth-audit rewrite from a cold start, `src_v2/`; hold-out CLEAN 264 (beat v33's 266) but FULL regressed on live; base lost 37 v33 columns; rejected (see MODEL_ANALYSIS 4.1) |

v38 (base + `ARVT_1_flt` planned-time features) is not in the table. It failed
the deployed-recipe gate, so no upload followed (MODEL_ANALYSIS section 4).

The `RECAP.md` file tracks every attempt, including the failures. The
`docs/MODEL_ANALYSIS.md` file carries the twelfth-pass audit: the MSE budget
by label class, the data and method findings, and the ordered measures for
the next model. Earlier passes stay in git.

## Model card (v21)

This card describes the v21 base. The shipped base `R_all_v26` differs in 3
ways:

- 3 seeds (42-44), early-stopped on a random 12 % of the train rows.
- 97 features. v26 dropped the 38 `ec_*` and 18 `opdi_*` columns.
- Hold-out on Jan+Jul 2025 for the 3-member mean: FULL 392.33, CLEAN 266.46.

The LIRF head still reads 153 columns (MODEL_ANALYSIS F1).

- **Algorithm:** LightGBM regressor, `linear_tree=True`, `linear_lambda=1.0`.
- **Objective:** RMSE.
- **Rows kept:** 2,084,659 departures at the 10 target airports for 2025 (only
  `y > 0`).
- **Split:** train = months 2 to 6 and 8 to 10; early-stop = 11 and 12;
  hold-out = 1 and 7 (Jan+Jul).
- **Features:** 153 total.
  - 10 categoricals: airport, destination, runway, stand, aircraft type, wake,
    market segment, operator, flight rule, flight type.
  - 14 time and schedule: `hour`, `dow`, `month`, `sched_delay`,
    `sd_mod_86400`, `mvt_eobt1`, `mvt_iobt`, `eobt1_sched`, `eobt1_iobt`, the
    signed-log copies of the 5 previous fields, `flt_null`, `flt_id_null`.
  - 18 METAR: crosswind and headwind on the active runway, gust, visibility,
    ceiling, cloud codes, precip / snow / thunder / freezing, and the Step 7
    temperature block (`tmpc`, `dwpc`, `dewpt_spread`, `ice_accretion_1hr`,
    `deicing_gate = wx_precip * (tmpc < 3)`).
  - 13 congestion v2 (ADES-keyed arrival counts, same-runway ARR/DEP counts,
    arrival taxi-in rolling mean, next-10-min queue).
  - 6 turnaround (`ground_time`, `slack_eobt`, `slack_sched`, `in_delay`,
    `taxi_in_prev`, `link_ambiguous`) linking each departure to the last
    arrival on the same stand, using `BLOCK_TIME_UTC_mvt` for the join.
  - 4 disruption meters over a 60-minute window before take-off
    (`dep_sd_mean_60m`, `dep_fnull_60m`, `arr_txi_mean_60m`, `arr_txi_p90_60m`).
  - 18 Eurocontrol daily ATFM per airport.
  - 16 operator historic encoders per (op × airport / runway / hour-bin / stand).
  - Physical: OSM haversine stand→runway, real path length, turn count.
  - 5 advanced physical (scheduled pushback load, runway diversity, secs since
    last DEP on runway, runway-bank intensity, ADES arrival ATFM delay).
  - 9 OPDI raw event counts (entry-runway, entry-taxiway, exit-parking, all in
    prev 15/30/60 min) — leak-fixed by a 600 s lag on the window end.
  - 9 OPDI live-taxi rolling stats (mean / median / count of ACTUAL observed
    taxi times of OTHER completed flights, 30/60/120 min).
- **Hyperparameters (Optuna trial 31, kept, plus `linear_tree`):**
  - `learning_rate` 0.0228, `num_leaves` 220 (halved from 440 for linear
    leaves), `min_data_in_leaf` 76, `feature_fraction` 0.674, `bagging_fraction`
    0.937, `lambda_l1` 0.089, `lambda_l2` 0.342, `min_gain_to_split` 1.83.
  - Best iteration on Nov+Dec early-stop: 372.

- **Hold-out RMSE on Jan+Jul 2025** (single number is misleading — see
  `docs/MODEL_ANALYSIS.md` section 2 on the 1.5 s measurement noise floor):

  | metric | value |
  |---|---|
  | FULL RMSE (all rows) | 425.92 |
  | CLEAN RMSE (`30 <= y <= 7,200`) | 274.08 |
  | per-airport clean (LIRF / EGLL / LFPG / LEMD) | 546.6 / 278.4 / 275.5 / 188.5 |

## Tail rules of v21 (historical)

The v33 rules are in the delegation table above. v22 replaced the rules below
with the LIRF regime head and retired the Section 6.3 classifier. The text
stays for the record. See `src/predict_v21_final.py`.

1. **Step A band table.** LIRF, null flight record, `sd > 14,400`. Prediction:
   `P(fb) * sd + P(24h) * (86,400 + 1,150) + (1 - P(fb) - P(24h)) * v21_base`.
   Bands and probabilities are in the appendix of `docs/MODEL_ANALYSIS.md`.
   Covers 43 rows on the ranking set.
2. **Section 6.3 classifier.** LIRF, null flight record, `3,600 < sd <= 14,400`.
   A LightGBM binary classifier for `|y - sd| < 60` trained on the LIRF-null
   subset with flight-number prefix, stand prefix, aircraft, runway, hour and
   `sd`. Applied as `p * sd + (1 - p) * 1,220` (mean of normal LIRF-null rows).
   The raw probability is shrunk by 0.6 (a conservative de-calibration).
   Covers 305 rows.
3. **ITY340 rule.** LIRF, any flight record, `sd > 70,000`, outside the Step A
   cell. Prediction: `(5/6) * (86,400 + R_all_v21) + (1/6) * R_all_v21`.
   Fires on 1 row per year at LIRF. This row alone is worth about 18 s of
   full RMSE in expectation.

## Data sources (all open)

| Source | Data | Licence |
|---|---|---|
| PRC / OpenSky Network | Movements at 10 EU airports (2025 train, 2026 ranking) | Challenge data licence |
| Iowa State ASOS | METAR reports, hourly, per station | Public domain |
| OurAirports | Runway coordinates, headings, lengths | Public domain |
| OSM Overpass | `aeroway=parking_position`, `taxiway`, `runway` | ODbL |
| Eurocontrol NM | Daily ATC pre-dep delay, ATFM slot adherence, airport traffic | Open |
| OPDI (PRC + OSN) | Flight events (entry-runway, entry-taxiway, exit-parking, …) | Open (CC-BY 4.0, confirmed by organiser) |
| VRS StandingData | Aircraft-type metadata | Open |

## Quick start

Python 3.13, one virtualenv.

```bash
pip install pandas pyarrow numpy scikit-learn lightgbm optuna openpyxl networkx minio

# 1. Fetch open external data (~60 min, several rate-limited APIs)
python src/build_eurocontrol_daily.py     # daily ATFM per airport
python src/build_osm_stands.py            # OSM stand centroids
python src/build_osm_taxi_paths.py        # OSM taxiway graph
python src/fetch_opdi_events.py           # OPDI ADS-B events (~50 min, 560 MB)
python src/build_opdi_taxi_v2.py          # OPDI taxi-out record aggregation
# METAR: no committed fetch script; download Iowa State ASOS CSVs to external/metar/

# 2. Train the base regressor (3 members, linear_tree)
python src/train_r_all_v26.py

# 3. Train the LIRF regime head, then build the Step A band table
python src/train_lirf_regime.py           # LIRF feature list and LIRF-only encoders
python src/train_r_norm_lirf_seeds.py     # 5 R_norm_LIRF members, seeds 42-46
python src/train_lirf_regime_v23.py       # p_fb gate, fallback-rate maps, calibrator
python src/build_lirf_band_table_v30.py   # band table with exclusive classes

# 4. Predict on the ranking set
python src/predict_v33.py kind-mango_v33.parquet

# 5. Submit
python -c "\
from minio import Minio; \
c = {l.split('=')[0].strip(): l.split('=',1)[1].strip() for l in open('.osn_credentials.txt') if '=' in l}; \
Minio('s3.opensky-network.org', access_key=c['access_key'], secret_key=c['secret_key'], secure=True) \
    .fput_object(c['bucket'], 'kind-mango_v33.parquet', 'submission/kind-mango_v33.parquet')"
```

Full reproduction steps are in [`REPRODUCE.md`](REPRODUCE.md).

### Rewrite (`src_v2/`, v39, live 352.19)

Cold-start rebuild of the section-4 measures from `docs/MODEL_ANALYSIS.md`.
One evaluation harness (P1), one data layer (A1–A8), neighbour EOBT tempo
(B1), take-off order (B2), stand re-occupation gap (B3), queue between
EOBT_1 and MVT (B4), plan-taxi residual (B5), anchor deltas (B6); 3-seed
constant-leaf base on clean rows (C2, C3, C5); regime heads at seven
airports (C1); null-flight LIRF tail head (D1); post-processing (E1–E3).
Ships as `kind-mango_v39.parquet`. Regressed live by +50 s vs v33 because
the base carries 60 columns against v33's 97. Kept in the repo as the
anchor for the "layer measures on the v33 stack" path.

```bash
python -m src_v2.cli build      # cache 2 feature frames (~4 min)
python -m src_v2.tests          # F4 label/coverage assertions
python -m src_v2.cli full kind-mango_v39.parquet   # fit + hold-out + predict
python -m src_v2.cli upload kind-mango_v39.parquet # MinIO upload
```

## Layout

```
src/
  # v33 pipeline (current)
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
docs/                           # PRC brief + MODEL_ANALYSIS.md (twelfth-pass audit, measures for the next model)
RECAP.md                        # session log for every attempt, submissions and scores
```

## Approaches tried and RULED OUT

For transparency, and so the next replicator does not waste days on the same
dead-ends:

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

See [`REPRODUCE.md`](REPRODUCE.md).

## Acknowledgements

- Eurocontrol Performance Review Commission (PRC), OpenSky Network — challenge
  data and infrastructure.
- Iowa State University Environmental Mesonet — METAR archive.
- David Megginson — OurAirports open data.
- OpenStreetMap contributors — stand and runway geometry.
- OPDI project (PRC + OSN) — ADS-B derived flight events.
- Virtual Radar Server community — StandingData aircraft-type lookup.
