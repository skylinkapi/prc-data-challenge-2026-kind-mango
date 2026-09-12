# Model analysis — eleventh pass

Status on 2026-09-12: live best **301.87 s (v33)**, rank 44 of 111
(`README.md:9-10`). v34, v35 and v36 scored 302.07, 302.09 and 302.05 and
are rejected (`RECAP.md:31-33`). v37 scored 302.52 and is rejected (section 4.1).

This pass replaces every earlier audit. The old passes stay in git:

- ninth pass: `git show 88923a9:docs/MODEL_ANALYSIS.md`
- tenth pass: `git show 41345e0:docs/MODEL_ANALYSIS.md`

Citations of the form "10th F7" point to finding IDs of the tenth pass.
Findings in this file restart at F1. The analysis reads code, docs and model
files only. It trains no model and runs no pipeline.

## 0. Units and evidence rules

| quantity | value | source |
|---|---|---|
| v33 live MSE | 301.87² = 91,125 | `RECAP.md:34` |
| 1 s of live RMSE at this level | about 604 MSE (2 × 302) | arithmetic |
| member-draw lottery on live | 268 MSE, about 0.44 s | `88923a9:151,332` |
| hold-out evidence bar | 2 s of CLEAN RMSE (`30 <= y <= 7,200`) | `train_r_all_v26.py:138`; user rule |
| label-free ensemble price | `A * (k - m) / (m * (k - 1))` | `88923a9:305` |

A number without a citation is marked **unpriced** or **unmeasured**. A live or
hold-out gap under the bar is not evidence of a cause.

Guardrails: no `AOBT_3_flt`, no `LOBT_flt`, no row value derived from the
leaderboard. One change per upload. Stacked changes regressed in v32, v34 and
v35 (`RECAP.md:32-33,36`).

## 1. Component map

The v33 submission runs `src/predict_v30.py:main` with 5 `R_norm_LIRF` files
(`src/predict_v33.py:12-16`). All other arguments keep their v30 defaults
(`src/predict_v30.py:53-68`).

| # | component | role | evidence (live) | defects | priced ceiling |
|---|---|---|---|---|---|
| 1 | Training data and label filter | Build 2025 DEP rows with `y > 0` | v13 unfiltered labels: -130 s (`RECAP.md:52,98`) | F12 | about 0; gate at 9 airports 0.35 s (10th F1) |
| 2 | Feature families | Supply 97 base and 153 LIRF columns | v26 dropped `ec_*`/`opdi_*`: -13.1 s (`RECAP.md:39`) | F1, F2, F3, F16 | F1 unpriced; analogy 636 MSE |
| 3 | Base regressor | Predict every non-LIRF row | v24 3 seeds: -0.65 s; v26: -13.1 s (`RECAP.md:39,43`) | F4, F9, F14 | 7-member price 1,204 MSE, refuted live |
| 4 | LIRF regime head | Predict LIRF rows as a fallback mixture | v22 -9.8, v23 -2.8, v29 -0.45, v30 -1.28, v33 -0.11 s (`RECAP.md:34-45`) | F1, F5, F6, F13 | oracle gap 13,700 MSE (`88923a9:374`) |
| 5 | Step A band table | Set LIRF null-flight rows with `sd > 14,400` | v16 -58 s (`RECAP.md:50,120`) | F7 | 43 ranking rows; smoothing priced +95 MSE worse (`88923a9:287`) |
| 6 | ITY340 rule | Set LIRF rows with `sd > 70,000` outside Step A | inside v20 -13.8 s and v29 -0.45 s bundles | F8 | 1 row; unpriced alone |
| 7 | Post-processing | Clip, average, fill NaN | v36 paired zero fix: +0.18 s (`RECAP.md:31`) | F9, F10, F15 | F9 at most 67 MSE |
| 8 | Submission and evaluation | Write template rows; price changes | v33 landed 0.01 s from price (`RECAP.md:34`) | F14 | framework holds for `R_norm`; failed for base (v32) |

### 1.1 Training data and label filters

- Purpose: build one row per 2025 departure at the 10 target airports.
- Inputs: 12 monthly training parquets. Output: `dep` frame and context frame.
- Code: filter to `ADEP_mvt` or `ADES_mvt` in targets (`train_r_all_v26.py:40`).
  DEP rows at targets (`:44`). Label filter `y > 0` (`:53`). Hold-out months
  {1, 7} (`:25,78`). Random 12 % stop mask from `default_rng(1234)` (`:84-87`).
- LIRF head: train months {2-6, 8-10}, stop months {11, 12}
  (`train_lirf_regime.py:35-37,100-102`). Genuine rows for `R_norm_LIRF`:
  `|y - sd| >= 60` and `y < 80,000` (`train_r_norm_lirf_seeds.py:41-42`).
- Evidence: the unfiltered label fix is the largest live gain, -130 s
  (`RECAP.md:17-25`).
- Defects: F12. The `y > 0` filter drops 0.023 % of rows (10th F5).
- Price: closed. No filter change has a priced gain.

### 1.2 Feature families

- Purpose: describe schedule, flight plan, weather, load, geometry and history.
- Inputs: movement rows and open data. Output: numeric and categorical columns.
- Base list: 97 columns (`models/lgbm_r_all_v26.features.txt:1-97`), built by
  `train_r_all_v26.py:98-102`.
- LIRF list: 153 columns (`models/lirf_regime.features.txt:1-153`), built by
  `train_lirf_regime.py:111-115`. `p_fb` adds 14 fallback-rate columns
  (`train_lirf_regime_v23.py:41-46,172-177`).

| family | module | base | LIRF head | note |
|---|---|---|---|---|
| time, schedule, OBT deltas, signed-log copies | `train_lgbm_v21.py:67-77`, `train_r_all_v21.py:37-46` | yes | yes | `mvt_eobt1` rank 1 in v21 (`RECAP.md:194`) |
| METAR and temperature | `features_weather.py:95-139` | yes | yes | `tmpc` rank 26 (`RECAP.md:174`) |
| congestion v2 | `features_congestion_v2.py:50-121` | yes | yes | NaN taxi-in read as 0 s (`:86`) |
| Eurocontrol daily `ec_*` (38) | `features_eurocontrol.py:21-43` | **no** | **yes** | 0 % July 2026 coverage (`RECAP.md:151`) |
| operator encoders `openc_*` (16) | `features_operator.py:15-66` | yes | yes, LIRF-only fit | `MIN_COUNT = 20` (`:23`) |
| stand-runway geometry | `features_taxi_distance.py:72-95` | yes | yes | |
| advanced physical (5) | `features_advanced.py:23-122` | yes | yes | `ades_arr_atfm_delay_today` reads the `ec` table (`:111-118`) |
| OSM taxi path (3) | `features_osm_path.py:32-60` | yes | yes | |
| OPDI event counts (9) | `features_opdi.py:39-98` | **no** | **yes** | coverage collapse at 5 airports (`RECAP.md:153`) |
| OPDI live taxi (9) | `features_opdi_live.py:47-86` | **no** | **yes** | 600 s lag fix (`:66-68`) |
| turnaround (6) | `features_turnaround.py:27-99` | yes | yes | `slack_sched` rank 13 (`RECAP.md:178`) |
| disruption (4) | `features_disruption.py:46-96` | yes | yes | NaN schedule read as 0 s (`:74`) |
| fallback rates (14) | `train_lirf_regime_v23.py:52-96` | no | `p_fb` only | `fbrate_flt_prefix` rank 1 (`RECAP.md:159`) |

Modules not read by v33: `features_congestion.py` (its output is overwritten,
see F15), `features_opdi_climate.py`, `features_opdi_extended.py`,
`features_opdi_live_v2.py`, `features_opdi_v2.py`, `features_openap.py`,
`features_runway_config.py`, `features_ssl.py`, `features_vrs.py`.

- Evidence: v26 removed `ec_*` and `opdi_*` from the base for -13.1 s live.
  v25 imputed `ec_*` with 2025 medians and lost 58 s (`RECAP.md:149-153`).
- Defects: F1, F2, F3, F16.
- Price: F1 unpriced. A crude analogy gives 636 MSE (see F1).

### 1.3 Base regressor

- Purpose: predict taxi-out for every row outside LIRF.
- Inputs: 97 columns, all-airport encoders (`predict_v30.py:107-109`).
  Output: `r_all`, the mean of 3 boosters (`predict_v30.py:136-145`).
- Recipe: `BEST_PARAMS` (`train_lgbm_v21.py:52-61`), `linear_tree=True`,
  `linear_lambda=1.0`, `num_leaves=220`, seeds 42-44, early stop 100 on the
  random stop set (`train_r_all_v26.py:27,114-124`).
- Evidence: v24 3-seed honest stop -0.65 s; v26 -13.1 s (`RECAP.md:39,43`).
  v27, v28 and v32 changed the base and regressed by +0.32, +0.77 and +0.13 s
  (`RECAP.md:36,40-41`).
- Defects: F4 (tuner), F9 (per-member clip), F14 (pricing).
- Price: the 7-member mean priced 1,204 MSE (5,417 × 4 / 18). Live v32 moved
  +0.13 s (`RECAP.md:36`). The price did not hold.

### 1.4 LIRF regime head

- Purpose: model the LIRF fallback class (`y ≈ sd`) apart from genuine taxis.
- Inputs: 153 columns with LIRF-only encoders (`predict_v30.py:125-130`); 14
  rate columns from all-month maps (`:121-123`).
- Outputs, per LIRF row:
  1. `R_norm_LIRF`: mean of 5 clipped members, then `min(mean, 4,431)`
     (`predict_v30.py:152-158`).
  2. `p_fb_cal`: v23 classifier through an isotonic map (`:163-169`).
  3. Mixture `p_fb_cal * sd + (1 - p_fb_cal) * R_norm_LIRF`; `R_norm_LIRF`
     alone when `sd` is NaN (`:176-182`).
- Training: `R_norm` members on genuine rows, `num_leaves=127`
  (`train_r_norm_lirf_seeds.py:41-68`). `p_fb` binary on `|y - sd| < 60`
  (`train_lirf_regime_v23.py:119,182-194`). Isotonic fit on the stop months
  (`:197-198`).
- Evidence: v22 -9.8 s, v23 -2.8 s, v29 -0.45 s, v30 -1.28 s, v33 -0.11 s
  (`RECAP.md:34-45`). v34 and v35 refit `p_fb` and lost 0.21 and 0.22 s
  (`RECAP.md:32-33`), inside the lottery.
- Defects: F1, F5, F6, F13.
- Price: v33 5-member mean 74 MSE, landed (`RECAP.md:34`). Oracle gap 13,700
  MSE (`88923a9:374`), no lever priced against it.

### 1.5 Step A band table

- Purpose: set LIRF rows with no flight record and `sd > 14,400` by class
  frequency.
- Inputs: `sd`, `ADEP_mvt`, `FLIGHT_ID_mvt`, the LIRF mixture. Output: new
  value for the cell rows.
- Build: 11 hand bands, gate 14,400 s, classes fb / 24 h / normal made
  exclusive (`build_lirf_band_table_v30.py:17-20,45-48`). Fit on all 12 months
  (`:31-38`). Counts 1 to 27 per band (`models/lirf_band_table_v30.json:51-131`).
- Apply: `p_fb * sd + p_24h * (86,400 + mean_24h_extra) + p_norm * p_final`
  (`predict_v23.py:76-94`). `p_final` holds the LIRF mixture at that point
  (`predict_v30.py:182,186`).
- Evidence: v16 -58 s (`RECAP.md:120`); v30 exclusive classes inside -1.28 s.
- Defects: F7.
- Price: smoothing priced +95 MSE worse (`88923a9:287`). Other changes unpriced.

### 1.6 ITY340 rule

- Purpose: hedge one LIRF row class with a flight record and `sd > 70,000`.
- Inputs: `sd`, `cell_A`. Output: `(5/6) * (86,400 + 1,150) + (1/6) * sd`
  (`predict_v30.py:35-36,190-194`).
- Evidence: bundled with the turnaround and disruption families in v20 and
  with the `R_norm` clip in v29 (`RECAP.md:38,47`). No isolated live number.
- Defects: F8.
- Price: unpriced. The row is worth about 18 s of full RMSE in expectation
  (`README.md:150-151`).

### 1.7 Post-processing

- Purpose: turn member outputs into one finite, non-negative value per
  template row.
- Steps in `predict_v30.py`:
  1. Clip each base member at 0 (`:143`), then average (`:145`).
  2. Clip each `R_norm` member at 0 (`:155`), then average (`:157`).
  3. Clip the final vector at 0 (`:196`).
  4. Left-merge on the template; fill a missing row with the median of all
     predictions (`:208-214`).
- Evidence: v36 removed step 1's per-member clip and filled zero rows per
  airport. Live +0.18 s (`RECAP.md:31`).
- Defects: F9, F10, F15.
- Price: F9 at most 67 MSE (see F9). F10 unmeasured.

### 1.8 Submission and evaluation protocol

- Submission: 344,841 template rows, scored pairs 344,841
  (`submission/kind-mango_v36.result.json`). Name pattern `kind-mango_v<n>`.
- Hold-out: months {1, 7} of 2025. CLEAN RMSE on `30 <= y <= 7,200`
  (`train_r_all_v26.py:138-142`).
- Label-free price: ambiguity on the 2026 ranking set
  (`price_ensemble.py:180-181`; `88923a9:305`).
- Pooling policy: supervised pools exclude {1, 7}; lookups use 12 months
  (10th F2, M11).
- Evidence: the `R_norm` price landed to 0.01 s (`RECAP.md:34`). The base price
  missed by about 1,356 MSE (`RECAP.md:36`).
- Defects: F14.

## 2. Findings

### F1. The LIRF head reads the two feature families that v26 removed

- Fact: `models/lirf_regime.features.txt:58-95` lists 38 `ec_*` columns.
  Lines 126-143 list 18 `opdi_*` columns. `p_fb` reads the same list plus 14
  rate columns (`train_lirf_regime_v23.py:172-177`). `predict_v30.py:106,115-116`
  builds all three families for these boosters.
- Why it matters: `ec_*` has 0 % coverage in July 2026, 55 % of the scoring
  set (`RECAP.md:151`). The same defect in the base cost 13.1 s live
  (`RECAP.md:149`). Imputation lost 58 s (`RECAP.md:152`).
- Size: the base fix moved 8,154 MSE (316.85² - 303.71²). LIRF holds 26,899 of
  344,841 rows, 7.8 % (`RECAP.md:143,166`). Scaled by row share, the analogy
  gives 636 MSE, about 1.05 s. This is an analogy, not a price.
- Unmeasured: January 2026 `ec_*` coverage; `opdi_*` coverage at LIRF in 2026;
  2025 non-null share of each column.
- Door status: the ninth pass closed "removing `ec_*` from the LIRF head"
  (`88923a9:397-398`). No committed audit records a measurement for it. The
  lever needs the priced test of section 4 before a ship.

### F2. The base keeps one column from the Eurocontrol daily table

- Fact: `ades_arr_atfm_delay_today` joins `daily_features.parquet` on
  destination and date (`features_advanced.py:111-118`). The base reads it
  (`models/lgbm_r_all_v26.features.txt:84`).
- Why it matters: `ec_*` from the same table has 0 % July 2026 coverage.
  RECAP attributes the v28 loss to its removal with the word "likely"
  (`RECAP.md:41,148`).
- Price: unmeasured coverage, unpriced effect.

### F3. NaN-to-zero injections bias two families (10th F3)

- `features_congestion_v2.py:86` fills missing arrival taxi-in with 0 s before
  the 60-minute mean.
- `features_disruption.py:74` fills a missing schedule delay with 0 s.
- Price: unpriced. The fix needs a base retrain (component 3 risk).

### F4. Hyperparameters come from a censored tuner that selected on the hold-out (10th M1, M2)

- `tune_lgbm.py:67` clips `sched_delay` to [-1,800, 3,600]. Line 68 keeps only
  `30 <= y <= 7,200`. Lines 78-79 and 122-131 early-stop and score each trial on
  months {1, 7}.
- `num_leaves` 440 became 220 by hand (`train_r_all_v26.py:119`).
  `linear_lambda` has no search.
- A purified tuner exists, uncommitted, with no recorded result
  (`tune_lgbm_v36.py:1-12`).
- Price: unpriced.

### F5. The `p_fb` calibration and lookups share fit rows (10th M3, F2, M11)

- The isotonic map fits on the same November-December rows that stopped the
  classifier (`train_lirf_regime_v23.py:191-198`).
- Rate maps use all 12 months for scoring (`train_lirf_regime_v23.py:89-95`).
  The band table uses all 12 months (`build_lirf_band_table_v30.py:31-45`).
- v34 and v35 changed these pools. Both moved live by +0.2 s, inside the
  268 MSE lottery (`RECAP.md:32-33`).
- Price: unpriced.

### F6. `R_norm_LIRF` carries a post-mean cap at 4,431 s (10th M5)

- `predict_v30.py:37,158`. The v29 bundle that added it scored -0.45 s together
  with the ITY340 formula (`RECAP.md:38`).
- Price: no isolated price. Do not remove without one.

### F7. Step A holds 0/1 probabilities and feeds the mixture into the normal term (10th M4)

- 0/1 values: bands 25,000-40,000, 40,000-50,000, 70,000-100,000 and
  100,000+ carry a probability of exactly 1 from 18, 16, 4 and 1 rows
  (`models/lirf_band_table_v30.json:90-137`).
- Normal term: `p_norm * p_final` uses the LIRF mixture (`predict_v23.py:94`;
  `predict_v30.py:182,186`). The effective `sd` weight is
  `p_fb + p_norm * p_fb_cal`. The builder writes `mean_norm` but no code reads it
  (`build_lirf_band_table_v30.py:62,64`).
- Dead default: `mean_24h_extra = 1,150` (`predict_v23.py:92-93`) never acts.
  Every band with `p_24h > 0` stores a value (`lirf_band_table_v30.json:112,120,128`).
- Price: smoothing +95 MSE worse (`88923a9:287`). Normal-term change unpriced.
  A 20-row LIRF change cost 260 MSE live in v31 (`88923a9` section 3.2).

### F8. ITY340 constants come from one training row (10th M6)

- `P24_ITY = 5/6`, threshold 70,000 s, offset `86,400 + 1,150`
  (`predict_v30.py:35-38,193`).
- Price: unpriced. Keep as is.

### F9. The per-member clip ships 23 rows at exactly 0 s (10th F7)

- `predict_v30.py:143` clips each base member at 0 before the mean. v33 holds
  23 zero rows: 18 LSZH, 3 LTFM, 2 LEBL (10th F7, `41345e0:91-95`).
- Price correction: if each true value is 1,000 s, the gain of a perfect fill
  is 23 × 1,000² / 344,841 = 67 MSE. The tenth pass states 252 MSE
  (`41345e0:96`). 252 MSE needs a true value near 1,940 s on every row.
- v36 removed the clip and filled 44 rows. It also moved 66 other rows, up to
  3,802 s. Live +0.18 s (`RECAP.md:31,141`). The live gap is inside the lottery,
  so it does not separate the two parts.

### F10. Rows outside the 10 target airports receive the global median

- `predict_v30.py:76` keeps DEP rows at `TARGET_ICAOS`, 10 codes
  (`features_weather.py:16-17`). Template rows outside that set get the median of
  all predictions (`:210-213`).
- The brief lists 11 airports, with LTAI (`docs/PRC_Data_Challenge_2026_BRIEF.md:15,86`).
- Unmeasured: the v33 `NaN filled` count (`predict_v30.py:215` prints it; no
  record exists).

### F11. Non-LIRF rows with `sd > 70,000` have no guard (10th F8)

- 51 ranking rows outside LIRF; 251 training rows, all genuine, mean
  `y = 1,130 s` (10th F8, `41345e0:109-118`).
- Price: unpriced; class risk high.

### F12. Fake fallback labels stay in base training at 9 airports (10th F1, M10)

- Only LIRF has a gate. A perfect `< 1 s` gate at 9 airports is worth 0.35 s
  (10th F1). Detectors at EGLL, LEBL and LTFM lost 5 s live (`RECAP.md:122`).
- Status: closed.

### F13. The LIRF genuine-row gap is the largest structured residual (10th M8)

- Model-to-baseline ratio 0.92 at LIRF, 0.55-0.65 elsewhere; 13,700 MSE
  (`88923a9:374`).
- Price: no lever priced against it. F1 is the first train/serve defect found
  inside this gap.

### F14. The ensemble price is exact for its definition; the base draw still lost (10th M7)

- `price_ensemble.py:181` uses `A_7 * 4 / 18`. With `A` as the mean squared
  member spread, `A * (k - m) / (m * (k - 1))` gives 4/18 for `k = 7`, `m = 3`
  (`88923a9:305`). The tenth-pass claim of `4/21` is wrong (`41345e0:156-162`).
- The v32 base change priced -1,204 MSE and landed about +152 MSE
  (`RECAP.md:36`). The price is an expectation over draws. Members 45-48 needed a
  retry and multi-thread `linear_tree` is not bit-exact (`RECAP.md:144`).

### F15. The pipeline holds dead code and fragile joins (10th F4, P1-P5)

- `add_congestion` runs at `predict_v30.py:103`. `add_congestion_v2` then
  overwrites all 10 of its columns (`features_congestion.py:35-39`;
  `features_congestion_v2.py:20-27,119-120`). No score effect.
- `load_training_categories` sets category lists (`predict_v30.py:41-50,132-134`).
  Each booster file stores `pandas_categorical` and LightGBM 4.7.0 remaps to
  it (for example `models/lgbm_r_all_v26_s42.txt`). No score effect.
- Two frames joined by a positional assert (`predict_v30.py:126-130`).
- 12 copies of the feature build (10th P1); no manifest (10th P3); shared
  mutable entry point (10th P5).

### F16. No coverage monitor exists (10th F6, M9)

- No script compares per-column non-null share between 2025 and the 2026
  ranking frame. F1 and F2 stayed hidden for this reason.

### F17. The `p_fb` gate is the last single-draw model in the LIRF head

- Fact: v33 averages 5 `R_norm_LIRF` members but serves one `p_fb` booster
  (`train_lirf_regime_v23.py:186-194,231`). Its parameters set
  `feature_fraction 0.8` and `bagging_fraction 0.9` with no seed, so a retrain
  gives a different draw.
- Why it matters: the mixture is linear in `p_fb`. A probability spread `δ`
  moves a row by `δ × (sd - R_norm)`, and `sd - R_norm` reaches thousands of
  seconds at LIRF. Step A reads the mixture, so the same spread reaches the
  cell rows.
- Price tool: the ambiguity decomposition is exact for a linear head, as for
  the v33 `R_norm` mean (`88923a9:305`; `RECAP.md:34`).
- Outcome: shipped as v37, regressed (section 4.1).

### F18. `ARVT_1_flt` carries a planned taxi-out signal that no model reads

- Fact: no script reads `ARVT_1_flt` (grep of `src/`). It is the arrival time
  of the flight-plan (M1) trajectory, a planned value like `EOBT_1_flt`
  (`docs/PRC_Data_Challenge_2026_BRIEF.md:170`). `ARVT_1 - EOBT_1` holds the
  planned taxi-out plus the planned flight time. Coverage: 98.9 % of 2025 DEP
  rows, 98.5 % of ranking DEP rows.
- Signal, 2025 hold-out months {1, 7}, CLEAN rows, with medians fit on the other
  10 months: `R = plan_block - route median` has Spearman 0.19 with
  `y - route median`. The mean excess taxi rises monotonically from 19 s in the
  bottom decile of `R` to 247 s in the top decile. The correlation per airport
  is +0.31 at EHAM, +0.28 at EDDM and +0.03 at LIRF.
- Leak check against the ethics rule (never rebuild the actual off-block
  time):
  - share of rows with `|R - y_res| < 30 s`: 5.97 %, against 5.01 % with `R`
    shuffled within the airport;
  - rows with `y_res > 3,600 s`: median `R` is 24 s, and 0 of 225 rows match
    within 60 s.

  The feature does not track the realised taxi. It does not read
  `AOBT_3_flt`, `ARVT_3_flt` or `LOBT_flt`.
- Fast A/B test on the purified v26 frame: `linear_tree` off, learning rate
  0.08, train {2-6, 8-10}, stop {11, 12}, hold-out {1, 7}; the 97 base columns
  against the same plus 3 new columns:

  | seed | CLEAN | FULL | EHAM | LFPG | EDDF | EGLL | LIRF |
  |---|---|---|---|---|---|---|---|
  | 42 | **-2.15 s** | -0.41 | -13.0 | -7.8 | -5.5 | -1.2 | +5.7 |
  | 43 | **-3.35 s** | -10.87 | -11.5 | -7.0 | -5.6 | -13.1 | +6.1 |

  Both seeds clear the 2 s bar with the same sign. The new columns rank 24 to
  41 of 100 by gain. LIRF gets worse, but the LIRF head serves LIRF rows.
- Drift, no labels: with route medians fit on 2025 months {2-6, 8-12}, the
  per-airport median of `R` sits within ±55 s in Jan/Jul 2025. In Jan/Jul
  2026 it is 31-93 s higher at 9 airports (EDDM +93, EHAM +87, LFPG +78) and
  34 s lower at EGLL. The upper quartile rises by 55-174 s at those 9 airports. Longer 2026 flight times would read as longer
  planned taxis. Section 4 gate 1 addresses this.
- Price: unpriced on live. Evidence: hold-out CLEAN on a fast recipe only.

## 3. Ranked levers

Order: expected live gain divided by risk. "Unpriced" gains carry the analogy
or no number. Risk grades cite the closest live precedent.

| rank | change | price | risk | verification |
|---|---|---|---|---|
| 1 | Add `ARVT_1_flt` planned-time features to the base (F18) | unpriced live; fast-recipe hold-out CLEAN -2.15 and -3.35 s on 2 seeds | medium: base change (v27, v28, v32 regressed); 2026 median drift -34 to +93 s | section 4 |
| — | Replace the single `p_fb` booster with a 5-seed mean, v23 recipe unchanged (F17) | **closed**: priced -151 MSE, live +396 MSE (v37, section 4.1) | — | shipped and regressed |
| — | Retrain `R_norm_LIRF` without the columns that lose 2026 coverage (F1) | **closed at 103 MSE**, section 4.2 | — | gate 2 failed |
| 2 | Fill only the 23 v33 zero rows with the airport median; keep the per-member clip (F9) | at most 67 MSE if truth is 1,000 s; loss if truth is near 0 | low size, but below the 268 MSE lottery | diff vs v33 = 23 rows only; do not upload alone |
| 3 | Refit `p_fb` without the same columns (F1) | unpriced | high: two `p_fb` refits lost 0.2 s (`RECAP.md:32-33`) | replay with the mask on the `p_fb` input only |
| 4 | Purified retune of the base (F4) | unpriced | high: base changes v27, v28, v32 regressed | tuned vs `BEST_PARAMS` on hold-out, >= 2 s CLEAN; 2026 diff report |
| 5 | Fit the isotonic map on a month block apart from the stop set (F5) | unpriced | high: `p_fb` artefact changes regressed | calibration error on unseen block |
| 6 | Resolve `ades_arr_atfm_delay_today` coverage in the base (F2) | unmeasured | high: v28 | coverage report first; no base retrain before a price |
| 7 | Guard non-LIRF rows with `sd > 70,000` (F11) | unpriced | high: class risk, 51 rows | calibrated gate plus dominance check |
| 8 | Remove NaN-to-zero fills (F3) | unpriced | high: base retrain | hold-out >= 2 s CLEAN |
| 9 | Change the Step A normal term to `R_norm_LIRF` (F7) | unpriced | high: v31 lost 260 MSE on 20 LIRF rows | price by class scenarios as in `88923a9` section 4 |

Closed doors, unchanged: the 9-airport fallback gate, Step A clip and
smoothing, the 12-month refit, seasonal weights, per-airport caps, XGBoost or
CatBoost in the base, the 7-seed base, `AOBT_3_flt`, `LOBT_flt`, leaderboard
row values, and the `R_norm_LIRF` skew retrain (section 4.2).

## 4. The single next ship

**Change (v38):** add planned-time features from `ARVT_1_flt` (F18) to the
base regressor only. Retrain the 3 base seeds with the v26 recipe plus the new
columns. Keep the LIRF head, Step A, ITY340 and post-processing on v33
defaults. LIRF rows do not read the base, so v38 moves non-LIRF rows only.

**Candidate columns.** `plan_block = ARVT_1 - EOBT_1`, `arvt1_mvt = ARVT_1 -
MVT`, and `plan_taxi_res = plan_block - route median`. The route key is
`(ADEP, ADES, aircraft type)`.

**Acceptance test.** Do the gates in order. If a gate fails, stop, record the
number in this file, and do not upload.

1. **Drift-robust form.** Build 2 variants of `plan_taxi_res`:
   - (a) the static 2025 route median;
   - (b) the trailing 7-day route median over earlier rows only (`MVT < t`),
     computed inside each year.

   On the fast recipe of F18, keep the variants that give at least 2 s CLEAN
   gain on both seeds. From those, pick the one with the smallest 2026 shift of
   the per-airport median residual. Pass if that shift is under 50 s at every
   airport.
2. **Deployed recipe.** Retrain seeds 42-44 with `src/train_r_all_v26.py`
   settings and the chosen columns. Pass if all 3 hold:
   - the 3-member hold-out CLEAN RMSE beats the v26 3-member mean by at least
     2 s (`train_r_all_v26.py:137-142`);
   - no non-LIRF airport is worse by more than 5 s CLEAN;
   - every best iteration is above half the median.
3. **Serve parity.** The default `predict_v33.py` still rebuilds v33 to 0.0000 s.
4. **Serve.** The v38 file has 344,841 rows, no NaN, no negative value. It
   differs from v33 on non-LIRF rows only. Report the per-airport mean shift
   against v33.
5. **Label-free 2026 sanity.** Per airport, the mean v38-minus-v33 shift on the
   ranking set is within ±60 s. A larger shift means the drift of gate 1 reached
   the predictions.

Upload v38 alone. No live price exists. The hold-out CLEAN gain is the
evidence, and the base-change record (v27, v28, v32) is the risk.

#### v38 gate log, 2026-09-12

1. **Gate 1 failed.** Fast recipe, CLEAN delta against the 97-column base:

   | variant | seed 42 | seed 43 | max 2026 median shift |
   |---|---|---|---|
   | (a) static route median | -1.55 s | -2.38 s | 93 s (EDDM) |
   | (b) trailing 7-day median | +0.53 s | -2.61 s | 19.5 s (EGLL) |

   Variant (a) on seed 42 gave -2.15 s in the first A/B run and -1.55 s here.
   The run-to-run noise of the fast recipe is about 0.6 s.
2. **Change of plan, user decision.** Run gate 2 with variant (a) anyway.
   Gate 2 carries the evidence bar. Gate 5 (±60 s mean shift per airport)
   replaces the gate 1 drift limit. Upload only if gates 2 to 5 pass.
   - Code: `src/features_plan.py`; `src/train_r_all_v26.py --plan` writes the
     `lgbm_r_all_v38_*` files and `lgbm_r_all_v38.holdout.json`;
     `src/predict_v38.py` serves them.
3. **Gate 3 passed.** The default `predict_v33.py` still rebuilds v33 with a
   maximum difference of 0.0 s on 344,841 rows.
4. **Gate 2 failed, twice, with identical numbers.** Deployed recipe, 100 features:

   | seed | best iteration | hold-out FULL | hold-out CLEAN |
   |---|---|---|---|
   | 42 | 1,863 | 385.42 | 264.73 |
   | 43 | **6** | 651.72 | 418.71 |
   | 44 | 1,269 | 391.44 | 265.65 |
   | 3-member mean | — | 439.61 | 277.27 |
   | shipped v26 3-member mean | — | 392.33 | 266.46 |

   - Seed 43 early-stops at iteration 6 in both runs. Seed 42 repeats iteration
     1,863 exactly, so the collapse is deterministic, not thread noise. The
     shipped v26 seed 43 trains normally on the same recipe without the new
     columns (`RECAP.md:39`).
   - The iteration check fails, and the mean is +10.8 s CLEAN worse.
   - Seeds 42 and 44 alone beat the v26 3-member mean as single models. Do not
     ship a hand-picked subset of seeds: that selects seeds on the hold-out.
   - The raw columns stay within ±84,000 s in both years, so a unit outlier does
     not explain the collapse.

**v38 stopped. No upload.** The `ARVT_1` signal stays open for one change of
form with fresh gates:

- use `plan_taxi_res` alone, clipped to ±3,600 s;
- drop the raw `plan_block` and `arvt1_mvt`, which carry the flight time and
  the 2026 drift into the `linear_tree` leaf models.

### 4.0 Closed: v37 `p_fb` seed mean

**Change (v37):** serve the mean of 5 calibrated `p_fb` members, seeds 42-46,
each with the v23 recipe and its own isotonic map on the stop months. Keep the
base, `R_norm_LIRF`, Step A, ITY340 and post-processing on v33 defaults.

- Train: `src/train_p_fb_lirf_seeds.py`
- Price: `src/price_p_fb_seeds.py`
- Serve: `src/predict_v37.py`, through `predict_v30.main(p_fb_members=...)`

**Acceptance test.** Do the gates in order. If a gate fails, stop, record the
number in this file, and do not upload.

1. **Recipe.** Every best iteration is above half the median of the five.
   Every stop AUC is within 0.005 of the v23 value 0.858 (`RECAP.md:160`).
2. **Parity.** `predict_v33.py` on the refactored `predict_v30.py` rebuilds
   `kind-mango_v33.parquet` to 0.0000 s on all 344,841 rows.
3. **Draw check.** On the 2026 LIRF rows, the mean squared distance from the
   v23 final prediction to the 5-member mean is at most 3 × `1.5 × A`. An
   independent draw sits at `1.5 × A` in expectation. A larger distance means v23
   is not a draw from this recipe, so the price does not apply to it.
4. **Price.** `A` of the final LIRF predictions, scaled by 26,899 / 344,841, is
   at least 268 MSE, the member-draw lottery. Record the predicted live RMSE
   `sqrt(91,125 - gain)`.
5. **Serve.** The v37 file has 344,841 rows, no NaN, no negative value, and
   differs from v33 on LIRF rows only.

Upload v37 alone. Compare the live score with the gate 4 prediction.

### 4.1 v37 gate results and live debrief, 2026-09-12

| gate | result | pass |
|---|---|---|
| 1 recipe | best iterations 193, 252, 189, 135, 127; stop AUC 0.8576-0.8609 (`models/lirf_p_fb_seeds.log.json`) | yes |
| 2 parity | `predict_v33.py` rebuild: max difference 0.0 s on 344,841 rows | yes |
| 3 draw | v23 distance to the mean / `1.5 × A` = 0.97 (`submission/v37_price.json`) | yes |
| 4 price | `A` = 1,934 MSE on LIRF rows, gain 151 MSE, predicted 301.62 s | **no**, under 268 MSE |
| 5 serve | 344,841 rows, no NaN, no negative; 25,548 LIRF rows differ, 0 non-LIRF rows; mean change 16.67 s | yes |

Gate 4 failed. More seeds cannot clear it: the gain limit is 1,934 × 5/4 scaled,
189 MSE. The user chose to upload v37 against the bar.

**Live: 302.52 s** (`submission/kind-mango_v37.result.json`), +0.65 s against
v33. The live MSE rose by 396. The price predicted a fall of 151. The miss is
547 MSE, twice the 268 MSE lottery.

What the score implies, from aggregate scores only:

1. The 5-member mean scores 91,521 MSE. By the decomposition, the average
   member of this recipe scores about 91,672 MSE.
2. The v23 booster scores 91,125 MSE inside v33. It beats the average member of
   its own recipe by about 547 MSE on the 2026 labels.
3. The draw check measures distance, not error. It cannot see this gap.

Lessons:

- The ambiguity price holds only when the shipped model is an average draw on
  the scoring labels. The tool landed once (v33, `R_norm`) and missed twice
  (v32 base, v37 `p_fb`). A miss outside the lottery is evidence that the
  shipped single model is a good draw. Do not replace a shipped member on a
  label-free price alone.
- Keep the v23 `p_fb` booster. Close the seed-mean lever for `p_fb` and for the
  base. v37 shipped under the 268 MSE bar and regressed. Keep the bar.
- One slot remains on 2026-09-12 (UTC). No priced candidate clears the bar.

### 4.2 Closed: `R_norm_LIRF` retrain without the skewed columns

**Change:** retrain the 5 `R_norm_LIRF` members with the v33 recipe minus the
column set `S`. Keep `p_fb`, the base, Step A, ITY340 and post-processing on
v33 defaults.

**Acceptance test.** Do the gates in order. If a gate fails, stop, record the
number in this file, and close the lever.

1. **Coverage report, no labels.** For each column in
   `models/lirf_regime.features.txt`, compute the non-null share on LIRF DEP
   rows for 2025 and for the ranking set, January and July apart. Set `S` to
   every column with a gap above 20 points. Expected members: the 38 `ec_*`
   columns. Pass if `S` is not empty.
2. **Priced replay, no training.** Score the v33 LIRF head on the 2025
   hold-out LIRF rows twice. Arm A uses the columns as built. Arm B sets `S` to
   NaN in the `R_norm` input with the 2026 per-month pattern of gate 1. Pass if
   Arm B LIRF CLEAN RMSE is worse than Arm A by at least 2 s.
   Record the price: `ΔMSE × (LIRF ranking rows / 344,841)`, and the predicted
   live RMSE `sqrt(91,125 - price)`.
3. **Retrain.** Run the `train_r_norm_lirf_seeds.py` recipe with the feature
   list minus `S`, seeds 42-46. Pass if:
   - every best iteration is above half the median;
   - on the masked hold-out, the new head beats Arm B by at least 2 s LIRF CLEAN;
   - on the unmasked hold-out, the new head is not worse than Arm A by more
     than 2 s LIRF CLEAN.
4. **Serve.** Pass if:
   - `predict_v33.py` still rebuilds v33 to 0.0000 s;
   - the new file differs from v33 on LIRF rows only;
   - the file has 344,841 rows, no NaN and no negative value.

#### Gate results, 2026-09-12

Scripts ran from the session scratchpad on cached LIRF frames: 160,704 rows for
2025, 26,899 rows for the 2026 ranking set.

1. **Gate 1 passed.** `S` holds 15 columns. 14 are daily `ec_*` traffic and
   pre-departure delay columns, same day and lag 1. Their coverage is 100 % in
   2025 and January 2026, 0 % to 3.2 % in July 2026. The last is
   `opdi_runway_entries_prev_60m`: 82 % in 2025, 60 % in January 2026.
   A row counts as covered for an `opdi_*` column only if it is non-null and
   non-zero, because OPDI counts read 0 when no event exists
   (`features_opdi.py:31-32`). The `opdi_live_*` columns hold 0.1 % coverage
   in 2025, so they carry no skew. The 12 arrival-ATFM `ec_*` columns sit at
   4 % in both years.
2. **Gate 2 failed.** The v33 LIRF mixture on 2025 LIRF rows in months {1, 7}
   scores CLEAN 466.52 s with `S` as built. With `S` masked to the 2026
   pattern in the `R_norm` input, it scores 468.25 s. The delta is +1.73 s,
   under the 2 s bar. The price is 103 MSE on the ranking set, or 301.87 to
   301.70 s. That is under the 268 MSE lottery. The F1 analogy of 636 MSE
   overstated the damage 6-fold.
3. **Extra replay, mask on the `p_fb` input only:** CLEAN +8.02 s worse, but
   full MSE 5,098 lower on the ranking scale. The signs disagree, so a few
   extreme rows drive the full MSE. The `p_fb` rate maps also read the
   hold-out months (F5). This is not evidence for lever 3.

Lever 1 is closed at 103 MSE. No upload followed from this test.

## 5. Contradictions

| # | claim A | claim B | resolution |
|---|---|---|---|
| C1 | README names v30 as the scoring stack and `predict_v30.py` as the entry (`README.md:36,43,189,222`) | v33 is the best submission (`README.md:9`; `RECAP.md:34`; `predict_v33.py:12-16`) | v33 is current; README is stale |
| C2 | README model card: v21, 153 features (`README.md:85,93`) | base reads 97 features (`models/lgbm_r_all_v26.features.txt`) | 97 for the base, 153 for the LIRF head |
| C3 | README tail rules: Section 6.3 classifier and ITY340 as `(5/6)(86,400 + R_all_v21) + (1/6) R_all_v21` (`README.md:134-151`) | 6.3 retired in v22 (`RECAP.md:168`); ITY340 constant formula (`predict_v30.py:193`; `README.md:48`) | code governs |
| C4 | README: `R_norm_LIRF` 5-member mean "awaiting live confirmation" (`README.md:54-56`) | live 301.87 confirms the price (`README.md:79`) | confirmed |
| C5 | README: Step A uses `86,400 + 1,150` (`README.md:47`) | Step A uses per-band `mean_24h_extra` 1,003, 1,203.5, 1,167 (`predict_v23.py:91-94`; `lirf_band_table_v30.json:112,120,128`) | code governs |
| C6 | v26 removed `ec_*` and `opdi_*` (`README.md:75`); "both shipped until v26 removed them" (`41345e0:85-89`) | LIRF head still reads both (`models/lirf_regime.features.txt:58-95,126-143`) | F1 |
| C7 | Ninth pass closes "removing `ec_*` from the LIRF head" (`88923a9:397-398`) | no committed audit records a measurement; the tenth-pass closed list omits it (`41345e0:295-298`) | reopen only through section 4 |
| C8 | Brief: 11 airports with LTAI (`docs/PRC_Data_Challenge_2026_BRIEF.md:15,86`) | code and README: 10 airports (`features_weather.py:16-17`; `README.md:4`) | F10; count the template rows outside the 10 |
| C9 | Tenth pass: zero rows worth 252 MSE of "the live 91,192" (`41345e0:96`) | 23 rows at 1,000 s give 67 MSE; 91,192 is the v30 MSE, v33 is 91,125 | F9 |
| C10 | Tenth pass: exact factor `4/21` (`41345e0:156-162,211`) | ninth-pass formula gives 4/18 (`88923a9:305`; `price_ensemble.py:181`) | F14 |
| C11 | Tenth pass M11 and the v36 debrief name a cause for 0.18-0.22 s live gaps (`41345e0:179-185`; `RECAP.md:31-32`) | the member-draw lottery is 268 MSE, about 0.44 s (`88923a9:151,332`) | the gaps are not evidence |
| C12 | Tenth pass: Step A uses `p_norm * base` (`41345e0:32`) | code uses `p_norm` times the LIRF mixture (`predict_v23.py:94`; `predict_v30.py:182,186`) | F7 |
| C13 | Rule: no 0/1 probability from a finite count (`88923a9:43,250`; `41345e0:309`) | Step A ships four bands at probability 1 from 1 to 18 rows (`lirf_band_table_v30.json:90-137`) | smoothing priced worse; rule not applied to the table |
| C14 | README quick start trains `R_norm_LIRF` with `train_lirf_regime.py` (`README.md:184`) | v33 reads the `train_r_norm_lirf_seeds.py` members (`predict_v33.py:12`) | add the seeds script to the steps |
| C15 | RECAP "Where we stand": rank 45 at 370.63 s; 2/5 slots used on v14, v15; "nothing gets under v13" (`RECAP.md:63-79,202-214`) | status 301.87 s, rank 44 (`RECAP.md:3`; `README.md:9-10`) | blocks are stale |
| C16 | README: Section 6.3 mix uses 1,220 s (`README.md:145`) | RECAP: 1,150 s (`RECAP.md:138`) | retired component; no score effect |
