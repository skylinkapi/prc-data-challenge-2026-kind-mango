# Model analysis — twelfth pass

Status on 2026-09-13: live best **301.87 s (v33)**, rank 44 of 111. Top score
263.46 s. v34 to v37 regressed by 0.18 to 0.65 s. v38 failed its gate. v39
shipped the section-4 rewrite from a cold start and scored **352.19 s live**
(+50.32 vs v33); the rewrite dropped 37 columns the audit had not accounted
for (turnaround, congestion, OSM path, taxi-distance, disruption, OPDI-live).
The audit's expected-gain table in section 5 assumed those families stayed
in the base and only the section-4 measures were layered on top.

This pass replaces the eleventh pass. It does not price 0.1-s levers on the
v33 stack. It audits the construction of the model for the causes of the
302 s score: the data, the label, the feature families, the model class, the
tail rules, the evaluation protocol and the code. Section 4 is the ordered list
of measures for the new model. It holds no code.

Earlier passes stay in git:

- eleventh pass: `git show 72bb3fd:docs/MODEL_ANALYSIS.md`
- tenth pass: `git show 41345e0:docs/MODEL_ANALYSIS.md`
- ninth pass: `git show 88923a9:docs/MODEL_ANALYSIS.md`

Evidence rules:

- Every number with a table id (T1 to T14) comes from a label-free diagnostic
  that this pass ran on the 12 training parquets and on `ranking.parquet`.
  The appendix holds the tables. The scripts ran from the session scratchpad
  and are not in the repo. Section 4, measure P1 asks for their port.
- Two quick signal tests (T12, T13) train a plain LightGBM on July 2025 rows
  with 14 to 17 columns, weeks 1 to 3 against week 4. They rank candidate
  features. They are not hold-out evidence for the deployed frame.
- A number without a table id cites a file and line, or is marked
  **unmeasured**.

Guardrails, unchanged: no `AOBT_3_flt`, no `LOBT_flt`, no row value derived
from the leaderboard, one change per upload.

## 0. Units

| quantity | value | source |
|---|---|---|
| v33 live MSE | 301.87² = 91,125 | `RECAP.md` |
| 1 s of live RMSE at this level | about 604 MSE | arithmetic |
| ranking rows | 344,841 DEP rows, all at the 10 target airports | T7 |
| hold-out rows, 2025 months 1 and 7, `y > 0` | 344,336 | T4 |
| one row with a 24-h label error, predicted at 1,000 s | 85,400² / 344,841 = 21,150 MSE, +34 s live | arithmetic |
| one tail row missed by 10,000 s | 290 MSE, +0.5 s | arithmetic |
| one clean row missed by 1,000 s | 2.9 MSE | arithmetic |

The hold-out and the ranking set have almost the same row count. MSE per row
on the hold-out maps one-to-one onto the live MSE.

## 1. Where the live MSE sits

### 1.1 The label has four classes

The label is exact: `TAXITIME_SEC_mvt = MVT_TIME - BLOCK_TIME` on 100 % of
DEP rows, and the mirror identity holds on 100 % of ARR rows (T1). The label
error therefore comes from the airport's `BLOCK_TIME`, not from the join.

| class | definition | 2025 share | rows in 2025 | note |
|---|---|---|---|---|
| clean | `30 <= y <= 7,200`, not fallback | 90.9 % | 1,895,000 | the regressor's domain |
| fallback | `|y - sd| <= 5` with `sd = MVT - SCHED` | 16.6 % at LIRF; 3.8 to 5.8 % at EDDM, EGLL, LEBL, LEMD, LFPG, LTFM; 0.6 % at EDDF, EHAM, LSZH | about 90,000 | `BLOCK_TIME` equals the schedule (T2, T8) |
| tail | `7,200 < y <= 80,000`, not fallback | 0.007 % | 139 | real long holds at EGLL, LFPG, LTFM; messy LIRF rows (T5) |
| 24-h | `y > 80,000`, not fallback | 0.001 % | 14 | `BLOCK_TIME` one day early (T6) |

The fallback definition changed in this pass. Section 2, D1 gives the
evidence. The deployed code uses `|y - sd| < 60` (`train_lirf_regime.py:38`).

### 1.2 The MSE budget on the 2025 hold-out

A naive predictor, the per-airport clean median fitted on the other 10 months,
scores 666.1 s on the hold-out. Its MSE splits by class (T4):

| class | MSE share of 443,700 | rows |
|---|---|---|
| clean | 150,800 | 313,000 |
| fallback (`< 60` definition) | 147,000 | 31,000 |
| 24-h | 128,500 | 7 |
| tail | 17,400 | 46 |

Seven rows hold 29 % of the naive MSE. The same seven rows, predicted at
`86,400 + normal taxi`, cost 28 MSE. A perfect regime oracle on the fallback
and 24-h classes takes the naive predictor from 666 s to 410 s with no change
to the clean rows (T4).

### 1.3 The budget of the deployed model

The deployed base scores CLEAN 266.46 s on the hold-out
(`models/lgbm_r_all_v38.holdout.json`, `clean_old`). That is 71,000 MSE on
91 % of the rows, or about 64,500 MSE of the 91,125 live total, if the 2026
clean rows behave like the 2025 hold-out. The other 26,600 MSE sit in the
fallback, tail and 24-h classes and in the 2025-to-2026 shift. No end-to-end
hold-out score of the v33 stack exists in the repo, so the split is an
estimate (M5).

Arithmetic for the target: 263 s is 69,200 MSE. With the artefact classes
unchanged at 26,600, the clean rows would need 42,600 MSE, or 218 s CLEAN.
That is not a realistic single move. The new model must move both parts:

1. the clean-row error, from 266 s toward 245 s CLEAN, worth about 10,500 MSE;
2. the artefact classes, from about 26,600 toward 15,000 MSE.

Section 4 orders the measures by these two budgets.

### 1.4 The ceiling of the ethics stance

`README.md:25-30` states that `AOBT_3_flt` reconstructs the hidden off-block
time on 98.9 % of ranking rows and that the team excludes it. `AOBT_3_flt`
matches the actual off-block within 60 s on 17 to 24 % of 2025 rows (T3,
column `aobt3_diag_only`, measured for the data description only). The gap
between 302 s and 263 s is not evidence that a legal model reaches 263 s.
This pass sets no target below what its own measurements support.

## 2. Findings — data integrity

### D1. The fallback class is a 5-second spike, and it exists at 7 airports

- Fact: the density of `|y - sd|` per second, 2025 DEP rows (T8):
  - LIRF: 1.50 % per second inside `|y - sd| <= 5`, 0.044 % per second from
    6 to 59 s. A 34-fold spike.
  - EGLL, EDDM, LEBL, LEMD, LFPG, LTFM: 0.34 to 0.47 % per second inside
    `<= 5`, 0.07 to 0.10 % per second from 6 to 59 s. A 4 to 6-fold spike.
  - EDDF, EHAM, LSZH: 0.058 to 0.062 % per second inside `<= 5`, 0.10 to
    0.11 % per second from 6 to 59 s. No spike. The centre is thinner than
    the shoulders, which is a smooth punctuality distribution.
- Fallback share with the `<= 5 s` definition, months 1 and 7 (T11): LIRF
  15.3 %, LEBL 5.8 %, LTFM 4.7 %, EDDM 4.2 %, LEMD 3.9 %, EGLL 3.8 %,
  LFPG 3.2 %, EDDF 0.6 %, EHAM 0.6 %, LSZH 0.7 %.
- Why it matters:
  - The deployed definition `< 60 s` labels 20.3 % of LIRF rows as fallback.
    3.8 points of those are punctual genuine rows (T8). The `p_fb` classifier
    and the Step A table learn a target with 19 % contamination at LIRF.
  - The tenth pass used `<= 1 s` at the 9 other airports and priced a perfect
    gate at 0.35 s (`41345e0:58`). That window holds 3 of the 11 integer
    seconds of the spike. The ceiling scales by about 3.7, to about 1.3 s.
  - The MSE ceiling of a perfect gate against the clean median at the 9
    non-LIRF airports is 4,892 on the hold-out (T11). The deployed base reads
    `sched_delay` and `mvt_eobt1` with linear leaves, so it already recovers
    part of it. The residual against the deployed model is **unmeasured**.
- Label-free 2026 check (T9): the same statistic on ARR rows,
  `|BLOCK - SCHED| <= 5 s`, gives the same 7-airport pattern and moves by at
  most 0.2 points from 2025 to 2026 at 9 airports, and by +1.0 point at LIRF.
  The reporting artefact is present in the scoring set at the 2025 rate.

### D2. The 24-h class decides tens of seconds and has 14 training rows

- Fact (T6): 14 rows in 2025 with `y > 80,000`: 12 at LIRF, 1 at LFPG, 1 at
  LSZH. All 14 have a null flight record. 11 of 14 have `sd > 14,400`. 10 of
  the 12 LIRF rows have `sd` between 50,500 and 93,600 s. `y - 86,400` is a
  normal taxi, 770 to 2,000 s, on every row.
- The LFPG row (`sd = 1,740`) and the LSZH row (`sd = 939`) look like normal
  departures on every column. No feature separates them. Each such row in
  the scoring set costs about 21,000 MSE, +34 s live.
- The deployed Step A table fits 11 bands on 127 LIRF rows with counts from
  1 to 27 (`models/lirf_band_table_v30.json`). Four bands carry a probability
  of exactly 1. The table includes the hold-out months (F5 of the eleventh
  pass). The band `50,000 to 60,000` mixes 6 fallback and 3 24-h rows; the
  hedge is MSE-optimal for that split and costs 340 MSE per fallback row and
  1,350 MSE per 24-h row on the ranking scale.
- Label-free 2026 check (T9): ARR rows with `y > 80,000` are 0 in both 2025
  and 2026. The 24-h error is a DEP-side artefact. Its 2026 rate is unknown.

### D3. The tail class is a real operational class, and `mvt_eobt1` tracks it

- Fact (T5): 139 rows with `7,200 < y <= 80,000` in 2025. EGLL 34, LFPG 36,
  LIRF 40, LTFM 21, LSZH 5, EHAM 3.
- At EGLL, LFPG and LTFM the rows have a flight record on 71 to 83 % of rows,
  `sd` median about 10,000 s, and `mvt_eobt1` within 0 to 20 % of `y` on the
  sampled EGLL rows. These are long ground holds where the airline updated
  the EOBT to the real pushback. The regressor can learn them from
  `mvt_eobt1` when it sees enough of them; there are 34 per year at EGLL.
- At LIRF the tail rows have a null flight record on 78 % and `sd` median
  2,550 s. They belong with the LIRF null-flight class of D2, not with the
  operational holds.

### D4. The 2025-to-2026 structure is stable, with two exceptions

- Fact (T10): per airport, the null-flight share, the `sd` quantiles, the
  EOBT coverage, the stand and runway null shares and the ADEP mismatch share
  sit within 0.5 points of the 2025 months 1 and 7, except:
  - EHAM: null flight record 1.97 % to 3.16 %; `sd > 14,400` 0.39 % to
    1.31 %. The messy class tripled at EHAM.
  - EGLL: null flight record 1.16 % to 0.62 %.
- The template covers every DEP row at the 10 airports and no other row.
  LTAI has no DEP row in the ranking set (T7). Eleventh-pass F10 closes.
- Unseen categorical levels in 2026 (T14): EDDM stands 4.55 % of rows,
  EDDF operators 1.46 %, all other columns under 0.6 %. 3,978 ranking rows sit
  on a stand with fewer than 20 training rows. These rows fall to the
  missing-value branch of every stand split and to NaN in the stand encoder.

### D5. External data coverage in 2026

- METAR (T14): every month of 2025 and 2026 has 1,482 to 1,489 reports per
  station with 100 % temperature and visibility. No cliff.
- Eurocontrol daily: the pre-departure delay family is 100 % in January 2026
  and 0 % in July 2026. The arrival ATFM column that the base still reads
  (`ades_arr_atfm_delay_today`) is 41 to 63 % non-null in every period. It is
  sparse, not shifted. Eleventh-pass F2 closes.
- OPDI (T14): the per-flight taxi records fall from 1,547 to 1 at EDDM and
  from 458 to 6 at LEBL between July 2025 and January 2026, and LIRF holds 0
  to 3 records in any month. The LIRF head reads 18 `opdi_*` and 9
  `opdi_live_*` columns for a signal that does not exist at LIRF.

### D6. Rows and keys

- `MVT_ID_mvt` is unique and integer-valued in a double column (T1). The
  template and the ranking DEP rows match 1:1.
- 49 DEP rows share airport, flight number and take-off minute with another
  row; 3 `FLIGHT_ID_mvt` values appear twice (T1). Negligible.
- 0.019 % of DEP rows have `y <= 0`. The filter `y > 0` is correct.

### D7. Timestamp resolution shapes the label

- `MVT_TIME` carries microseconds. `SCHED_TIME` and `BLOCK_TIME` are on a
  coarser grid at 7 airports, and on a fine grid at EDDF, EHAM and LSZH (T8:
  the integer-valued residuals cluster at 0 to 6 s and at 60 s at the 7
  airports). A regime definition that uses exact equality, or that uses a
  window wider than the grid, reads the wrong class. `<= 5 s` matches the
  grid at every airport.

## 3. Findings — method

### M1. One MSE regressor serves a four-class label at 9 airports

- The base regressor (`train_r_all_v26.py`) trains on every `y > 0` row at the
  10 airports with the L2 objective. It receives fallback rows at 3.2 to
  5.8 % at six airports (D1), and it has no regime head there. LIRF alone has
  the `p_fb` mixture.
- With `sched_delay` as a feature and linear leaves, the base can imitate
  `y = sd` inside a leaf. That is an implicit mixture with no calibration and
  no bound. The 24-h and tail rows enter the same fit and pull leaf values.
- The LIRF head is the right structure. It applies to 7.8 % of the rows. The
  fallback class outside LIRF is 4.5 times larger than the tenth pass measured
  (D1), and it has no head.

### M2. `linear_tree` runs on unscaled, zero-imputed features

- LightGBM's own guidance for `linear_tree` is: encode missing values as NaN,
  not 0; rescale features to similar mean and standard deviation; expect the
  linear leaf to extrapolate. The deployed frame violates the first two:
  - zeros for missing: `wind_cross_kt`, `wind_head_kt`, `gust_kt`
    (`features_weather.py:122-124`); every congestion count
    (`features_congestion_v2.py:57`); `arr_taxi_in_mean_60m` sums NaN taxi-in
    as 0 (`:86`); `dep_sd_mean_60m` reads a missing schedule as 0
    (`features_disruption.py:74`); `secs_since_last_dep_same_rwy` defaults to
    3,600 (`features_advanced.py:99-101`); `ceiling_ft` defaults to 25,000
    (`features_weather.py:133`); OPDI counts read 0 when no event exists.
  - scale: `sched_delay` spans -2,000 to 100,000; counts span 0 to 60;
    coordinates sit near 50; `openc_*` medians sit near 1,000.
- Symptoms on the live file: 23 rows at exactly 0 s; 7 non-LIRF predictions
  over 7,200 s (T13); v36 moved 66 rows by up to 3,802 s when it removed one
  clip; v38 seed 43 stopped at iteration 6 on a deterministic collapse
  (`72bb3fd` section 4).
- The signed-log copies (`train_r_all_v21.py:37-38`) are a partial patch for
  scale. They do not fix the zero encoding.
- Price of a switch to constant leaves with explicit anchor features:
  **unmeasured**. The linear tree gave -15.6 s live in v21 together with
  three other changes. Section 4, measure C2 sets the test.

### M3. Hyperparameters come from a censored tuner and were never retuned

- `tune_lgbm.py:67-68` clips `sched_delay` to `[-1,800, 3,600]` and keeps
  `30 <= y <= 7,200`. It early-stops and scores each trial on the hold-out
  months. `num_leaves` went from 440 to 220 by hand for the linear tree
  (`train_r_all_v26.py:151`). `linear_lambda = 1.0` has no search.
- The purified tuner `tune_lgbm_v36.py` exists with no recorded result.

### M4. The evaluation protocol hides the live metric

- Every gate since v24 uses CLEAN RMSE on `30 <= y <= 7,200`
  (`train_r_all_v26.py:173`). The live metric is FULL RMSE. Seven rows hold
  29 % of the naive FULL MSE (section 1.2). A change that moves those rows
  is invisible to every gate.
- No script scores the v33 stack end to end on the 2025 hold-out. The base
  and the LIRF head report on different row sets. The number that the
  leaderboard measures was never measured offline.
- The band table, the rate maps and the isotonic map read the hold-out months
  (eleventh pass F5). The hold-out FULL number for LIRF is therefore
  optimistic, and the v34 refit that removed the leak scored inside the noise.
- The label-free ambiguity price held once (v33) and missed twice (v32, v37).
  It prices seed variance only. It cannot see bias.

### M5. The feature set misses the strongest cheap signals

Quick signal tests on July 2025 clean genuine rows, plain LightGBM, weeks 1
to 3 against week 4 (T12, T13). The base has airport, runway, operator, type,
stand, hour, weekday, `sd`, `mvt_eobt1`, `mvt_iobt`, `eobt1_sched`, three
load counts, the arrival taxi-in mean, `dep_sd_mean_60m` and `dep_fnull_60m`.

| candidate family | definition | RMSE change | per-airport range |
|---|---|---|---|
| neighbour EOBT tempo | median and mean of `mvt_eobt1` over departures in the previous 30 min at the airport; the same on the same runway; the same over the same operator in the previous 2 h | **-9.2 and -10.9 s** on two seeds, with `dep_sd_mean_60m` in the base | LIRF -31 to -34, LTFM -19 to -21, LEBL -12 to -15, LSZH -10 to -11, EGLL -2 to -5, EHAM 0 |
| take-off order | count of departures within 30 min that filed a later EOBT and took off earlier; count that filed an earlier EOBT and took off later | -8.2 s (without the tempo family) | LIRF -32, LEBL -14, LTFM -10, EGLL -7 |
| stand re-occupation gap | seconds from the last arrival in-block on the same stand to take-off, when under 1,500 s, else NaN | -2.9 s | EGLL -10 |
| queue between planned off-block and take-off | departures and arrivals at the airport, and departures on the same runway, with take-off between `EOBT_1` and `MVT` | -0.9 s | EGLL -4, LIRF -2 |
| all four families | | **-15.5 s** (291.9 to 276.5 s) | |

- The stand re-occupation gap is also a hard floor: `y >= gap - 30` holds on
  95 to 100 % of the rows where it exists (T12). It exists on 4 to 28 % of
  rows. The floor alone binds on 0.22 % of rows and moves the RMSE by 0.04 s.
  Its value is as a feature, not as a post-processor.
- Every input is present on the 2026 ranking rows: `MVT_TIME` and `EOBT_1_flt`
  of every DEP row, and `BLOCK_TIME` of every ARR row. None reads a DEP
  `BLOCK_TIME`, `AOBT_3_flt` or `LOBT_flt`. The organiser permits features from
  other movements (`RECAP.md`, Discord 2026-09-09). The take-off order family
  reads movements up to 30 min after `MVT`. Measure C1 tests a backward-only
  variant, which sits inside the organiser's stated wording.
- The ninth pass closed "queue and bank features" (`88923a9:399`). That
  closure has no recorded measurement, and it did not test EOBT-anchored
  constructions. T12 and T13 reopen it with numbers.

### M6. Target encoders read contaminated labels

- `features_operator.py:37-42` computes median, count and standard deviation
  of `y` per operator key on the training months. The median resists the tail.
  The standard deviation does not: one 24-h row sets `openc_*_std` to about
  10,000 for its key. The encoders include fallback rows at their airport's
  rate.
- The fallback-rate maps for `p_fb` (`train_lirf_regime_v23.py:52-96`) use the
  `< 60 s` definition (D1).

### M7. Post-processing carries unpriced constants

| constant | value | evidence | file |
|---|---|---|---|
| `R_NORM_CLIP` | 4,431 s | bundled in v29 with the ITY340 formula, -0.45 s | `predict_v30.py:38` |
| `NORMAL_MEAN_LIRF` | 1,150 s | all-month mean | `predict_v30.py:39` |
| `P24_ITY` | 5/6 | one 2025 row | `predict_v30.py:36` |
| `ITY_SD_THRESHOLD` | 70,000 s | one 2025 row | `predict_v30.py:37` |
| `GATE_SD` | 14,400 s | hand pick | `build_lirf_band_table_v30.py:20` |
| `FB_TOL` | 60 s | hand pick, wrong grid (D1) | `train_lirf_regime.py:38` |
| `K_SMOOTH` | 30 | hand pick | `train_lirf_regime_v23.py:39` |
| `MIN_COUNT` | 20 | hand pick | `features_operator.py:23` |
| OPDI lag | 600 s | leak fix | `features_opdi_live.py:67` |
| METAR tolerance | 45 min | hand pick | `features_weather.py:115` |
| turnaround tolerance | 12 h | hand pick | `features_turnaround.py:53` |
| `mean_24h_extra` fallback | 1,150 s | never fires | `predict_v23.py:93` |
| NaN fill | global median of predictions | never fires, 0 rows (T7) | `predict_v30.py:220-223` |
| per-member clip at 0 | | 23 zero rows shipped | `predict_v30.py:159` |

None of the hand picks has a recorded sweep.

### M8. The stack is 150 scripts with 12 copies of the feature build

- `predict_v30.py` imports from 6 training scripts and 13 feature modules. It
  builds `add_congestion` and then overwrites all 10 of its columns with
  `add_congestion_v2` (eleventh pass F15).
- No feature manifest, no unit test, no fixed random state in `p_fb`, no
  end-to-end hold-out script, a global mutable `EC_NUM_COLS`
  (`features_eurocontrol.py:46`), and pickled encoders bound to pandas
  internals.
- Reproduction takes about 2 hours and is not bit-exact
  (`REPRODUCE.md:96-97`).
- This is the reason the eleventh pass could not price most of its findings.
  Every measurement needs the full 2-hour build.

## 4. Measures for the new model

The list is ordered inside each group. Groups P and A come first because
every later measure depends on them. Each measure names its acceptance test.
Do not ship a measure that fails its test.

### P. Protocol

- **P1. Build one evaluation harness before any model.** One script builds
  the feature frame once for 2025 and for 2026 and caches it. One script
  scores any stack end to end on the 2025 hold-out months 1 and 7 and
  reports FULL RMSE, CLEAN RMSE, and the MSE per class (clean, fallback,
  tail, 24-h) per airport. Port the diagnostics of this pass (T1 to T14)
  into the repo as that script's report.
  Test: the harness reproduces the v33 file to 0.0 s on the ranking set, and
  it reports the v33 FULL and CLEAN hold-out numbers, which the repo does not
  hold today.
- **P2. Gate on FULL and on the class table, not on CLEAN alone.** A change
  passes when FULL improves by at least 2 s, CLEAN does not worsen by more
  than 1 s, and no class at any airport worsens by more than 500 MSE on the
  hold-out scale.
- **P3. Fit every lookup, encoder, band table, rate map and calibrator on
  months 2 to 6 and 8 to 12 only.** The hold-out months 1 and 7 enter nothing
  before the final refit.
- **P4. Run the label-free 2026 checks on every candidate file.** Per
  airport: mean and quantiles of the predictions against the 2025 hold-out
  labels (T7); the ARR-side fallback rate (T9); the count of predictions
  over 7,200 s and over 80,000 s; the count of rows at exactly 0 s; the count
  of unseen categorical levels (T14). Record the report next to the file.
- **P5. One change per upload, and a written price before the upload.** The
  price is the hold-out FULL delta from P1 plus the 2026 report from P4.
  Keep the 268 MSE lottery bar for seed-only changes.

### A. Data layer

- **A1. Define the label classes once, in config, with the 5-second
  fallback window.** Fallback is `|y - sd| <= 5`. 24-h is `y > 80,000` and
  not fallback. Tail is `7,200 < y <= 80,000` and not fallback. Every script
  reads the definition from config. Test: the class shares reproduce T2 and
  T11.
- **A2. Treat EDDF, EHAM and LSZH as airports without a fallback class.**
  Their `|y - sd| <= 5` share is the punctuality background (D1). Train no
  fallback head there.
- **A3. Add a per-column coverage monitor to the harness.** For every
  feature, report the non-null and non-zero share for 2025, January 2026 and
  July 2026, per airport. Fail the build when a column drops by more than 20
  points between 2025 and either 2026 month. This is the v26 lesson (-13 s)
  as a rule.
- **A4. Drop the OPDI families from every model.** Coverage collapses at
  EDDM and LEBL in 2026 and never existed at LIRF or LTFM (D5). The LIRF head
  still reads 27 OPDI columns.
- **A5. Keep the Eurocontrol daily family out of every model, including the
  one arrival column.** It is sparse in both years (D5) and its family is 0 %
  in July 2026.
- **A6. Encode missing values as NaN in every feature.** No zero, no 3,600,
  no 25,000 default (M2). Counts of events in a window are 0 when the window
  is observed and empty; they are NaN when the context is absent.
- **A7. Compute target encoders on clean rows only, with a robust spread.**
  Median and interquartile range per key, fitted on months 2 to 6 and 8 to
  12, clean class only (M6). Add a count column. Keep `MIN_COUNT` as a config
  value and sweep it in C4.
- **A8. Give unseen and rare stands a group fallback.** Map each stand to its
  apron or terminal prefix, and use the prefix as a second categorical. 4.55 %
  of EDDM rows and 1.2 % of all rows sit on stands with fewer than 20 training
  rows (D4).

### B. Features

- **B1. Add the neighbour EOBT tempo family.** For each departure: median,
  mean and 75th percentile of `mvt_eobt1` over the departures with take-off in
  the previous 30 and 60 min at the airport; the same on the same runway; the
  same over the same operator in the previous 2 h. Exclude the row itself.
  Quick test: -9 to -11 s CLEAN with the schedule-based tempo already in the
  base (T13). Test: hold-out FULL and CLEAN per P2, on the deployed frame.
- **B2. Add the take-off order family, backward-only first.** For each
  departure: among departures with take-off in the previous 30 min, the count
  that filed a later `EOBT_1`. Then the symmetric count in the next 30 min as
  a second variant. Quick test: -8 s for the two-sided form (T13). Test: per
  P2, and record the delta between the backward-only and the two-sided form.
- **B3. Add the stand re-occupation gap as a feature.** Seconds from the last
  arrival in-block on the same stand to take-off when under 1,500 s, else NaN.
  Quick test: -3 s, EGLL -10 s (T13). Use it as a feature, not as a floor.
- **B4. Add the queue-between-anchors family.** Departures at the airport,
  departures on the same runway and arrivals at the airport with a movement
  time between `EOBT_1` and `MVT`, and between `IOBT` and `MVT`. Quick test:
  -1 s (T12). Keep only if P2 passes on the deployed frame.
- **B5. Add the planned-time residual from `ARVT_1_flt` in its clipped form
  only.** `plan_taxi_res` clipped to ±3,600 s, with the route median from the
  fit months. The raw `plan_block` and `arvt1_mvt` collapsed a seed in v38
  (`72bb3fd` section 4). Test: per P2 with the seed-collapse check.
- **B6. Add the anchor deltas that the regime head needs.** `mvt_eobt1`,
  `mvt_iobt` and `sd` exist. Add `eobt1_minus_iobt`, `sd_minus_mvt_eobt1`,
  the flag `eobt1_equals_sched` (55 % of rows, T3), and the flag
  `flt_null`. The fallback classes attach to these anchors (D1, T3).
- **B7. Remove dead and duplicate columns.** `ice_accretion_1hr` is all NaN
  (`RECAP.md`). `add_congestion` v1 is overwritten. `stand_lat`, `stand_lon`,
  `rwy_thr_lat`, `rwy_thr_lon` duplicate the stand and runway categoricals.
  Test: CLEAN does not worsen by more than 1 s after removal.

### C. Model

- **C1. Train a regime head at every airport with a fallback class.** One
  binary classifier per airport group for `|y - sd| <= 5`, trained on months
  2 to 6 and 8 to 12, early-stopped on a blocked month pair, calibrated on a
  third block. Inputs: the anchor deltas of B6, operator, flight-number
  prefix, stand prefix, aircraft type, hour, and the fallback-rate encoders
  with the `<= 5 s` definition. Airports: LIRF, LEBL, LTFM, EDDM, LEMD, EGLL,
  LFPG. Serve `p_fb * sd + (1 - p_fb) * R_norm` as at LIRF today.
  Ceiling on the hold-out: 4,892 MSE at the 6 non-LIRF airports against the
  clean median (T11); the gain against the deployed base is smaller and is
  the number to record. Test: per P2, per airport, and calibration error
  under 0.02 on the calibration block.
- **C2. Decide the leaf type by measurement, not by inheritance.** Train the
  base twice on the A6 frame with the B families: constant leaves with the
  anchor deltas and their signed logs, and linear leaves with the same
  columns rescaled. Test: P2 on both; ship the winner; record the count of
  predictions at exactly 0, over 7,200 and the max absolute shift against
  the other variant on the 2026 set.
- **C3. Train the clean-row regressor on clean rows only, and let the heads
  own the other classes.** The base of C2 trains on `30 <= y <= 7,200`, not
  fallback. The regime head of C1 owns the fallback class. The tail head of
  D1 below owns the tail and 24-h classes. Test: end-to-end FULL per P1.
  This reverses the v21 choice of an unfiltered base. v21 won because the
  tail had no owner then; the heads are the owner now. If the end-to-end FULL
  is worse than the unfiltered base by more than 2 s, keep the unfiltered
  base and record the number.
- **C4. Retune once, on the purified protocol, for the chosen leaf type.**
  Use `tune_lgbm_v36.py` as the protocol: uncensored target, the final
  feature list, blocked stop months, hold-out untouched until the final
  comparison. Sweep `num_leaves`, `min_data_in_leaf`, `learning_rate`,
  `feature_fraction`, `lambda_l2`, and `MIN_COUNT` of A7. Test: tuned against
  the current parameters on the hold-out FULL, at least 2 s.
- **C5. Fix every seed and average 3 members only after C1 to C4 land.**
  `p_fb` today has no seed (`train_lirf_regime_v23.py:186-189`). Seed
  averaging is priced by the ambiguity rule and is the last step.

### D. Tail policy

- **D1. Build one tail head for the null-flight class at every airport.**
  Rows with a null flight record and `sd > 3,600`. Three classes: fallback,
  24-h, normal, with the 5-second definition. Fit a multinomial model on the
  fit months with `sd`, `sd mod 86,400`, hour, stand prefix, flight-number
  prefix and airport, with a Dirichlet prior of 2 pseudo-counts per class.
  Serve the MSE-optimal mixture `p_fb * sd + p_24 * (86,400 + normal) +
  p_norm * normal`, with `normal` from the clean regressor. This replaces the
  11-band table with counts of 1 to 27, the four bands at probability 1, and
  the ITY340 rule. Test: hold-out FULL on the null-flight rows at least equal
  to the deployed table; no probability of exactly 0 or 1; the 2026 count of
  predictions over 80,000 s recorded before upload.
- **D2. Never predict a 24-h value on a row that has a flight record.** All
  14 rows of 2025 have a null flight record (D2). A false 24-h prediction on
  a genuine row costs 21,000 MSE.
- **D3. Give the long-hold class its own evidence.** The tail rows at EGLL,
  LFPG and LTFM follow `mvt_eobt1` (D3). Check on the hold-out that the C2
  regressor, with the B families, predicts over 5,000 s on those rows. If not,
  add a hold detector for `mvt_eobt1 > 5,400` with a flight record, and
  serve `max(regressor, 0.8 * mvt_eobt1)` on the detected rows. Test: FULL
  on the tail class per airport.
- **D4. Accept the undetectable 24-h rows.** The LFPG and LSZH rows of 2025
  (D2) have no signal. Expect about one such row per two months. Do not add
  a hedge for them: a hedge on every null-flight row costs more than it
  saves (ninth pass section 6.1).

### E. Post-processing

- **E1. Clip the final prediction to `[30, upper]` where `upper` is
  `100,000`.** Do not clip members. Fill nothing with a global median; the
  template has 0 rows outside the model (T7). Test: 0 rows at exactly 0 s.
- **E2. Remove `R_NORM_CLIP`, `NORMAL_MEAN_LIRF`, `P24_ITY` and
  `ITY_SD_THRESHOLD`.** D1 replaces them. Record the FULL delta of each
  removal in the harness.
- **E3. Write the submission with the template's integer dtype.**
  `TAXITIME_SEC_mvt` is `int32` in the template and `float64` in the v33 file
  (T7). Round to the nearest second.

### F. Code and reproducibility

- **F1. One package, one feature builder, one config, one entry point per
  stage.** Layout per `AGENTS.md` section 3: `features/`, `heads/`, `train/`,
  `predict/`, `evaluate/`, with config in one file. Retire the 12 copies of
  the feature build behind a `v_old/` directory or delete them after the new
  stack reproduces v33.
- **F2. Save every fitted artefact as parquet or JSON, not pickle.** Encoders,
  rate maps, band tables and calibrators. Bind no artefact to pandas
  internals.
- **F3. Cache the feature frame.** The 2-hour build blocked every price in
  the eleventh pass. A cached frame turns a price into a 2-minute job.
- **F4. Add tests for the label identity, the class shares, the coverage
  monitor and the serve parity.** Each is one assertion on the cached frame.
- **F5. Fix `num_threads` for the final refit and record the LightGBM
  version.** Multi-threaded linear trees are not bit-exact (`REPRODUCE.md`).

## 4.1. v39 outcome (2026-09-13)

The section-4 measures were built in order and shipped as `kind-mango_v39`
after two failed uploads (int32 dtype was silently rejected; the second try
with `v2b` naming was also ignored — the scorer needs `v<int>.parquet` and
float64). The stack lives under `src_v2/`.

Hold-out (2025 months 1 and 7), per the harness:

| metric | v39 | v33 (RECAP) |
|---|---|---|
| FULL RMSE | 338.06 | 392.33 |
| CLEAN RMSE | 264.26 | 266.46 |

Class MSE shares (per-row) on hold-out:

| class | v39 share | contribution |
|---|---|---|
| clean | 67,003 | 194 rows worth of the total |
| fallback | 7,772 | 22 rows worth |
| tail | 14,522 | 42 rows worth |
| 24h | 24,859 | 72 rows worth |

The 24h share sits at LFPG (20,211 alone, one row per T6) and at LIRF
(4,648). D4 accepted the LFPG row. D1 handled the LIRF 24h class down from
32,787 (base + regime only) to 4,648 (base + regime + null-flight LIRF head).

Live: **352.19 s**, +50.32 s against v33.

Cause: the audit assumed the base kept the 97 v33 columns; the `src_v2/` base
holds 60. The dropped families were turnaround (v20, live -13.8 s in a
bundle), congestion v2 and disruption (rolled into v20), OSM path, taxi
distance, weather beyond the four base columns, OPDI-live, and the
Eurocontrol ATFM column that A5 removed. Section 5 priced +10 to +25 s of
gain from the section-4 measures on top of v33; a cold-start rebuild loses
the base and re-runs the whole climb from v20's 345 s.

The section-4 measures still improved what they touched: CLEAN moved from
266 (v33 hold-out) to 264, and the 24h class at LIRF fell by 28,000 MSE. The
regressions are structural: the base's own error grew because it lost
features, not because the measures were wrong.

The path that actually beats v33 is to layer the section-4 measures on the
v33 stack, one at a time under P5. The `src_v2/` code base is the anchor for
that layering: the harness (P1), the coverage monitor (A3), the neighbour
tempo (B1) and the take-off order (B2) are the highest-priced items from
section 5 that a v33 layering could adopt without a base rebuild.

Two upload slots remain on 2026-09-13.

## 5. Expected gains

Grades: A = measured on the deployed frame or live; B = measured in a quick
test; C = ceiling arithmetic; D = unmeasured.

| measure | expected live gain | grade | risk |
|---|---|---|---|
| B1 neighbour EOBT tempo | 5 to 10 s | B | the quick test used 17 columns; the deployed frame holds 97 |
| B2 take-off order | 3 to 6 s, part shared with B1 | B | two-sided form reads later movements; test backward-only |
| C1 fallback heads at 6 airports | 1 to 4 s | C | ceiling 4,892 MSE against the median; residual against the base unmeasured |
| C2 leaf type decision | 0 to 3 s | D | the linear tree has live evidence inside a bundle |
| D1 tail head with priors | 0 to 3 s in expectation, high variance | C | 20 to 40 rows decide it; the deployed table is already MSE-optimal per band |
| B3 stand gap, B4 queue, B5 plan residual | 1 to 3 s together | B | small, cheap |
| A3 to A6 hygiene | 0 to 2 s | D | prevents the next v25-type loss |
| C4 retune | 0 to 2 s | D | never done on the purified protocol |
| E3 integer dtype | 0 | A | correctness only |

The sum of the graded gains reaches 10 to 25 s. That range crosses 300 s with
margin and does not reach 263 s. Section 1.4 states why no lower target is
set.

## 6. Contradictions with earlier passes

| # | earlier claim | this pass | resolution |
|---|---|---|---|
| C1 | tenth pass F1: a perfect `<= 1 s` gate at 9 airports is worth 0.35 s | the spike is 11 s wide on the grid; `<= 1 s` holds 3 of 11 seconds; the ceiling against the median is 4,892 MSE (T11) | D1; measure C1 |
| C2 | ninth pass closed "queue and bank features" (`88923a9:399`) | EOBT-anchored tempo and order features move a quick model by 8 to 12 s (T12, T13) | M5; measures B1, B2 |
| C3 | eleventh pass F10: template rows outside the 10 airports get the global median | 0 template rows sit outside the 10 airports (T7) | closed |
| C4 | eleventh pass F2: `ades_arr_atfm_delay_today` has an unmeasured coverage cliff | 41 to 63 % non-null in every period (T14) | sparse, not shifted; measure A5 drops it for a different reason |
| C5 | ninth pass section 7: "a score below 300 s needs information the model does not have" | the information is in the ranking file: other departures' `EOBT_1` and `MVT`, other arrivals' `BLOCK_TIME` | M5 |
| C6 | eleventh pass table 1.1: "no filter change has a priced gain" | the fallback window is a filter, and its grid is wrong at every airport (D1, D7) | measure A1 |
| C7 | `RECAP.md` v26 debrief: `ec_*` has 0 % July 2026 coverage | the pre-departure family is 0 %; the arrival ATFM family is 49 % (T14) | both true; family-level statement |

## Appendix — diagnostic tables

All tables come from the 12 training parquets (2025) and `ranking.parquet`
(2026). DEP rows at the 10 target airports with `y > 0` unless stated.
`sd = MVT_TIME - SCHED_TIME` in seconds.

### T1. Keys and identities

- 4,167,797 rows; 2,085,047 DEP and 2,082,750 ARR.
- `|y - (MVT - BLOCK)| <= 1` on 100.000 % of DEP rows; `BLOCK` null on 0 %.
- `|y - (BLOCK - MVT)| <= 1` on 100.000 % of ARR rows.
- `MVT_ID_mvt` unique and integer-valued.
- 49 DEP rows duplicate (airport, flight, minute); 3 `FLIGHT_ID_mvt` repeat.

### T2. Label classes per airport, 2025, `< 60 s` fallback definition

| airport | fallback `< 60` | 24-h | tail | clean | rows | clean median | clean p99 |
|---|---|---|---|---|---|---|---|
| EDDF | 6.83 | 0.00 | 0.00 | 93.17 | 230,141 | 838 | 1,823 |
| EDDM | 9.54 | 0.00 | 0.00 | 90.46 | 167,334 | 775 | 1,929 |
| EGLL | 9.04 | 0.00 | 0.01 | 90.94 | 239,546 | 1,320 | 2,701 |
| EHAM | 6.79 | 0.00 | 0.00 | 93.21 | 247,951 | 742 | 1,781 |
| LEBL | 9.28 | 0.00 | 0.00 | 90.71 | 179,705 | 906 | 1,916 |
| LEMD | 7.96 | 0.00 | 0.00 | 92.04 | 212,242 | 985 | 1,931 |
| LFPG | 7.58 | 0.00 | 0.02 | 92.38 | 239,552 | 955 | 2,448 |
| LIRF | 20.31 | 0.01 | 0.02 | 79.66 | 160,704 | 1,018 | 2,634 |
| LSZH | 5.69 | 0.00 | 0.00 | 94.08 | 134,907 | 713 | 1,711 |
| LTFM | 9.72 | 0.00 | 0.01 | 90.27 | 272,965 | 964 | 2,631 |

Shares in percent. `y <= 0`: 0.019 % overall, 0.23 % at LSZH.

### T3. Which timestamp `BLOCK` matches: share of rows with `|y - (MVT - X)| < 60`

| airport | SCHED | EOBT_1 | IOBT | LOBT | AOBT_3 (description only) | EOBT_1 = SCHED | null flight |
|---|---|---|---|---|---|---|---|
| EDDF | 6.83 | 9.19 | 8.23 | 8.18 | 20.45 | 54.75 | 0.85 |
| EDDM | 9.54 | 13.04 | 11.16 | 11.15 | 23.54 | 55.78 | 0.60 |
| EGLL | 9.04 | 12.65 | 10.22 | 10.20 | 17.73 | 52.92 | 0.59 |
| EHAM | 6.79 | 9.13 | 8.56 | 8.56 | 22.92 | 54.81 | 1.65 |
| LEBL | 9.28 | 12.94 | 11.32 | 11.19 | 24.10 | 53.46 | 1.03 |
| LEMD | 7.96 | 11.92 | 10.06 | 9.98 | 24.12 | 53.28 | 0.44 |
| LFPG | 7.58 | 10.81 | 9.99 | 9.99 | 19.95 | 62.73 | 1.57 |
| LIRF | 20.31 | 22.80 | 20.60 | 20.60 | 17.03 | 39.78 | 0.93 |
| LSZH | 5.72 | 7.43 | 6.94 | 6.94 | 23.36 | 59.79 | 1.70 |
| LTFM | 9.72 | 10.45 | 10.53 | 10.45 | 7.91 | 58.40 | 1.33 |

Fallback rate by flight record, `< 60` definition: with a record 5.7 to
9.7 % (LIRF 20.1 %); without a record 0.2 to 3.9 % at 9 airports, 48.5 % at
LIRF.

### T4. MSE decomposition, hold-out months 1 and 7, naive per-airport clean median

344,336 rows. Naive RMSE 666.1 s. MSE contribution per airport and class:

| airport | clean | tail | fallback `< 60` | 24-h | total |
|---|---|---|---|---|---|
| EDDF | 13,674 | 0 | 697 | 0 | 14,371 |
| EDDM | 9,316 | 0 | 604 | 0 | 9,920 |
| EGLL | 24,626 | 3,025 | 1,901 | 0 | 29,551 |
| EHAM | 15,687 | 611 | 720 | 0 | 17,018 |
| LEBL | 8,453 | 0 | 1,673 | 0 | 10,126 |
| LEMD | 9,366 | 0 | 674 | 0 | 10,040 |
| LFPG | 23,917 | 11,136 | 1,249 | 20,144 | 56,447 |
| LIRF | 17,379 | 1,739 | 137,207 | 108,371 | 264,696 |
| LSZH | 7,026 | 801 | 388 | 0 | 8,215 |
| LTFM | 21,322 | 119 | 1,928 | 0 | 23,369 |

Class totals: clean 150,766; fallback 147,041; 24-h 128,516; tail 17,431.
With a perfect regime oracle (fallback rows at `sd`, 24-h rows at `86,400 +
median extra`): RMSE 410.3 s; fallback 100 MSE; 24-h 28 MSE. 52 rows with
`y > 7,200` and not fallback hold 32.9 % of the naive MSE.

### T5. Tail rows, `7,200 < y <= 80,000`, not fallback, 2025

| airport | rows | `y` median | `sd` median | null flight | `sd > 3,600` |
|---|---|---|---|---|---|
| EGLL | 34 | 8,491 | 9,933 | 0.29 | 1.00 |
| EHAM | 3 | 10,972 | 9,956 | 0.33 | 0.67 |
| LFPG | 36 | 8,262 | 9,975 | 0.17 | 0.97 |
| LIRF | 40 | 8,104 | 2,550 | 0.78 | 0.40 |
| LSZH | 5 | 9,538 | 9,946 | 0.60 | 0.80 |
| LTFM | 21 | 7,731 | 12,360 | 0.90 | 1.00 |

EGLL sample, `y` against `mvt_eobt1`: 10,210 / 10,385; 9,611 / 10,085;
10,615 / 11,637; 8,232 / 8,586; 9,468 / 9,894; 8,275 / 9,297; 10,069 /
10,134; 7,311 / 7,555. Correlation of `y` with `sd` on the tail: -0.11.

### T6. 24-h rows, `y > 80,000`, not fallback, 2025

| airport | month | `y` | `sd` | flight record | `y - 86,400` |
|---|---|---|---|---|---|
| LFPG | 1 | 84,240 | 1,740 | null | -2,160 |
| LIRF | 1 | 87,177 | 60,118 | null | 777 |
| LIRF | 3 | 87,247 | 65,463 | null | 847 |
| LIRF | 6 | 88,392 | 67,686 | null | 1,992 |
| LIRF | 6 | 87,168 | 11,694 | null | 768 |
| LIRF | 7 | 87,361 | 73,260 | null | 961 |
| LIRF | 7 | 87,170 | 93,535 | null | 770 |
| LIRF | 7 | 88,132 | 84,236 | null | 1,732 |
| LIRF | 7 | 87,186 | 58,563 | null | 786 |
| LIRF | 9 | 87,543 | 50,581 | null | 1,143 |
| LIRF | 11 | 87,480 | 56,520 | null | 1,080 |
| LIRF | 11 | 87,598 | 60,599 | null | 1,198 |
| LIRF | 12 | 87,605 | 86,102 | null | 1,205 |
| LSZH | 5 | 87,341 | 939 | null | 941 |

Two more LIRF rows have `y > 80,000` and `|y - sd| < 60`; they are fallback
rows with a one-day schedule offset. LIRF rows with a flight record and
`sd > 70,000`: one in 2025, `y = 87,002`, `sd = 87,001` (a fallback row).
Non-LIRF rows with `sd > 70,000`: 251, `y` median 1,111, max 3,032.

### T7. Ranking set and the v33 file

- 689,534 rows; 344,841 DEP; every DEP row is at one of the 10 airports; 0
  template rows outside them. `BLOCK` and `y` null on 100 % of DEP rows.
  Months: January 152,719; July 192,122.
- v33 file: `float64` predictions; 23 rows at 0 s (LSZH 18, LTFM 3, LEBL 2);
  36 rows under 30 s; max 111,654; 103 rows over 7,200 (LIRF 96, EGLL 4,
  LFPG 2, EHAM 1); 3 rows over 80,000.
- Prediction mean against the 2025 hold-out label mean: EDDF 921 / 875,
  EDDM 869 / 832, EGLL 1,371 / 1,382, EHAM 874 / 798, LEBL 954 / 957, LEMD
  1,031 / 1,011, LFPG 1,003 / 1,051, LIRF 1,284 / 1,274, LSZH 809 / 757,
  LTFM 1,144 / 1,070.

### T8. Density of `|y - sd|` per second, 2025

| airport | `<= 5` | `<= 10` | `< 60` | density `<= 5`, % per s | density 6 to 59, % per s |
|---|---|---|---|---|---|
| EDDF | 0.64 | 1.23 | 6.83 | 0.058 | 0.104 |
| EDDM | 4.59 | 4.98 | 9.54 | 0.417 | 0.092 |
| EGLL | 4.45 | 4.87 | 9.04 | 0.405 | 0.085 |
| EHAM | 0.68 | 1.25 | 6.79 | 0.062 | 0.103 |
| LEBL | 5.13 | 5.61 | 9.28 | 0.466 | 0.077 |
| LEMD | 4.09 | 4.46 | 7.96 | 0.372 | 0.072 |
| LFPG | 3.77 | 4.13 | 7.58 | 0.343 | 0.071 |
| LIRF | 16.55 | 18.10 | 20.31 | 1.505 | 0.044 |
| LSZH | 0.63 | 1.12 | 5.70 | 0.057 | 0.094 |
| LTFM | 4.72 | 5.15 | 9.72 | 0.429 | 0.093 |

Shares in percent of rows. Integer-valued residuals at the 7 spike airports
sit at 0, 2, 4, 5, 6 and at 60 s; none sit at 8 to 45 s.

### T9. ARR-side reporting artefact, 2025 months 1 and 7 against 2026

`|BLOCK - SCHED| <= 5` on ARR rows, percent:

| airport | 2025 | 2026 | 2026 January | 2026 July |
|---|---|---|---|---|
| EDDF | 0.46 | 0.39 | | |
| EDDM | 2.79 | 2.64 | | |
| EGLL | 1.89 | 2.05 | | |
| EHAM | 0.48 | 0.50 | | |
| LEBL | 1.90 | 1.88 | | |
| LEMD | 1.83 | 1.95 | | |
| LFPG | 2.03 | 2.11 | | |
| LIRF | 8.00 | 8.98 | 10.55 (`< 60`) | 11.99 (`< 60`) |
| LSZH | 0.51 | 0.54 | | |
| LTFM | 1.74 | 1.75 | | |

ARR rows with `y > 80,000`: 0 in both years. ARR taxi-in medians move by at
most 19 s at any airport. ARR rows: 343,999 in 2025 months 1 and 7; 344,693
in 2026.

### T10. DEP structure, 2025 months 1 and 7 against 2026, percent of rows

| airport | null flight 2025 / 2026 | `sd > 14,400` 2025 / 2026 | `sd` median 2025 / 2026 | `mvt_eobt1` median 2025 / 2026 |
|---|---|---|---|---|
| EDDF | 1.63 / 1.16 | 0.51 / 0.42 | 1,494 / 1,497 | 1,254 / 1,248 |
| EDDM | 0.87 / 0.92 | 0.13 / 0.24 | 1,254 / 1,265 | 1,074 / 1,075 |
| EGLL | 1.16 / 0.62 | 0.25 / 0.27 | 1,734 / 1,615 | 1,563 / 1,560 |
| EHAM | 1.97 / 3.16 | 0.39 / 1.31 | 1,536 / 1,469 | 1,233 / 1,226 |
| LEBL | 1.70 / 1.47 | 0.41 / 0.37 | 1,419 / 1,400 | 1,228 / 1,201 |
| LEMD | 0.87 / 0.64 | 0.26 / 0.32 | 1,531 / 1,582 | 1,284 / 1,307 |
| LFPG | 2.22 / 1.84 | 0.41 / 0.60 | 1,803 / 1,677 | 1,441 / 1,143 |
| LIRF | 1.50 / 1.42 | 0.59 / 0.51 | 1,616 / 1,561 | 1,257 / 1,201 |
| LSZH | 2.34 / 2.85 | 0.17 / 0.32 | 1,454 / 1,538 | 1,234 / 1,242 |
| LTFM | 1.40 / 1.51 | 0.72 / 0.75 | 1,315 / 1,383 | 1,257 / 1,259 |

`sd` is null on 0 % of rows in both years. Stand and runway are null on
under 0.02 %. ADEP mismatch between `_mvt` and `_flt` is 0 %. 2026 rows in
the Step A cell: 43. Non-LIRF rows with `sd > 70,000`: 51. LIRF rows with a
flight record and `sd > 70,000`: 1.

### T11. Fallback `<= 5 s`, hold-out months 1 and 7, and the perfect-gate ceiling

| airport | share % | rows | `sd` median | clean median | MSE if predicted at the clean median | `sd` in 600 to 1,800, % |
|---|---|---|---|---|---|---|
| EDDF | 0.55 | 204 | 791 | 837 | 67 | 80 |
| EDDM | 4.15 | 1,123 | 721 | 774 | 273 | 70 |
| EGLL | 3.82 | 1,535 | 1,264 | 1,320 | 1,019 | 90 |
| EHAM | 0.57 | 234 | 712 | 743 | 84 | 63 |
| LEBL | 5.81 | 1,683 | 960 | 907 | 1,291 | 82 |
| LEMD | 3.89 | 1,375 | 978 | 986 | 376 | 94 |
| LFPG | 3.20 | 1,265 | 901 | 955 | 737 | 87 |
| LIRF | 15.28 | 4,053 | 1,140 | 1,017 | 125,535 | 74 |
| LTFM | 4.66 | 2,168 | 963 | 962 | 967 | 91 |
| LSZH | 0.70 | 156 | 723 | 711 | 78 | 71 |

MSE on the 344,336-row scale. Non-LIRF sum: 4,892.

### T12. Quick signal test 1, July 2025 clean genuine rows

Plain LightGBM, 130,037 train rows (weeks 1 to 3), 53,458 test rows (week 4).
Base: airport, runway, operator, type, stand, hour, weekday, `sd`,
`mvt_eobt1`, `mvt_iobt`, `eobt1_sched`, `dep_prev60`, `dep_prev15`,
`arr_prev60`.

| variant | RMSE | delta | per airport (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM, LSZH) |
|---|---|---|---|
| base | 291.94 | | |
| + take-off order (2 columns) | 283.72 | -8.22 | -0.2, -0.7, -6.8, +0.3, -14.1, -2.1, -4.5, -31.6, -9.9, -8.1 |
| + neighbour EOBT tempo (4 columns) | 279.52 | -12.41 | -2.6, -2.7, -10.5, -0.3, -15.9, -2.9, -4.6, -36.2, -24.9, -12.1 |
| + stand re-occupation gap | 289.01 | -2.93 | -1.3, +1.1, -9.9, -2.2, -2.1, -1.1, -3.1, +0.2, -3.6, -3.2 |
| + all three families | 276.45 | -15.49 | |
| queue between `EOBT_1` and `MVT` (4 columns), separate run with `rwy_prev30` in the base | 290.07 against 290.93 | -0.85 | +1.2, -0.5, -3.8, -0.8, -0.3, +0.1, -0.8, -1.9, +0.3, -1.3 |

Stand re-occupation: the last in-block on the same stand is under 1,500 s
before take-off on 3.9 to 28.1 % of rows; `y >= gap - 30` holds on 95.3 to
100 % of them; median gap 289 to 660 s. Spearman correlation with `y`:
departures between `EOBT_1` and `MVT` 0.33 to 0.61; `dep_prev60` 0.01 to
0.22; `mvt_eobt1` 0.45 to 0.64.

### T13. Quick signal test 2, control with the deployed tempo columns in the base

Base of T12 plus `arr_txi_mean_60m`, `dep_sd_mean_60m`, `dep_fnull_60m`.
Added: median and mean of `mvt_eobt1` over the previous 30 min at the
airport, and the median on the same runway.

| seed | base | + EOBT tempo | delta | per airport |
|---|---|---|---|---|
| 1 | 288.34 | 277.47 | -10.87 | -3.4, -5.2, -4.8, -0.5, -14.5, -1.2, -7.0, -34.0, -20.6, -9.6 |
| 2 | 287.38 | 278.18 | -9.20 | -2.5, -4.6, -2.1, +0.7, -12.2, -1.0, -3.6, -30.9, -19.4, -10.7 |

Gain share: `dep_sd_mean_60m` 1.8 % in the base; the three tempo columns 2.2,
1.0 and 2.7 % with them added.

### T14. External coverage and unseen levels

- METAR, EGLL, LIRF, LTFM: 1,482 to 1,489 reports per month in 2025-01,
  2025-07, 2026-01 and 2026-07; temperature and visibility 100 % non-null.
- Eurocontrol daily: rows through 2026-07-31. Non-null share of the arrival
  ATFM column: 0.42, 0.63, 0.41, 0.49 for 2025-01, 2025-07, 2026-01, 2026-07.
  Pre-departure delay family: 1.0, 1.0, 1.0, 0.0.
- OPDI taxi records per airport, 2025-07 / 2026-01 / 2026-07: EDDF 615 /
  5,582 / 11,388; EDDM 1,547 / 1 / 995; EGLL 3 / 2 / 15; EHAM 67 / 10 / 65;
  LEBL 458 / 6 / 58; LEMD 1 / 1 / 64; LFPG 0 / 0 / 3; LIRF 0 / 0 / 3; LSZH
  11,770 / 5,630 / 12,588; LTFM 0 / 0 / 0.
- Unseen (airport, value) pairs in 2026, percent of rows: stand 0.46 total,
  EDDM 4.55, EDDF 0.83; operator 0.44 total, EDDF 1.46; type 0.02; runway
  0.00; destination 0.24. 3,978 rows on stands with under 20 training rows.
