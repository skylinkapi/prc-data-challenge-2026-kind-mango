# Model analysis, fifteenth pass: audit of v47 and the measures for the new model

Status on 2026-09-17: live best **296.57 s (v47)**, 111 teams, top score
263.46 s. This file is the source of truth for the next model. It replaces
the fourteenth pass. Earlier passes stay in git:

- fourteenth pass (levers L1 to L9, plan under 290 s):
  `git show acb157a:docs/MODEL_ANALYSIS.md`
- twelfth and thirteenth passes (diagnostic tables T1 to T17, v39 debrief):
  `git show b193868:docs/MODEL_ANALYSIS.md`

This pass prices no new lever. It audits the construction of v47 for the
causes of the remaining error: data integrity, leakage, feature mathematics,
the model class, the tail rules, the evaluation protocol and the code.
Section 3 states why the model sits at 296.57 s. Section 4 holds the
findings. Section 5 is the list of measures for the new model. The file
holds no code.

## Progress, 2026-09-23: section 5 measures merged with `WINNING_PLAN.md`

Live best **281.87 s (v67)**. On 2026-09-23 the rank was 57 of 178 at 287.33 s (v57). `docs/WINNING_PLAN.md` ranks
levers L1 to L15; this table maps them onto the measures of section 5.
Live deltas come from the official API (WINNING_PLAN section 3). The team
stance stays on track A: no lever reads `AOBT_3_flt` or `LOBT_flt`
(section 4.9, MX1).

### Done

| measure | upload | live delta, s | state |
|---|---|---|---|
| MS1, MS2 bounds and floors | v48 | -2.33 | accepted |
| MS3 member median | v50 | -0.42 | accepted on v48; v51 to v57 dropped it (L13) |
| MS4 fail on a missing row | v48 onward | | served (`assert_ms4`) |
| MD2, MH1 LIRF head without drifted columns | v49 | +0.42 | rejected |
| MB3 base on served rows | v51 | -5.57 | accepted |
| MB4 calendar | v52 | +0.11 | rejected |
| MB1 anchored offset on `mvt_eobt1` | v54 | +35.86 | rejected, closed |
| MB8 `R_norm`, gate, route medians | v55, v56, v57 | -0.42, -0.33, -0.17 | accepted |
| MB8 operator encoders | v58 | +1.29 | rejected, closed |
| MF4 arrival drift | v59 | +0.37 | rejected, closed |
| MD4 out-of-fold encoders | v60 | +2.25 | rejected, closed |
| L4 v45 parameters on the v57 recipe (MB5) | v61 | +0.99 | rejected |
| L13 MS3 on the v57 base | v62, not uploaded | 0 | no-op: largest non-LIRF member spread 2,376 s, under the 3,600 s trigger; v62 equals v57 on every row |
| L12 Step A normal term reads `R_norm` (MH5, T1) | v63 | -0.24 | accepted; 18 LIRF cell rows move, all down by 100 to 4,241 s. The plan bar is -0.30 s; a deterministic change with no retrain has no retrain noise |
| L10 out-of-fold isotonic map for the v56 gate (MH1, fixes L3) | v64 | -1.14 | accepted; 25,988 LIRF rows outside Step A move, mean +8.7 s, max 1,805 s; rows above 7,200 s 125 to 105 |
| L9 `plan_nm_taxi` = `ARVT_1 - EOBT_1` minus its (ADEP, ADES, type) median (MF5, fixes C2) | v65 | -1.22 | accepted; base retrained with 118 columns; non-LIRF shifts -1.3 to +2.7 s, max 2,188 s |
| L11 weather at `EOBT_1` (MF1), five METAR columns on the v65 base | v66 | +0.33 | rejected; one LSZH row on the 8 January 2026 snow day moved 1,257 to 5,536 s |
| L3 CatBoost second class, 0.5 blend on the base outside LIRF (MB7, fixes M5) | v67 | -2.87 | accepted; hold-out price -2,079 MSE at w 0.5; like-for-like LightGBM control 1,636 MSE worse than v46, CatBoost 1,358 better |
| MB7 on the LIRF head: CatBoost blended 0.5 into `R_norm` | v68 | +0.03 | rejected; hold-out -435 MSE on genuine LIRF rows did not transfer, as v49 |
| Converged base CatBoost (round cap 15,000 instead of 5,000) | v69, not built | hold-out -2,171 against -2,079 MSE at w 0.5 | stopped: the 92 MSE difference is under the 150 MSE bar set before the run; early stop at 6,480 rounds |
| Second base CatBoost, variant B (depth 10, seed 43, `rsm` 0.5), averaged with the v67 member | v70, not built | hold-out -2,150 against -2,079 MSE | stopped: +71 MSE is under the 150 MSE bar; B alone is weaker than A (-1,946) |
| MF2 runway-queue structure, backward windows (same-runway and heavy counts at 5 and 10 min, same-direction departures, runway-set age) on the LightGBM half | v71 | -0.015 | not accepted (inside noise); hold-out -524 to -596 MSE at equal rounds did not transfer; OSM crossings and remote flag not built |
| L7 check: day-level correlation of departure fallback share and arrival schedule-grid share, 2025 | | | passes only at LIRF (0.43) and EDDM (0.34); the other 8 airports sit at 0.02 to 0.19 |
| Discord tail rules: blend heavy-hold rows toward the 2025 curve of y on `mvt_eobt1`; cap normal rows | not uploaded | hold-out +1,639 / -3 MSE | rejected on the 2025 hold-out (`src_v3/measure_tail_rules.py`); see below |
| MP9 read every score | | | v44 299.846, v45 296.125, v50 293.816 now read |
| MX1 organiser ruling | | | resolved 2026-09-18 |

The API scores change one reading: the L1 retune (v45 to v46) cost
+3.01 s live. v61 put the v45 parameters back on the v57 recipe and
landed 288.32 s, +0.99 s. One reading, grade D: MB3 removed the LIRF and
24-h rows that the tuner followed (P3), and the retuned parameters now
suit the served rows. L14 (blend of both recipes) needed v61 within
0.5 s of v57 and does not run.
v61 moved 5 rows of one EHAM day by more than 3,000 s: each has a flight
record, `mvt_eobt1` of 3,905 to 7,443 s and `sd` of 12,543 to 21,014 s.

### Open, in upload order (track A)

| next | WINNING_PLAN lever | measure | upload |
|---|---|---|---|
| 1 | L7 day-level artefact share | MH3 | LIRF and EDDM gate features only; the 2025 check passes there |
| 2 | L15 final 12-month refit | MB8 | last |

A Discord post (2026-09-23) said: make the outliers of the submission look
like the outliers of the training set. The 2026 prediction tail at EHAM is
fatter than the 2025 label tail (37.5 against 3.3 rows per 10,000 above
3,600 s). 143 of these rows fall on 3 to 9 January 2026, all with a flight
record and `mvt_eobt1` of 4,197 to 13,414 s: a real disruption week, like
5 January 2025. Conditioned on `mvt_eobt1`, the v65 predictions sit below
the 2025 label means (-1,216 s above 7,200 s). The v46 members score
January and July 2025 out of sample:

1. A blend with the isotonic curve of y on `mvt_eobt1` (record rows,
   `mvt_eobt1` > 3,600 s, 5,192 rows) costs +390 to +6,711 MSE at base
   weights 0.75 to 0, and costs at every airport. The row-level base beats
   the band mean.
2. A cap at the 2025 99.99 % label quantile of record rows with
   `mvt_eobt1` < 1,800 s moves 3 hold-out rows (-3 MSE) and 10 rows of
   v65. MB3 and MS1 already hold this tail.

Neither rule ships.

Every open lever builds on v67: `python -m src_v3.predict_v67 --weight 0.5` (the
v65 stack plus the full-year CatBoost member `models/catboost_r_all_v67.cbm`). A base retrain builds on
`train_v57_base.train_base` with the `plan_nm_taxi` column.

v64 test on held-out months (`models/lirf_regime_v64.meta.json`): log loss of the gate
falls from 0.383 to 0.364 for `sd` <= 3,600 s and from 0.140 to 0.122 for 3,600 to
14,400 s. For `sd` > 14,400 s (510 rows) log loss is flat (0.1209 to 0.1216) and the
mean bias grows from -0.25 to -2.04 points. That bin fails the plan's per-bin test.

Not on track A: L1, L2, L5, L6 and L8 read `AOBT_3_flt` or `LOBT_flt`.
No WINNING_PLAN lever covers MP1 to MP8, MD1, MD3, MD5 to MD10, MB2,
MB6, MH2, MH4, MF2, MF3 or MC1 to MC7. The stop rule of WINNING_PLAN
11.4 replaces the stop rule of section 6.

## 0. Scope, evidence and units

### 0.1 How this pass was made

1. Eight review lenses read the code, the JSON reports, `RECAP.md` and the
   earlier passes independently: data integrity, leakage, feature
   mathematics, model method, evaluation protocol, pipeline code, strategic
   gap, and an outside review.
2. The lenses returned 93 raw findings. This pass merged the duplicates into
   the findings of section 4.
3. A finding marked **V** was re-read in this pass against the cited code
   lines, JSON keys or git files. A finding marked **R** rests on the lens
   reading of the cited code and was not re-read line by line.
4. The audit machine holds no training data, no ranking data, no trained
   booster and no LightGBM install. No number in this file comes from a new
   model run.

### 0.2 Evidence rules

- A number carries its source: a file and line, a JSON key, a `RECAP.md`
  line, a twelfth-pass table (T1 to T17) or arithmetic.
- MSE uses the 344,841-row scoring scale unless the text says "hold-out
  scale" (344,336 rows). The two scales differ by 0.15 %.
- Grades: A measured live or on the paired harness; B measured on a partial
  frame; C arithmetic ceiling or scenario; D reasoning only.
- A number without a source is marked **unmeasured**.

### 0.3 Units

| quantity | value | source |
|---|---|---|
| v47 live MSE | 296.57² = 87,954 | `submission/kind-mango_v47.result.json` |
| top score | 263.46² = 69,411 | `RECAP.md` |
| gap to the top | 18,543 MSE | arithmetic |
| gap to 290 s | 87,954 - 84,100 = 3,854 MSE | arithmetic |
| 1 s of live RMSE at 296.57 s | 593 MSE | arithmetic |
| one non-LIRF row served at 18,000 s on a 1,000 s taxi | 17,000² / 344,841 = 838 MSE, +1.4 s | arithmetic |
| one 24-h row served at a normal taxi | about 21,000 MSE, +34 s | T4, v39 debrief |
| one clean row missed by 1,000 s | 2.9 MSE | arithmetic |

## 1. The deployed model, exactly

v47 runs `src/predict_v47.py`. It calls `src/predict_v30.py:main` with the
v47 base members, the v41 LIRF regressor members, no cap, and 20 extra
columns joined by movement id (`src/predict_v45.py:24-28`).

| component | rows served | files | trained by |
|---|---|---|---|
| base | every row outside LIRF | `models/lgbm_r_all_v47_s{42,43,44}.txt`, 117 columns | `src/train_r_all_v47.py`: all 12 months, every `y > 0` row at 10 airports including LIRF, fixed rounds 693, 2,045 and 1,154, parameters of `models/tune_lgbm_v43.tuned_params.json`, linear leaves, encoders copied from v26 |
| LIRF regressor `R_norm` | genuine term of every LIRF row | `models/lgbm_r_norm_lirf_v41_s{42..46}.txt`, 166 columns | `src/train_r_norm_lirf_v41.py`: rows with `abs(y - sd) >= 60` and `y < 80,000`, train months 2 to 6 and 8 to 10, stop months 11 and 12, 127 leaves, `BEST_PARAMS`, `linear_lambda` 1.0 |
| LIRF gate `p_fb` | probability that `BLOCK_TIME` is within 60 s of the schedule | `models/lgbm_p_fb_lirf_v23.txt`, 167 columns, isotonic map | `src/train_lirf_regime_v23.py`: all LIRF rows, same months, no seed, 14 fallback-rate columns |
| mixture | `p_fb * sd + (1 - p_fb) * R_norm` on every LIRF row | `src/predict_v30.py:200-203` | |
| Step A | LIRF, null `FLIGHT_ID_mvt`, `sd > 14,400`: `p_fb * sd + p_24h * (86,400 + m) + p_norm * mixture` | `models/lirf_band_table_v30.json`, 11 bands, 127 rows | `src/build_lirf_band_table_v30.py`, all 12 months |
| ITY340 rule | LIRF, `sd > 70,000`, outside Step A: `(5/6) * (86,400 + 1,150) + (1/6) * sd` | `src/predict_v30.py:36-39, 211-215` | one 2025 row |
| extra columns | 13 tempo, 6 tempo quartiles, `plan_taxi_res` | `src_v2` ranking frame, `models/tempo_p2575_rank.parquet`, `models/plan_taxi_res_rank.parquet` | `src_v2/frame.py`, `src/build_tempo_p2575.py`, `src/build_plan_taxi_res.py` |
| post-processing | per-member clip at 0, mean, final clip at 0, left merge on the template, NaN fill with the global median | `src/predict_v30.py:163, 217, 229-235` | |

Serve order: the base mean fills every row; the mixture overwrites LIRF
rows; Step A overwrites the cell rows and reads the mixture as its normal
term; the ITY340 rule overwrites LIRF rows with `sd > 70,000` outside the
cell; the final clip runs last.

## 2. Where the error sits

### 2.1 Hold-out budget of the last end-to-end stack (v41)

Hold-out scale, `models/v41.holdout.json`. FULL 319.93 s, CLEAN 250.83 s.

| airport | clean | fallback | tail | 24-h |
|---|---|---|---|---|
| EDDF | 4,921 | 23 | 0 | 0 |
| EDDM | 2,986 | 69 | 0 | 0 |
| EGLL | 8,349 | 225 | 1,144 | 0 |
| EHAM | 4,718 | 22 | 467 | 0 |
| LEBL | 3,392 | 816 | 0 | 0 |
| LEMD | 3,310 | 128 | 0 | 0 |
| LFPG | 8,018 | 222 | 9,991 | 20,245 |
| LIRF | 13,984 | 5,934 | 524 | 1,273 |
| LSZH | 2,651 | 25 | 449 | 0 |
| LTFM | 8,036 | 226 | 106 | 0 |
| total | 60,365 | 7,688 | 12,681 | 21,517 |

- Two LFPG rows with a null flight record and no observable hold about
  29,650 of the total (fourteenth pass 2.2).
- The LIRF cells are in-sample for the band table and the rate maps
  (finding L2). The LIRF 24-h cell is not a measurement.
- The same stack scores 102,355 on the hold-out and 89,589 live.

### 2.2 Rows the base serves (v46 paired report, non-LIRF)

Hold-out scale, `models/lgbm_r_all_v46.holdout.json`, key `v46`. Appendix
A7 gives the airports.

| class | MSE | movable |
|---|---|---|
| clean | 44,517 | yes |
| fallback | 1,726 | yes, by regime heads (T5) |
| tail | 12,140 | about 2,700 outside the one LFPG row |
| 24-h | 20,232 | no, one LFPG row |

### 2.3 Paired price against live outcome

MSE, negative is an improvement. Live deltas are arithmetic on the result
files.

| ship | change | paired price | live delta |
|---|---|---|---|
| v32 | 7-seed base, 5-member `R_norm` | -1,278 (seed ambiguity) | +78 |
| v33 | 5-member `R_norm` only | -74 (seed ambiguity) | -66 |
| v37 | 5-seed gate mean | -151 (seed ambiguity) | +396 |
| v40 | 13 tempo columns on the base | -805 | -1,146 |
| v41 | tempo columns on `R_norm`, no cap | -415 | -390 |
| v44 + v45 + v46 | tempo quartiles, plan residual, retune | -1,864 | -106 |
| v47 | 12-month refit of the base | not priced | -1,530 |

Single feature changes against a same-run control landed. Stacked changes
did not. The largest recent gain had no offline price. v44 and v45 were
never scored (finding P6), so the stack cannot be split by lever.

## 3. Why the model sits at 296.57 s

The causes in order of expected MSE. Finding ids point to section 4.

1. **The evaluation protocol cannot price the decisions it makes.** One
   split, gates below its own noise, a calendar feature out of support on
   the scoring months, a tuner and a stop rule dominated by rows the base
   never serves, in-sample LIRF numbers, unscored intermediate uploads.
   Consequence: 1,864 MSE priced, 106 landed; the largest lever invisible
   offline; levers closed on noise. Findings P1 to P13, L2.
2. **The base regresses raw seconds with near-unregularised linear leaves**
   on unscaled, zero-imputed anchors, trains on rows it never serves, and
   has no upper bound. Single rows move by 5,000 to 18,000 s at airports
   with no long hold in 2025. Findings M1 to M3, D3. Range 500 to 4,500 MSE,
   grades C and D.
3. **The LIRF head reads 14 Eurocontrol columns that exist on every 2025
   row and on no July 2026 row.** This is the v26 defect (-13.1 s live on the
   base), never fixed in the head. Finding D1. About 700 MSE, range 0 to
   2,000, grade C.
4. **The tail rules serve raw frequencies from 1 to 27 rows**, with
   probabilities of exactly 0 and 1, and Step A counts the schedule weight
   twice. Findings T1 to T4. Range 0 to 1,500 MSE, grade C.
5. **Only the base was refit on 12 months.** The LIRF head, the gate, the
   calibrator, the encoders and the route medians never saw January or
   July. Finding M6. About 400 MSE, grade D.
6. **Reporting artefacts outside LIRF have no head, and two fallback
   definitions coexist.** Findings T5, D2. Ceiling 1,726 MSE, grade C.
7. **Train and serve see different encoders and tables.** Findings L1, L3.
   Unmeasured, grade D.
8. **2026 drift is logged, not handled.** Null-record share doubled at six
   airport-months; the EOBT_1 = schedule share moved 20 points at LTFM and
   25 points at LFPG in July; LFPG arrival runway counts collapsed; the
   operator-stand encoder lost 18 points at EDDF in July. Findings D3 to D6.
   Unmeasured.
9. **Information in reach is unused.** Weather at pushback, queue structure
   at the runway, 2026 arrivals as a drift and validation signal, a
   day-level artefact signal. Findings C3, S1 to S3, P11. Range 400 to 1,900
   MSE, grade D.
10. **The ethics stance sets a ceiling.** Finding X1. A team decision.

## 4. Findings

### 4.1 Evaluation protocol and decisions

**P1. One hold-out split decides every lever, with gates below its own
noise.** V, high.
- Evidence: every run from v40 to v48 uses months 1 and 7 as hold-out and
  the same random stop mask (`src/train_r_all_v40.py:94`,
  `src/train_r_all_v46.py:81`). Gates are 300 MSE
  (`src/train_r_all_v44.py:95-96`), 500 MSE (`src/train_r_all_v46.py:105`)
  and 200 MSE (L8). Single members of one recipe score CLEAN 271.58, 268.48
  and 276.56 s (`RECAP.md:221`), a spread of 4,400 MSE. The six near-null
  arms of Appendix A4 give served-clean deltas with mean +28 and standard
  deviation 89 MSE.
- Mechanism: a fixed threshold on one point estimate turns noise into
  accept and reject decisions. Sequential acceptance on one split biases the
  accepted prices upward.
- Cost: v44 to v46 priced -1,864 MSE and landed -106 (section 2.3). L8
  (-144), L3.e (+102) and v45term (+77) sit within two standard deviations
  of the null spread and carry no information.

**P2. `month` is a numeric feature, and the scoring months were out of
support in every paired run and in the tuner.** V, high.
- Evidence: `month` is numeric in `src/train_lgbm_v23.py:30` and is a served
  column in `models/lgbm_r_all_v47.meta.json`. Hold-out months are {1, 7}
  (`src/train_r_all_v26.py:30`). The tuner trains on {2..6, 8..10} and stops
  on {11, 12} (`src/train_lirf_regime.py:36-37`, `src/tune_lgbm_v43.py:53-54`).
- Mechanism: no training row has month 1. January rows take the month-2
  side of every month split. July rows fall between the 6 and 8 bins. Linear
  leaves with month on the path extrapolate. Every paired price from v40 to
  v46 was measured on that extrapolation. The tuner stop months lie above
  the training maximum.
- Cost: unmeasured. Bounded above by the v47 live gain, -1,530 MSE, which
  also holds the effect of more data and same-season data. January is 44 %
  of the scoring rows.

**P3. The tuner objective and every early stop are dominated by rows the
base never serves; the L1 gate passed on a one-airport rebound.** V, high.
- Evidence: the tuner split and the member stop set carry no airport or
  class filter (`src/tune_lgbm_v43.py:53-54`, `src/train_r_all_v46.py:79-82`).
  Three 2025 24-h rows sit in months 11 and 12 (T6). In the v46 report the
  retune moved LIRF fallback 13,077 to 8,172 and LIRF 24-h 42,274 to 12,426
  (hold-out scale), rows that the LIRF head serves. The served clean class
  moved -662, of which LSZH -527 (2,941 to 2,414). The previous arm had
  moved LSZH clean +315 (2,626 to 2,941, `models/lgbm_r_all_v45plan.holdout.json`).
- Mechanism: squared error on a few rows with `y` near `sd` or near 86,400
  dominates the selection metric. The search rewards parameters that let
  linear leaves follow `sd`. The served rows get the capacity that is left.
- Cost: served gain outside LSZH -135 MSE. Live v41 to v46: -106 MSE for
  three levers.

**P4. The random-row stop set measures interpolation inside a day, not
transfer to a new month.** V, medium.
- Evidence: the stop set is a random 12 % of rows
  (`src/train_r_all_v26.py:116-119`). Best iterations on identical data:
  1,275, 890, 1,240 (v40 control) and 609, 1,799, 1,015 (v46). A November and
  December stop gave 372 (`RECAP.md:222`).
- Mechanism: stop rows share day, weather, congestion windows, tempo windows
  and in-sample encoders with their training neighbours. A one-row 24-h
  swing moves the stop MSE more than the last hundreds of iterations. The
  argmin is noise.
- Cost: unmeasured. The round counts of every shipped member rest on it.

**P5. v47 has no validation, and its round counts use a factor that does
not match the data growth.** V, medium.
- Evidence: `src/train_r_all_v47.py:27` sets `SCALE = 1/(1 - 0.12) = 1.136`.
  v46 trained on 0.88 × 1,740,323 = 1,531,484 rows; v47 trains on 2,084,659
  rows, a ratio of 1.361. Rounds 693, 2,045, 1,154
  (`models/lgbm_r_all_v47.meta.json`). No validation set (`:68`). No v47
  hold-out report exists. The L2 proxy gate of the fourteenth pass (train on
  months 3 to 6 and 9 to 12) did not run (`RECAP.md:299-302`).
- Mechanism: a proportional rule asks for 829, 2,449 and 1,382 rounds. The
  seed-specific counts carry the noise of P4 into three non-exchangeable
  members. No labelled check exists for the shipped file.
- Cost: unmeasured. The v47 gain cannot be split between refit, round count
  and draw.

**P6. v44 and v45 were never scored; the "stacked prices do not transfer"
conclusion is one difference over three levers.** V, high.
- Evidence: `submission/` holds `kind-mango_v44.check.json` and
  `kind-mango_v45.check.json` and no result file for either.
  `RECAP.md:68-69` records "score pending". v46 and v47 went up the next
  day.
- Mechanism: one regressing lever, for example `plan_taxi_res` with 2025
  route medians, produces the same record as a general transfer failure.
- Cost: the failing lever, if one exists, is unknown and stays in the model.

**P7. The gate metric changed from the full class table to the served clean
class without a recorded decision.** V, medium.
- Evidence: twelfth-pass measure P2 gated on FULL and on every class per
  airport. Fourteenth-pass scripts gate on served clean only
  (`src/train_r_all_v46.py:105`). L7 moved served clean -293, fallback -4,
  tail -58 and 24-h -12, total -367, and was rejected as "short by 7"
  (`models/lgbm_r_all_v48wx.holdout.json`, `RECAP.md:70`).
- Mechanism: the live metric is the full MSE over four classes. The gate
  reads 44,517 of about 100,000 hold-out MSE.
- Cost: L7 discarded (-367 MSE on the old recipe). Tail changes on served
  rows go unpriced.

**P8. Paired controls are loaded from disk under non-deterministic
training, and three levers were priced on a superseded recipe.** V, low.
- Evidence: `src/train_r_all_v46.py:42-47` and
  `src/train_r_all_v48wx.py:37-42` score the shipped v45 members as the
  control. `REPRODUCE.md:96` states that multi-threaded linear-tree training
  is not bit-exact. The L7, L8 and L3.e arms train with `BEST_PARAMS`, 220
  leaves and `linear_lambda` 1.0 (`src/train_r_all_v48wx.py:52-55`) after v46
  replaced that recipe.
- Cost: unmeasured run-to-run term in every delta; enough to flip gates
  missed by 7 and 56 MSE.

**P9. The "268 MSE lottery" is a mean penalty, not a spread; the seed price
counted rows the base does not serve.** V, medium.
- Evidence: 268 = A_7 × (7/6) / 3 with a hold-out A_7 of 690, the expected
  penalty of a 3-member mean against an infinite ensemble.
  `src/price_ensemble.py:172-173` measures A_7 over all 344,841 rows,
  including the LIRF rows that the head serves. With the 2026 A_7 of 5,417
  (`submission/v32_price.json`) the same arithmetic gives 2,107 MSE. v37 kept
  the single v23 gate against a 5-seed mean on a +396 live delta
  (`RECAP.md:82`).
- Mechanism: a wrong noise scale set every gate. A mispriced seed test closed
  seed averaging. A live delta below the member-draw noise kept one gate
  draw by leaderboard selection.
- Cost: decision quality; the gate's leaderboard margin does not transfer.

**P10. The label-free 2026 checks compare files but gate nothing, and the
coverage monitor sees null shares only and skips the LIRF head.** V, medium.
- Evidence: `src/check_submission_2026.py:40-51` records per-airport
  `max_abs_diff` and global counts and applies no threshold. v47 against
  v46 moves one EDDM row by 17,996 s, one EHAM row by 9,064 s and one LEBL
  row by 5,898 s (Appendix A3). `src/check_coverage.py:42-48` measures
  non-null and non-zero shares; `:7-9` treats every LIRF-only column as
  informational; the threshold is 20 points (`:38`).
- Mechanism: an extreme row, a value drift and a head-only coverage collapse
  all pass.
- Cost: see M2, D1, D4 and D5.

**P11. The 2026 arrival labels were never used as a validation
instrument.** V, high.
- Evidence: 344,693 arrival rows of 2026 carry in-block time and taxi-in
  (T9, fourteenth-pass fact 7). `RECAP.md:201` used them once as a drift
  meter: 202.40 s on the 2025 hold-out, 214.96 s in 2026; EHAM +128.9 s,
  LIRF +28.9 s, LFPG +24.1 s, EGLL -40.9 s.
- Mechanism: a taxi-in model on the same code path, trained on 2025 arrivals
  and scored on 2026 arrivals, measures the 2025-to-2026 transfer of every
  lever with real labels. The team spent upload slots instead.
- Cost: unmeasured. It prices the 12-month refit before upload.

**P12. The reported best is a minimum over uploads.** R, low.
- Evidence: near-equivalent stacks spread 0.24 to 0.65 s live (v32 to v37);
  v33 (-66 MSE) and v46 (-106 MSE) were kept.
- Cost: roughly 200 to 400 MSE of the progress since v30 is selection on
  the scoring set, grade C.

**P13. The class table drops rows with 0 < y < 30 s.** V, low.
- Evidence: `src/train_r_all_v40.py:46-53` labels them "invalid"; `:62-64`
  sums four classes; FULL covers every row.
- Cost: the class sums do not reconcile with FULL; the zero-clip rows stay
  invisible to every gate.

### 4.2 Leakage and train-serve skew

**L1. The operator target encoders are fitted and applied in-sample; v47
mixes in-sample and out-of-sample months.** V, medium.
- Evidence: `src/features_operator.py:31-46` computes median, count and
  standard deviation per key with `MIN_COUNT` 20.
  `src/train_r_all_v26.py:111-112` fits on the non-hold-out rows and applies
  to the same rows. `src/train_r_all_v47.py:74-75` copies the v26 encoders
  while training on 12 months. The LIRF fallback-rate maps already use
  leave-one-month-out (`src/train_lirf_regime_v23.py:73-86`).
- Mechanism: each training row's label sits inside its own median and
  standard deviation (weight 1/20 at the floor). One 24-h row sets a key's
  standard deviation near 10,000 on the row that caused it. No 2026 row
  contains its own label. v47 trains ten months in-sample and two months
  out-of-sample on the same 16 columns.
- Cost: unmeasured, grade D.

**L2. Every LIRF lookup reads the hold-out months; the LIRF hold-out
numbers are in-sample.** V, medium.
- Evidence: `src/build_lirf_band_table_v30.py:32-38` reads all 12 parquets.
  `src/train_lirf_regime_v23.py:68-95` builds the out-of-fold rate folds and
  the scoring maps over all months; `:155` computes the base rate over all
  months. `src/eval_v33_holdout.py:12-13` states the LIRF numbers are
  optimistic. The LIRF 24-h class is 1,272.5731 in `models/v41.holdout.json`,
  `models/lirf_regime_v41.holdout.json` and `models/lirf_regime_v43.holdout.json`.
  T6 puts 6 of the 14 24-h rows of 2025 in months 1 and 7, 5 of them at LIRF.
  `NORMAL_MEAN_LIRF` is 1,150 s on all months against 1,094.41 s on the fit
  months (`models/v34_constants.json`).
- Mechanism: the table predicts rows it was built from. An honest
  replacement loses the paired comparison to a leaky incumbent. The v39
  honest head scored 4,648 on the class where the leaky table scores 1,273
  (T17).
- Cost: the live effect of the leak is small (v34 and v35 inside noise). The
  cost is the decisions it closed: Step A smoothing and the gate refits.

**L3. The gate is calibrated on its own early-stopping months, and its rate
features are built differently at training and serving.** V, medium.
- Evidence: `src/train_lirf_regime_v23.py:191-193` early-stops on months 11
  and 12; `:196-198` fits the isotonic map on the same months. Training rows
  read 11-month out-of-fold maps; serving reads 12-month maps (`:89-95`). The
  gate has no seed (`:186-189`).
- Mechanism: the calibrator learns selection-optimistic scores. The mixture
  multiplies a probability error by `sd`.
- Cost: a 0.1 error at `sd` = 30,000 s costs 24 MSE per row; the aggregate
  is unmeasured.

**L4. Two served columns count movements after the scored take-off.** V,
policy.
- Evidence: `src/features_congestion_v2.py:35-37, 92, 117` count movements in
  the 10 minutes after the row's take-off. Both columns are in
  `models/lgbm_r_all_v47.meta.json`. The organiser's permission names
  earlier movement rows (`RECAP.md:359`). The twelfth pass shipped the
  take-off order backward-only for that reason.
- Cost: an eligibility risk. The forward order pair priced -93 MSE (v42), so
  the RMSE value of forward counts is small.

**L5. `ades_arr_atfm_delay_today` is a same-day destination total with 4 to
10 % row coverage.** V, low.
- Evidence: `src/features_advanced.py:110-118` joins the destination's daily
  arrival ATFM delay on the take-off date. Per-row non-null share: 4.2 to
  10.0 % in 2025 and 4.0 to 9.5 % in July 2026 (`models/coverage_monitor.json`).
  Twelfth-pass T14 quoted 41 to 63 % from the daily table, not per row.
- Mechanism: the daily total holds delay that accrues after take-off; the
  column acts mostly through its missingness pattern.
- Cost: small, unmeasured.

**L6. Ranking context is truncated at month starts.** R, low.
- Evidence: `src/predict_v30.py:113-117` builds context from the ranking
  file only; the turnaround link reaches back 12 h
  (`src/features_turnaround.py:53`). 1 July 2026 rows have no 30 June
  context; 1 July 2025 training rows had it.
- Cost: at most the first 12 h of July, 1.6 % of July rows. The January 2026
  turnaround coverage drops of 3 to 5 points (Appendix A1) are not explained
  by this.

### 4.3 Data integrity and drift

**D1. The LIRF regressor and gate read 14 Eurocontrol columns that are
present on every 2025 row and absent in July 2026.** V, high.
- Evidence: `src/train_lirf_regime.py:75, 111-115` and
  `src/train_lirf_regime_v23.py:170-176` put the 38 `ec_*` columns of
  `eurocontrol_numeric_cols()` in both LIRF feature lists;
  `src/train_r_norm_lirf_v41.py:72-73` extends that list; `src/predict_v30.py:124,
  182, 191` serves them. `models/coverage_monitor.json` LIRF-only shares:
  7 same-day columns (airport traffic ×3, all pre-departure delay ×2, ATC
  pre-departure delay ×2) are 1.000 in 2025, 1.000 in January 2026 and 0.000
  in July 2026; their 7 lag-1 copies fall from 0.998 to 0.032. LightGBM's
  `NumericalDecision` (`include/LightGBM/tree.h`) converts NaN to 0.0 when
  the split's feature had no missing value in training.
- Mechanism: on every July 2026 LIRF row the gate and `R_norm` read zero
  daily traffic and zero pre-departure delay, below any training value.
  Split tests route the row to the lowest-traffic branch; linear leaves
  extrapolate. The 2025 hold-out has full coverage and cannot see it. The
  base carried the same defect until v26 (-13.1 s live). Twelfth-pass
  measures A4 and A5 prescribed the removal from every model; it never
  happened for the head.
- Cost: about 700 MSE (1.2 s) by pro-rating the v24-to-v26 live gain
  (8,154 MSE over about 177,000 July rows outside LIRF) to the roughly
  15,000 LIRF July rows. Range 0 to 2,000, grade C.

**D2. Two fallback definitions run in the same stack, and neither matches
the residual grid.** V, medium.
- Evidence: `FB_TOL` is 5 in `src/train_r_all_v40.py:30`, `src_v2/config.py:23`
  and `src/build_plan_taxi_res.py:22`; it is 60 in
  `src/train_lirf_regime.py:38` and `src/train_lirf_regime_v23.py:38`, and the
  band table uses `< 60` (`src/build_lirf_band_table_v30.py:46`). T8: integer
  residuals cluster at 0 to 6 s and at 60 s, none at 8 to 45 s. The strict
  `< 60` and `>= 60` tests put the 60-s grid point in the genuine class. L5
  trained `R_norm` on the 5-s mask and moved LIRF clean +3,546 and fallback
  -1,200 (`models/lirf_regime_v43.holdout.json`).
- Mechanism: at LIRF the rows with 6 to 60 s residual behave like fallback
  rows, which contradicts the twelfth-pass 5-s thesis. The report tables and
  the head partition the rows differently. Rows at exactly 60 s teach
  `R_norm` to follow `sd`.
- Cost: unmeasured. Every LIRF head decision reads cells that do not match
  the head's classes.

**D3. The null-record share doubled at six airport-months of 2026.** V,
medium.
- Evidence: `flt_null` non-zero share, 2025 against 2026 (Appendix A1):
  EHAM January 1.6 to 5.0 %; LSZH 1.5 to 3.0 % (January) and 2.7 % (July);
  LIRF July 0.9 to 1.7 %; LEBL July 1.0 to 2.0 %; LEMD July 0.4 to 0.9 %;
  EDDF January 0.8 to 1.4 %. Fourteenth-pass fact 8 names EHAM only.
- Mechanism: the null-record class holds the reporting artefacts (T3, T6)
  and the rows without an EOBT_1 anchor, where linear leaves read `sd` and
  `sd mod 86,400` alone. It is twice as frequent in the scoring set.
- Cost: unmeasured. The class produces the extreme rows of M2.

**D4. The EOBT_1 = schedule share moved by 13 to 26 points at three
airports in July 2026, and the strongest feature shifted by 298 s at
LFPG.** V, medium.
- Evidence: `eobt1_sched` non-zero share (EOBT_1 differs from the
  schedule), 2025 to July 2026: LTFM 40.3 to 19.6 % (strict monitor
  failure), LFPG 35.7 to 61.6 %, EGLL 46.5 to 59.3 %; LFPG January 35.7 to
  44.8 %. T10: LFPG `mvt_eobt1` median 1,441 s (2025 months 1 and 7) against
  1,143 s (2026). `RECAP.md:26-28` logged the failure as "data drift, not
  pipeline defects".
- Mechanism: the information content of `mvt_eobt1` changes by airport and
  month; the neighbour tempo columns move with it. The base has no flag for
  "EOBT_1 is a copy of the schedule" beyond the derived zero.
- Cost: unmeasured. LTFM is 13.8 % and LFPG 11.6 % of the scoring rows.

**D5. Same-runway arrival counts at LFPG collapsed in 2026 and the monitor
passed them.** V, medium.
- Evidence: `arr_same_rwy_prev_15m` non-zero share at LFPG 19.9 % in 2025,
  3.9 % in January and 3.8 % in July 2026; `arr_same_rwy_next_10m` 17.6 % to
  2.9 % and 2.6 %. A 16-point drop is under the 20-point rule
  (`src/check_coverage.py:38, 72`).
- Mechanism: the likely cause is a change of arrival runway identifiers or
  of runway use at LFPG. A zero count is a valid value, so no missing branch
  protects it.
- Cost: unmeasured.

**D6. Unseen and thin stands lose every stand signal at once, and stand
labels pool across airports.** V, medium.
- Evidence: T14: unseen stands on 4.55 % of EDDM rows and 0.83 % of EDDF
  rows; 3,978 rows on stands with under 20 training rows.
  `openc_op_apt_stand_*` non-null share: EDDF July 92.0 to 74.0 %, EDDM July
  90.2 to 80.4 %, LSZH January 88.3 to 78.8 %. `STAND_mvt` and `RUNWAY_mvt`
  are raw strings shared across airports (`src/train_lgbm_v21.py:34-36`). The
  base has no stand-prefix column (`models/lgbm_r_all_v47.meta.json`).
  LightGBM treats unseen categories as missing at prediction.
- Mechanism: an unseen stand loses the categorical, the encoder and the OSM
  geometry together. With `feature_fraction` 0.563, 44 % of trees cannot
  split on the airport and pool stand labels such as "A12" across airports.
- Cost: 100 to 300 MSE, grade C.

**D7. Two arrival taxi-in meters disagree; one reads missing values as 0
and unvalidated labels.** V, low.
- Evidence: `src/features_congestion_v2.py:86` fills missing taxi-in with 0
  and averages every arrival; `src/features_disruption.py:85-86` keeps 30 to
  7,200 s. T9: 8.98 % of LIRF arrivals in 2026 have in-block within 5 s of
  the schedule.
- Cost: small, unmeasured.

**D8. The tempo quartiles use a different neighbour pool from the shipped
tempo.** V, low.
- Evidence: `src/build_tempo_p2575.py:34` uses the frame as its own pool;
  `src_v2/frame.py:347-348` filters the training frame to `y > 0`; the shipped
  tempo uses every departure row (`src_v2/frame.py:353-357`); the ranking
  frame has no label filter.
- Cost: small; a train-serve pool difference on 6 served columns.

**D9. Rows with `y <= 0` and `0 < y < 30` form an unmeasured floor.** V,
low.
- Evidence: `y <= 0` on 0.019 % of 2025 rows, 0.23 % at LSZH (T2). The base
  keeps `0 < y < 30` rows (`src/train_r_all_v26.py:85`). v33 served 23 rows
  at 0 s and 36 under 30 s (T7).
- Cost: about 150 MSE floor if 2026 repeats the 2025 `y <= 0` rate; the
  1 to 29 s share is not reported.

### 4.4 Model method

**M1. The base regresses raw seconds instead of the off-block offset
against the plan.** V, high, missed opportunity.
- Evidence: the label is raw `y` (`src/train_r_all_v47.py:59`). `y` equals
  `MVT - BLOCK` exactly (T1). `mvt_eobt1` ranks first (`RECAP.md:258-265`) and
  equals `y + (BLOCK - EOBT_1)`. `BLOCK` is within 60 s of EOBT_1 on 7 to
  23 % of rows (T3). Linear leaves were adopted in v21 to reproduce the slope
  on the anchors. The README closure "Huber and log target, 408 s" was
  measured on the censored pre-v21 pipeline; v46const kept raw `y`.
- Mechanism: raw `y` spans 30 to 131,167 s. The offset `EOBT_1 - BLOCK` is
  bounded and close to stationary, and the fallback class sits at a known
  offset (`sd - mvt_eobt1`). A tree on the offset needs no linear leaf for
  the slope and cannot extrapolate beyond its leaf values.
- Cost: unmeasured, grade D; plausible range 500 to 3,000 MSE.

**M2. Linear leaves run on unscaled, constant-imputed inputs with a ridge
that does nothing on the features that extrapolate; nothing bounds the
output.** V, high.
- Evidence: linear leaves on raw `sched_delay`, `mvt_eobt1` and `mvt_iobt`
  in seconds next to flags and counts (`models/lgbm_r_all_v47.meta.json`);
  `linear_lambda` 0.0056 in v46 and v47, 1.0 before. Missing context enters
  as constants: wind and gust 0 (`src/features_weather.py:122-124`), ceiling
  25,000 ft (`:132-133`), congestion counts 0 when context is absent
  (`src/features_congestion_v2.py:57`), taxi-in 0 (`:86`), schedule delay 0
  (`src/features_disruption.py:74`), gap since the last departure 3,600 s
  (`src/features_advanced.py:99-101`). Symptoms: v47 against v46 moves one
  EDDM row by 17,996 s, one EHAM row by 9,064 s and one LEBL row by 5,898 s,
  at airports with 0, 3 and 0 tail rows in 2025 (Appendix A3, T5); v38 seed
  43 stopped at iteration 6 (`RECAP.md:81`); v47 serves 16 rows at 0 s. The
  only clip is at 0 (`src/predict_v30.py:217`).
- Arithmetic: a leaf of 291 rows with a 1,000-s spread on a feature in
  seconds has a design moment near 3 × 10⁸. A ridge of 1.0 or 0.0056 is
  negligible there and acts only on flags and small counts.
- Mechanism: a few extreme rows rotate a leaf's slope; every 2026 row routed
  there is extrapolated along it; constants are read as data.
- Cost: one non-LIRF row at 18,000 s on a 1,000-s taxi costs 838 MSE. The
  nine per-airport maximum moves sum to 1,412 MSE if v47 is the wrong side of
  each; the direction is not recorded. Grade C.

**M3. The base trains on rows it never serves.** V, medium.
- Evidence: `src/train_r_all_v47.py:55-59` applies no row exclusion;
  `src/predict_v30.py:197-203` overwrites every LIRF row. The training set
  holds 160,704 LIRF rows, 16.6 % of them fallback, and 12 of the 14 24-h
  rows (T6). `min_data_in_leaf` 291, `feature_fraction` 0.563.
- Mechanism: rows whose label the served value never reproduces shape splits
  and leaf slopes that served rows pass through. Twelfth-pass C3 lost the
  tail class by removing real holds; that was a different exclusion.
- Cost: unmeasured. A 25-minute paired run on the cached frame measures it.

**M4. The retune sampled 30 one-seed trials, hit five bounds, searched three
dead dimensions and pinned the leaf type.** V, medium.
- Evidence: `src/tune_lgbm_v43.py:74` creates an unseeded study; `:45` pins
  `linear_tree`; no categorical parameter is searched. Appendix A6: five of
  nine parameters sit at or next to a bound.
- Arithmetic: with L2 loss each row has hessian 1, so a leaf of 291 rows has
  a hessian sum of at least 291. `lambda_l2` at most 10 moves a leaf value by
  at most 3.4 %. `lambda_l1` at most 10 and `min_gain_to_split` at most 3 are
  negligible against gradient sums in seconds.
- Cost: 0 to 1,000 MSE, grade D.

**M5. The ensemble has no diversity.** V, medium.
- Evidence: members differ by seed only (`src/train_r_all_v47.py:64-66`) with
  `bagging_fraction` 0.977. The one second-class test (v46const) used 220
  leaves, raw `y`, the linear-leaf frame and a fixed 0.7/0.3 weight
  (`models/lgbm_r_all_v46const.holdout.json`). Fourteenth-pass S3 closed
  stacking without a number. No monotone constraint is set.
- Cost: 300 to 1,200 MSE, grade D.

**M6. Only the base was refit on 12 months.** V, medium.
- Evidence: `src/predict_v47.py:5` keeps every other component. `R_norm`
  trains on months 2 to 6 and 8 to 10 (`src/train_r_norm_lirf_v41.py:40-41`,
  `src/train_lirf_regime.py:37`); the gate and calibrator use the same months
  (`src/train_lirf_regime_v23.py:162-164, 197-198`); the encoders come from
  v26 (`src/train_r_all_v47.py:74-75`); the route medians use the fit months
  (`src/build_plan_taxi_res.py:21`). LIRF holds 21 % of the v41 hold-out MSE.
- Cost: about 400 MSE (0.7 s) by pro-rating the v47 base gain to LIRF's
  share, grade D.

**M7. Calendar features use UTC on a single training year.** R, low.
- Evidence: `hour`, `dow` and `month` come from UTC timestamps
  (`src/predict_v30.py:92-94`). Local time shifts by 1 h between January and
  July at nine airports and not at LTFM.
- Mechanism: bank structure moves by an hour between the two scoring months;
  month cells on one year can memorise 2025 events as seasonality.
- Cost: unmeasured, grade D.

### 4.5 LIRF head and tail rules

**T1. Step A reads the mixture as its normal term.** V, medium.
- Evidence: `src/predict_v30.py:203` writes the mixture into the prediction
  vector; `:207` passes it to `apply_stepA_v22`; `src/predict_v23.py:94`
  uses it as the normal term. `src/build_lirf_band_table_v30.py:62-65` stores
  `mean_norm` (960 to 1,434 s) and no serving code reads it. Eleventh-pass F7
  recorded the defect; nothing fixed it.
- Mechanism: the effective schedule weight is `p_band + p_norm × p_gate`,
  larger than either input. The combination is not a probability model.
- Arithmetic: band 14,400 to 16,000 s (n 27, `p_fb` 0.704, `p_norm` 0.296,
  `mean_norm` 960), `sd` 15,000, gate 0.8, `R_norm` 1,000: served value
  14,171 s against a hedge of 10,844 s; expected squared error 52.1 × 10⁶
  against 41.1 × 10⁶ s², 32 MSE per row.
- Cost: up to about 800 MSE on the 43 cell rows of 2026 if the gate sits
  near 0.8 there; the gate value on those rows is not recorded. Grade C.

**T2. The band table serves raw frequencies from 1 to 27 rows, including
probabilities of exactly 0 and 1.** V, medium.
- Evidence: Appendix A2. `p_fb` = 1 at 25,000 to 40,000 s (n 18), 40,000 to
  50,000 s (n 16) and above 100,000 s (n 1); `p_24h` = 1 at 70,000 to
  100,000 s (n 4); no prior (`src/build_lirf_band_table_v30.py:57-58`); the
  table is fitted on 12 months (L2). The 2026 null-record row at `sd` =
  111,654 is served at 111,654 (T16).
- Arithmetic: the five null-record 2025 rows above 70,000 s hold four 24-h
  rows and one fallback row. Pooled, they give `p_24h` 0.8 and a served
  value of 92,371 s: expected cost 270 MSE against 1,348 MSE for the n = 1
  band.
- Cost: about 1,080 MSE expected on that row, grade C; the variance is tens
  of thousands of MSE per row either way.

**T3. The ITY340 constants rest on one row that cannot separate the two
hypotheses.** V, low.
- Evidence: `src/predict_v30.py:36-39, 211-215`. The one 2025 LIRF row with a
  flight record and `sd > 70,000` has `sd` 87,001 and `y` 87,002 (T16),
  where the 24-h and fallback readings coincide. v39 lost 20,500 to 24,100
  MSE on one row by removing the rule.
- Cost: at most about 100 MSE between the two readings on the 2026 row. The
  hedge direction is right.

**T4. The mixture applies `p_fb × sd` without a bound on record-bearing
high-sd rows, where the calibrator has least support.** R, medium.
- Evidence: `src/predict_v30.py:200-203`. LIRF rows with `sd > 14,400`: 0.51 %
  of the 2026 LIRF rows, about 137 rows (T10); 43 of them are null-record and go to Step A. The
  calibrator is fitted on two months (L3).
- Cost: a systematic 0.1 error on about 90 record-bearing rows near
  `sd` = 30,000 s is up to about 2,200 MSE; sign unmeasured.

**T5. The six non-LIRF airports with a fallback spike have no regime head;
the closure rests on a cross-recipe comparison.** V, medium.
- Evidence: fallback share 3.2 to 5.8 % at EDDM, EGLL, LEBL, LEMD, LFPG and
  LTFM (T11). Residual served fallback MSE 1,726 (Appendix A7). Fourteenth-pass
  4.3 closed the lever on the v39 heads (1,850), which ran on a different
  base, encoder set and mask.
- Cost: ceiling 1,726 MSE (2.9 s), grade C.

### 4.6 Constants, definitions and serving

**C1. At least 34 hand constants shape the served prediction and none has a
recorded sweep.** V, medium.
- Evidence: Appendix A5; twelfth-pass M7. Two constants priced later moved
  the score: removing `R_NORM_CLIP` -187 MSE; exclusive band classes -1.28 s
  live.

**C2. `plan_taxi_res` does not measure the planned taxi.** V, low.
- Evidence: `src/build_plan_taxi_res.py:33-39` computes `MVT - ARVT_1` minus
  the median per `ADEP_ADES` and clips at ±3,600 s. `src/features_plan.py:9,
  20` defines a different quantity, `ARVT_1 - EOBT_1` per airport pair and
  type. Fourteenth-pass L9 describes "the planned taxi".
- Mechanism: the served column is the take-off delay against the flight plan
  minus a planned-block anomaly. The clip cuts a mixed quantity. The measured
  gain stands; the documented mechanism does not.

**C3. Weather joins at the end of the taxi, and the de-icing gate needs
precipitation.** V, medium.
- Evidence: `src/features_weather.py:113-116` joins backward from take-off
  within 45 min; `:138` sets the de-icing gate only with precipitation below
  3 °C. L7 (a join at EOBT_1) moved the served classes by -367 MSE on the old
  recipe. Strict monitor failures on the dewpoint spread in January 2026 at
  LEMD and LSZH.
- Mechanism: de-icing and low-visibility procedures act at pushback. Frost
  and freezing fog trigger de-icing without precipitation.
- Cost: 100 to 400 MSE; grade B for the EOBT join, D for the trigger.

**C4. Post-processing clips members before the mean and fills missing rows
with a global median.** V, low.
- Evidence: `src/predict_v30.py:163` clips each member at 0; `:217` clips at
  0 again; `:232-234` fills NaN with the median over all airports; no upper
  bound. v47 serves 16 rows at 0 s.
- Cost: about 40 MSE for the zero rows; the fill does not fire today (T7).

### 4.7 Reproducibility and code

**R1. v47 cannot be rebuilt from the repository.** V, medium.
- Evidence: no requirements or lock file; `models/**/*.txt` and `*.pkl` are
  gitignored, including feature lists and encoders; `REPRODUCE.md:3` stops at
  v33; no METAR fetch script (`REPRODUCE.md:44`); unseeded Optuna study
  (`src/tune_lgbm_v43.py:74`); `num_threads` -1 with linear leaves and no
  deterministic mode (`src/train_r_all_v47.py:66`); the LIRF feature list
  depends on which Eurocontrol sheets were downloaded
  (`src/features_eurocontrol.py:49-55`).
- Cost: eligibility risk; no number from v40 to v47 can be re-derived.

**R2. v44 to v47 train on the v36 cache with an id side file that no script
writes.** V, medium.
- Evidence: `src/train_r_all_v47.py:22, 36-43` imports `CACHE` and `ID_MAP`
  from `src/train_r_all_v40.py:23-25`; only readers reference
  `v36_tune_cache.ids.parquet`; `src/build_h1_frame_cache.py:4-5` records 169
  rows with a null id and null extra columns; the tuner used the repaired H1
  frame (`src/tune_lgbm_v43.py:18`); the id merges carry no one-to-one
  validation.
- Cost: under 1 MSE from the 169 rows; the tuned parameters and the trained
  members come from two frames.

**R3. Serving needs the training parquets to rebuild category lists from an
unordered set.** V, low.
- Evidence: `src/predict_v30.py:42-51, 110, 152-154`. Correct codes rely on
  LightGBM remapping pandas categories to the list stored in the booster.
  `src/eval_v33_holdout.py:70-72` builds yet another list from hold-out rows.
- Cost: zero today; catastrophic if the remap ever fails.

**R4. The serve path imports 30 of 184 scripts and repeats constants across
them.** R, low.
- Evidence: `TEMPO_COLS` appears in `src/train_r_all_v40.py:26-29` and
  `src/build_h1_frame_cache.py:25-28`; `src/predict_v30.py:121-122` calls
  `add_congestion` and then `add_congestion_v2`, which overwrites all ten
  columns; encoder identity between the cache build and the served pickle is
  assumed, not asserted (`src/tune_lgbm_v36.py:95-97`).
- Cost: zero today; maintenance and reproduction risk.

**R5. The output merge has no row-count or id-uniqueness guard.** R, low.
- Evidence: `src/predict_v30.py:139, 229-235`; `src/features_weather.py:107-118`
  reorders rows and leaves a duplicated index; the check runs after the fact
  (`src/check_submission_2026.py:32-36`).
- Cost: zero today (`used_pairs` 344,841); latent.

**R6. Two timestamp pipelines compute `mvt_eobt1` and are never asserted
equal on all rows.** R, low.
- Evidence: `src/train_lgbm_v21.py:70-73` against `src_v2/frame.py:35-40, 74`;
  the H1 content key compares them rounded, on 169 rows only.
- Cost: probably zero; unverified.

### 4.8 Strategic gaps

**S1. The feature set ignores the structure of the runway queue.** R,
medium.
- Evidence: the 117 served columns hold wake category for the row itself
  only, direction as the destination code only, and runway state as
  diversity, gap and bank counts. No column counts heavy departures ahead,
  same-direction departures, runway crossings on the taxi route, remote
  runways, minutes since the runways in use changed, or local hour.
- Cost: 300 to 1,500 MSE, grade D. The tempo family priced 9 to 11 s in a
  quick test and landed 2.55 s live.

**S2. No multi-day drift signal.** V, low.
- Evidence: `RECAP.md:201` measured per-airport arrival taxi-in drift of
  +129 s at EHAM and -41 s at EGLL between 2025 and 2026. L3.e tested only a
  30-minute residual (+102 served, inside noise).
- Cost: unmeasured, grade D.

**S3. No day-level signal of the reporting artefact.** V, low.
- Evidence: the gate's rate keys are static 2025 maps
  (`src/train_lirf_regime_v23.py:41-42`). The arrival-side artefact rate is
  observable per airport in 2026 (T9). Clustering by day is unmeasured.
- Cost: unmeasured; part of the 5,934 LIRF and 1,726 non-LIRF fallback MSE.

### 4.9 Policy

**X1. The exclusion of `AOBT_3_flt` and `LOBT_flt` sets a ceiling that
needs a written ruling.** V, decision.
- Evidence: `AOBT_3_flt` is present on 98.9 % of ranking rows and lies
  within 60 s of the off-block time on 17 to 24 % of 2025 rows at nine
  airports and 7.9 % at LTFM (T3); `LOBT_flt` equals it on 4.3 %. Brief
  section 7.2 warns against fields that reconstruct the label. The
  organiser wrote "Any provided value is fair to use as a proxy (or as
  itself) if it helps" (`RECAP.md`, Discord 2026-09-08) and later called
  reverse engineering of the ranking something "we despise" (2026-09-11).
- Arithmetic: if 17 to 24 % of rows became near-exact and those rows carried
  the average error, the live MSE falls to 66,850 to 73,000, or 258 to 270 s.
  The top score, 263.46 s, sits inside that range. This does not prove what
  other teams did.
- Status: **MX1 resolved 2026-09-18.** The organiser confirmed no formal
  rule bans reading `AOBT_3_flt`/`MVT_TIME_UTC_mvt` or trajectory-derived
  off-block times: "the model is for post-ops, not for tactical use" and
  "there are no such restrictions ... Practically speaking it won't be
  possible" (`RECAP.md`, Discord 2026-09-18). The ruling relies on poor
  ADS-B surface coverage at most airports to make the exploit impractical,
  not on a prohibition. The exclusion stays a team decision regardless.
  No measure in section 5 reads these fields. See `README.md` Ethics
  section for the recorded verbatim answer.

## 5. Measures for the new model

Each measure names the findings it fixes and its acceptance test. Section 6
orders them. No measure ships without its test.

### 5.1 Protocol, before any model change

- **MP1. Replace the single split with month-blocked cross-validation.**
  Six folds hold out the month pairs (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)
  and (6, 12) and train on the other ten months. Each fold early-stops on the
  two training months next to its hold-out pair. Report every price as the
  mean and the standard error of the per-fold paired delta, and report the
  (1, 7) fold alone. Fixes P1, P2, P4. Test: the fold standard error of an
  identical-recipe retrain is recorded before any lever runs.
- **MP2. Decide on intervals, not thresholds.** The price is the full MSE on
  the rows the component serves; the class table per airport is a
  diagnostic. Accept when the 90 % interval of the fold delta lies below 0.
  Reject when it lies above 0. Otherwise record "inconclusive", which never
  closes a lever. Retire the 200, 300, 400 and 500 MSE gates. Fixes P1, P7.
- **MP3. Retrain the control in the same run.** Fix the thread count, set
  deterministic training for pricing runs, and record the delta of an
  identical retrain once per recipe. Fixes P8.
- **MP4. Fit every artefact inside the fold.** Encoders, rate maps, base
  rates, the tail model, the calibrator, route medians and normal means are
  fitted on the fold's training months. Keep a fold set for evaluation and a
  full-year set for the ship. Fixes L1, L2, L3.
- **MP5. Select each component on the rows it serves.** Tune and early-stop
  the base on non-LIRF rows without the 24-h class; the LIRF regressor on
  LIRF genuine rows; the gate on log loss and calibration error. Fixes P3.
- **MP6. Build the arrival mirror.** Same frame builder and recipe, target
  arrival taxi-in on validated arrivals, trained on 2025, scored on 2026 per
  airport and month. Report the arrival-side delta next to every fold price.
  A lever that helps the 2025 folds and hurts 2026 arrivals does not ship.
  Fixes P11.
- **MP7. Gate every file on label-free checks.** Per airport: maximum
  prediction; counts above 3,600 s and 7,200 s against the 2025 support; a
  list of every row moved by more than 3,000 s with airport, `sd`,
  `mvt_eobt1` and record status; LIRF `p_fb` and `R_norm` quantiles by month;
  value quantiles of every served column by airport and month against the
  same 2025 month. Block the upload on an unexplained row or a strict drift
  failure. Fixes P10, D4, D5.
- **MP8. Make the coverage monitor strict for every column of every served
  component.** Compare against the same 2025 month and add a relative rule
  (share halved) next to the 20-point rule. Fixes D1, D5.
- **MP9. Keep upload discipline.** Fetch the v44 and v45 scores now. Read
  every score before building on it. Write the price before the upload.
  Report a selection-adjusted best next to the raw best. Fixes P6, P12.
- **MP10. Define the label classes in one module.** Add a "low" class for
  `0 < y < 30`. Assert that the class sums equal the FULL MSE in every
  report. Fixes P13.

### 5.2 Data layer

- **MD1. Define the fallback class once per airport from the residual
  grid.** Tabulate `abs(y - sd)` per second from 0 to 120 s per airport,
  include the 60-s grid point where it is a spike, and use that definition
  in reports, gates, regressor masks, tail tables and route medians. Fixes
  D2. Test: the histogram table sits in the next pass; a fold run with the
  new mask resolves the L5 contradiction.
- **MD2. Remove drifted columns from every component.** A column whose
  coverage in either 2026 month fails the MP8 rule against the same 2025
  month leaves every model. First action: the 38 `ec_*` and 18 OPDI columns
  leave the LIRF regressor and gate. Fixes D1. Test: MP8 passes on the head;
  the July 2026 LIRF `p_fb` and `R_norm` quantiles match the January 2026
  shape.
- **MD3. Encode absent context as NaN on every column.** Wind, gust,
  ceiling, taxi-in means, schedule-delay means and the gap since the last
  departure. Counts stay 0 only when the window holds observed movements.
  Fixes M2.
- **MD4. Compute target encoders out of fold.** Leave-one-month-out on clean
  rows; median, interquartile range and count per key; full-year map at
  serving; no standard-deviation column. Fixes L1.
- **MD5. Scope categoricals to the airport and add a stand fallback.** Use
  airport|stand and airport|runway levels, an apron or stand-prefix
  categorical with its own encoders, and nearest-known-stand geometry for
  unseen stands. Persist every category list as JSON at training. Fixes D6,
  R3.
- **MD6. Serve the drift flags and check the identifiers.** Add
  `eobt1_equals_sched` and the airport-hour shares of null records and of
  EOBT_1 = schedule. Compare LFPG arrival runway identifiers of 2025 and
  2026 and map them if they changed. Fixes D3, D4, D5.
- **MD7. Keep one arrival taxi-in meter on validated arrivals.** Drop
  arrivals with in-block on the schedule grid and outside 30 to 7,200 s; NaN
  when the window is empty. Fixes D7.
- **MD8. Use one neighbour pool for every tempo statistic.** Every departure
  row at the airport, label-agnostic, in training and ranking. Assert that
  the quartile count equals the tempo count. Fixes D8.
- **MD9. Replace `ades_arr_atfm_delay_today`** by the previous day's value or
  drop it, and price the change on folds. Fixes L5.
- **MD10. Give month starts their context.** Prepend the 31 December 2025
  movements of the training file for January; flag rows in the first 12 h of
  1 July. Fixes L6.

### 5.3 Target and base model

- **MB1. Train the base on the anchored offset.** Target `y - anchor`, with
  anchor `mvt_eobt1` where EOBT_1 exists, else `sd`. Prediction: anchor plus
  the predicted offset, bounded to the 2025 per-airport 0.1 to 99.9 % offset
  range. Keep `sd - anchor`, `eobt1_sched` and `eobt1_iobt` as features.
  Compare with the raw-`y` linear-leaf base on the folds. Fixes M1.
- **MB2. Condition any linear-leaf member.** Standardise numeric inputs on
  training statistics, apply MD3, search `linear_lambda` on the standard
  scale, and exclude `y > 80,000` rows from its fit. Fixes M2.
- **MB3. Train the base on the rows it serves.** Exclude LIRF rows, Step A
  cell rows and `y > 80,000` rows; keep `7,200 < y <= 80,000` rows. Fixes M3.
  Test: the served tail class per airport does not worsen on the folds.
- **MB4. Drop the numeric month.** Carry season with cyclic day of year,
  local hour and weekday with daylight saving time, public and school
  holidays, and weather. Fixes P2, M7.
- **MB5. Retune each model class in its own study.** Seeded sampler, at
  least 100 trials with pruning, fold-averaged served objective, bounds
  widened where v43 hit an edge (`num_leaves` to 1,024, `min_data_in_leaf` to
  2,000), dead dimensions removed, categorical parameters added
  (`cat_smooth`, `min_data_per_group`, `max_cat_threshold`). Fixes M4.
- **MB6. Choose one round count per recipe from the fold curves.** The final
  refit scales it by the measured row ratio, or the fold models ship as the
  ensemble. Record the ratio in the model metadata. Fixes P4, P5.
- **MB7. Replace seed averaging with model diversity.** An anchored
  constant-leaf LightGBM, a conditioned linear-leaf LightGBM and a CatBoost
  model with ordered target statistics on the high-cardinality columns;
  bagging below 0.8; non-negative blend weights fitted on out-of-fold
  predictions only; monotone constraints on queue counts, tempo medians and
  path length where the folds allow. Fixes M5, P9.
- **MB8. Refit every component and table on 12 months for the final ship.**
  The LIRF regressor, gate, calibrator, encoders, route medians and tail
  model included. Fixes M6.

### 5.4 Regime heads and tail rules

- **MH1. Rebuild the LIRF head.** MD2 columns removed; gate early-stopped on
  one blocked month pair and calibrated on another, or calibrated on
  out-of-fold predictions; calibration error reported in `sd` bins above
  14,400 s; seeded; one rate-map construction for training and serving.
  Fixes D1, L3, T4.
- **MH2. Extend the regime head to EDDM, EGLL, LEBL, LEMD, LFPG and LTFM**
  with the MD1 definition. Serve the mixture at an airport only where the
  fold class table beats the base. Fixes T5.
- **MH3. Measure the day-level clustering of the artefact before building
  on it.** If departures and arrivals show the artefact clustered by airport
  day, handler or operator in 2025, add the same-day arrival share with
  in-block on schedule and the same-day departure shares of null record and
  EOBT_1 = schedule to every gate. Fixes S3.
- **MH4. Replace the band table and the ITY340 constants with one smoothed
  three-class model.** Classes fallback, 24-h and normal on `sd`, for LIRF
  null-record rows with `sd > 14,400` and for every LIRF row with
  `sd > 70,000`. Monotone in `sd`, Dirichlet prior of 0.5 per class, bands
  merged to at least 10 rows, no probability of exactly 0 or 1, fitted in
  fold for evaluation. Fixes T2, T3, L2. Test: on 2025 rows it serves the
  `sd > 70,000` class within the v33 error; the served value and the cost
  under each class of the `sd` = 111,654 row are recorded before upload.
- **MH5. Combine the tail model and the gate as one posterior.** The normal
  term reads the genuine-row regressor or the band's normal mean, never the
  mixture. Fixes T1.
- **MH6. Keep the rules the evidence supports.** At LIRF every row with
  `sd > 70,000` keeps a 24-h and fallback hedge, with or without a record
  (v39). Outside LIRF `sd > 70,000` is a normal taxi (251 rows in 2025). No
  hedge for rows with no observable, such as the two LFPG rows.
- **MH7. Re-test the hold rule after MB1.** Raise the prediction toward
  `mvt_eobt1` when `mvt_eobt1 > 7,200` and a record exists; drop the rule if
  the anchored target already tracks the holds on the folds.

### 5.5 Features

- **MF1. Join weather at pushback.** METAR at EOBT_1 and at EOBT_1 minus
  20 min next to the take-off join. De-icing trigger: temperature at or
  below 3 °C with precipitation, freezing codes, snow, or a dewpoint spread
  at or below 2 °C. Price on folds. Fixes C3.
- **MF2. Add runway-queue structure, backward only.** Count and share of
  heavy and super departures on the same runway in the previous 5 and 10
  min; departures in the previous 10 min with a destination bearing within
  30° of the row's; active-runway crossings on the OSM route given the
  runways used in the previous 30 min; a remote-runway flag per airport and
  runway; minutes since the set of runways in use changed. Fixes S1.
- **MF3. Add local hour with daylight saving time and night-restriction
  windows per airport.** Fixes M7.
- **MF4. Add a multi-day drift feature.** Per airport, the 1-day and 7-day
  rolling median of validated arrival taxi-in minus the same 2025 month
  median. Fixes S2.
- **MF5. Rename `plan_taxi_res` for what it measures and add a true
  planned-taxi proxy.** `ARVT_1 - EOBT_1` minus the median per airport pair
  and type on fit months; price both. Fixes C2.
- **MF6. Replace the two forward-window columns by backward mirrors** unless
  MX1 rules them admissible. Fixes L4.

### 5.6 Serving and post-processing

- **MS1. Bound predictions per airport after the ensemble.** At EDDF, EDDM,
  LEBL and LEMD, which have no 2025 row above 7,200 s outside fallback, cap
  at the 2025 airport maximum of that class plus a margin. At EGLL, EHAM,
  LFPG, LSZH and LTFM, allow values above 7,200 s only with a flight record
  and `mvt_eobt1` above 5,400 s. LIRF follows MH4. Fixes M2.
- **MS2. Clip the ensemble mean, not the members**, and floor at the
  per-airport 0.1 % quantile of 2025 clean labels. Fixes C4.
- **MS3. Gate member disagreement.** Where members differ by more than
  3,600 s outside LIRF, serve the member median and list the row in the MP7
  report.
- **MS4. Fail on a missing prediction.** Fill with the airport median only
  after a logged failure; assert row count and id set equal to the template.
  Fixes C4, R5.

### 5.7 Code and reproducibility

- **MC1. Build one package.** One config that holds every constant with its
  provenance (Appendix A5), one frame builder with one timestamp
  normalisation, one feature list per component written next to its model.
  Archive the v36 cache, the id side file and the old scripts. Fixes R2, R4,
  R6, C1.
- **MC2. Sweep the served constants on folds in one batch.** Windows,
  tolerances, clips, defaults, minimum counts, smoothing weights. Fixes C1.
- **MC3. Store artefacts as parquet or JSON with hashes.** Encoders, category
  lists, the tail model and the calibrator knots; no pickle. Publish the
  served boosters as a release asset with hashes. Fixes R1, R3.
- **MC4. Pin every library version and make final training deterministic.**
  Lock file, fixed thread count, deterministic mode, seeded Optuna and
  boosters. Fixes R1.
- **MC5. Validate every id-keyed merge as one-to-one and assert row counts.**
  Assert `mvt_eobt1` agreement between any two pipelines that compute it.
  Fixes R2, R5, R6.
- **MC6. Complete the reproduction path.** Write the METAR fetch script,
  rewrite `REPRODUCE.md` for the new model end to end, and keep a parity test
  that rebuilds the reference file to 0.0 s before each upload. Fixes R1.
- **MC7. Reconcile `README.md` and `RECAP.md`.** The v44 and v45 scores, the
  1,864 against 1,922 MSE sums, the "ambiguity prediction" sentence, and the
  closures that section 8 reopens.

### 5.8 Policy

- **MX1. Ask the organiser in writing** whether `AOBT_3_flt`, `LOBT_flt` and
  forward windows over other flights' movements are admissible. Record the
  answer verbatim in `README.md`. Until then no measure reads them. Fixes X1,
  L4.
  **Resolved 2026-09-18** — answer recorded verbatim in `README.md` Ethics
  section and in `RECAP.md` Discord confirmations. No formal rule bans it;
  the exclusion stays in force as a team decision. No measure reads these
  fields.

## 6. Build order, gates and stop rule

The window closes on 2026-10-11, 23:59:59 CET (`docs/WINNING_PLAN.md`
3.1 item 10; the earlier 2026-10-31 date was wrong). Each phase ends in fold prices, an arrival
mirror delta and the MP7 gates.

1. **Phase 0, harness.** MP1 to MP10, MC1, MC5. Exit when the fold harness
   scores the v47 recipe, the identical-retrain noise is recorded and the
   arrival mirror is scored. No upload.
2. **Phase 1, defects with no expected loss.** MD2 with MH1, MS1 to MS4, MD3,
   MD4, MB4, MB3. Price each on folds. Ship the accepted set as one upload.
3. **Phase 2, target and base.** MB1, MB2, MB5, MB6. Choose target and leaf
   type on folds, retune, upload.
4. **Phase 3, heads and tail.** MD1, MH2 to MH5, MH7. Upload.
5. **Phase 4, features.** MF1 to MF6, MD5 to MD10. One fold price per group;
   ship the accepted union as one upload.
6. **Phase 5, diversity and final ship.** MB7, MB8, MC2 to MC7. Refit every
   component on 12 months, pass MP7, upload.

Rules:

- One pre-registered price per upload. Never build on an unscored upload.
- Stop rule: if phases 1 to 3 accept less than 2,000 MSE of fold gain in
  total, skip phase 4, run phase 5, and spend the remaining time on the
  documentation and the paper.

## 7. Expected gains

Planning ranges on the scoring scale. The prices do not add: section 2.3
shows that stacked prices lost most of their value. Only fold prices from
phase 0 onward decide.

| measure | low | high | grade | basis |
|---|---|---|---|---|
| MD2, MH1 drifted columns out of the LIRF head | 0 | 2,000 | C | v26 live gain pro-rated, about 700 |
| MS1 to MS3 bounds and disagreement gate | 0 | 1,500 | C | 838 per extreme row, direction unknown |
| MH4, MH5 smoothed tail model, correct normal term | 0 | 1,500 | C | 1,080 expected on one row, up to 800 on the cell |
| MB1, MB2 anchored target, conditioned leaves | 500 | 3,000 | D | |
| MB3 base on served rows | 0 | 1,000 | D | |
| MB4, MF3 calendar | 0 | 800 | D | |
| MD4 out-of-fold encoders | 0 | 500 | D | |
| MB8 12-month refit of every component | 200 | 700 | D | v47 gain pro-rated to LIRF, about 400 |
| MH2, MH3 fallback heads at six airports | 300 | 1,700 | C | ceiling 1,726 |
| MF1 weather at pushback | 100 | 400 | B | -367 on the old recipe |
| MF2 runway-queue structure | 300 | 1,500 | D | |
| MB7 model diversity | 300 | 1,200 | D | |
| MB5 retune on the served objective | 0 | 1,000 | D | |
| MD5 stand fallback | 100 | 300 | C | |
| range | 1,800 | 17,100 | | 293.5 s to 266.2 s if fully additive |

The gap to 290 s is 3,854 MSE; the gap to the top is 18,543 MSE.

## 8. What stays, what stays closed, what reopens

### 8.1 Keep

| item | evidence |
|---|---|
| unfiltered labels and unclipped `sd` | v13: -130 s live |
| the LIRF regime structure (mixture of schedule and regressor) | v22, v23: -12.6 s live |
| the LIRF hedge at `sd > 70,000` with or without a record | v39: +50 s live without it |
| the neighbour tempo family | v40, v41: -2.55 s live |
| removal of families with dead 2026 coverage | v26: -13.1 s live; extend to the head |
| final refit on 12 months | v47: -2.57 s live; extend to every component |
| parity test, coverage monitor, label-free check | extend per MP7 and MP8 |
| one change per upload with a written price | v40, v41 landed within 25 to 341 MSE of their price |
| no hedge for rows with no observable | ninth pass 6.1 |
| exclusion of `AOBT_3_flt` and `LOBT_flt` until a ruling | X1, MX1 |

### 8.2 Stays closed

| item | evidence |
|---|---|
| hard swaps between regimes | v15: 625 s live |
| imputation of dead columns with medians | v25: +58 s live |
| runway-configuration string categorical | +7.6 s hold-out |
| Step A extended to `sd > 3,600` | +9.8 s hold-out |
| clean-only base that drops real holds | v39, T17: tail class +1,869 |

### 8.3 Reopens under the fold protocol

| item | why the closure does not hold |
|---|---|
| gate refits | decided on live deltas below the member-draw noise; calibration defect L3 unfixed |
| fallback heads at six airports | closed on the cross-recipe v39 comparison |
| Step A smoothing | priced on a table fitted on the scored rows (L2) |
| second model class and stacking | v46const misconfigured; stacking closed without a number |
| weather at EOBT_1 (L7) | all served classes improved; missed a clean-only gate by 7 MSE |
| calendar (L8), arrival residual (L3.e) | inside the null spread on one split and an old recipe |
| anchored or transformed target | the only test ran on the censored pre-v21 pipeline |

## 9. Corrections to earlier passes

| earlier statement | this pass |
|---|---|
| fourteenth 2.1: member-draw lottery 268 MSE | a mean penalty on the hold-out, not a spread; 2,107 on 2026 by the same arithmetic (P9) |
| fourteenth 2.1: prices from the paired harness land | true for single changes; v44 to v46 landed 6 %; v44 and v45 were never scored (P6) |
| fourteenth 2.2: LIRF 24-h 1,273, hedge MSE-optimal per band | in-sample and unsmoothed (L2, T2) |
| fourteenth fact 8: structure stable except EHAM and EDDM stands | null-record share doubled at six airport-months; EOBT_1 = schedule share moved 13 to 26 points in July at LTFM, LFPG and EGLL; LFPG `mvt_eobt1` median -298 s (D3, D4) |
| fourteenth fact 9: Eurocontrol pre-departure family 0 % in July 2026 | the LIRF head still reads it (D1) |
| fourteenth L2: scale iterations by 1/0.88 | the row ratio is 1.361 (P5) |
| fourteenth L9: the plan residual isolates the planned taxi | the served column is `MVT - ARVT_1` minus a route median (C2) |
| fourteenth 4.3: gate refit, any form, closed | reopens (section 8.3) |
| fourteenth 4.2 S3: stacking closed | an out-of-fold blend has no leak path (MB7) |
| RECAP L1: retune passed the 500 MSE gate | -527 of -662 at LSZH, a rebound from the previous arm (P3) |
| RECAP H4: 4 strict cells, all data drift, not defects | drift is a model risk; the monitor skips the head's columns (D1, D4, P10) |
| twelfth D1, A1: fallback is a 5-s spike | v43 and T8 show a 60-s grid point at LIRF (D2) |
| twelfth T14: arrival ATFM column 41 to 63 % non-null | 4 to 10 % per departure row (L5) |

## Appendix

### A1. Coverage drift, 2025 against 2026, from `models/coverage_monitor.json`

Shares in percent. "Non-zero" is the share of rows with a non-zero value.

| column | measure | airport, period | 2025 | 2026 |
|---|---|---|---|---|
| 7 same-day `ec_*` columns in the LIRF head | non-null | LIRF, July | 100.0 | 0.0 |
| 7 lag-1 `ec_*` columns in the LIRF head | non-null | LIRF, July | 99.8 | 3.2 |
| `flt_null` | non-zero | EHAM, January | 1.6 | 5.0 |
| `flt_null` | non-zero | LSZH, January / July | 1.5 | 3.0 / 2.7 |
| `flt_null` | non-zero | LIRF, July | 0.9 | 1.7 |
| `flt_null` | non-zero | LEBL, July | 1.0 | 2.0 |
| `flt_null` | non-zero | LEMD, July | 0.4 | 0.9 |
| `eobt1_sched` | non-zero | LTFM, July | 40.3 | 19.6 |
| `eobt1_sched` | non-zero | LFPG, July | 35.7 | 61.6 |
| `eobt1_sched` | non-zero | EGLL, July | 46.5 | 59.3 |
| `arr_same_rwy_prev_15m` | non-zero | LFPG, January / July | 19.9 | 3.9 / 3.8 |
| `arr_same_rwy_next_10m` | non-zero | LFPG, January / July | 17.6 | 2.9 / 2.6 |
| `openc_op_apt_stand_median` | non-null | EDDF, July | 92.0 | 74.0 |
| `openc_op_apt_stand_median` | non-null | EDDM, July | 90.2 | 80.4 |
| `openc_op_apt_stand_median` | non-null | LSZH, January | 88.3 | 78.8 |
| `ground_time` | non-null | EDDF / EDDM / LEBL, January | 94.5 / 92.4 / 95.3 | 90.4 / 87.5 / 91.0 |
| `stand_gap` | non-null | LIRF, January | 4.0 | 1.8 |
| `ades_arr_atfm_delay_today` | non-null | all airports, July | 4.2 to 10.0 | 4.0 to 9.5 |

### A2. The deployed band table, `models/lirf_band_table_v30.json`

| band, s | n | `p_fb` | `p_24h` | `p_norm` | `mean_norm`, s |
|---|---|---|---|---|---|
| 14,400 to 16,000 | 27 | 0.704 | 0 | 0.296 | 960 |
| 16,000 to 18,000 | 14 | 0.786 | 0 | 0.214 | 1,434 |
| 18,000 to 20,000 | 16 | 0.875 | 0 | 0.125 | 1,232 |
| 20,000 to 22,000 | 8 | 0.875 | 0 | 0.125 | 1,377 |
| 22,000 to 25,000 | 8 | 0.875 | 0 | 0.125 | 1,078 |
| 25,000 to 40,000 | 18 | 1 | 0 | 0 | |
| 40,000 to 50,000 | 16 | 1 | 0 | 0 | |
| 50,000 to 60,000 | 9 | 0.667 | 0.333 | 0 | |
| 60,000 to 70,000 | 6 | 0.333 | 0.667 | 0 | |
| 70,000 to 100,000 | 4 | 0 | 1 | 0 | |
| above 100,000 | 1 | 1 | 0 | 0 | |

### A3. v47 against v46 on the 344,841 ranking rows, `submission/kind-mango_v47.check.json`

| airport | mean shift, s | mean absolute move, s | maximum move, s | 2025 tail rows (T5) |
|---|---|---|---|---|
| EDDF | -1.43 | 21.9 | 2,354 | 0 |
| EDDM | +4.10 | 20.4 | 17,996 | 0 |
| EGLL | -15.05 | 30.3 | 1,805 | 34 |
| EHAM | +4.36 | 26.9 | 9,064 | 3 |
| LEBL | +0.10 | 18.4 | 5,898 | 0 |
| LEMD | -3.96 | 18.9 | 1,566 | 0 |
| LFPG | -2.20 | 25.6 | 4,045 | 36 |
| LIRF | 0 | 0 | 0 | 40 |
| LSZH | +2.15 | 25.2 | 3,245 | 5 |
| LTFM | -8.46 | 30.6 | 3,258 | 21 |

Rows above 7,200 s: 110 (v46: 107). Rows at 0 s: 16 (v46: 17).

### A4. Near-null paired arms, served clean delta

| arm | delta, MSE | source |
|---|---|---|
| L3.b tempo quartiles of `mvt_iobt` | +16 | `RECAP.md` v45iobt |
| L3.c tempo by stand prefix | +77 | `RECAP.md` v45term |
| L3.d tempo by operator | +54 | `RECAP.md` v45op |
| L3.e arrival taxi-in residual | +102 | `models/lgbm_r_all_v48arr.holdout.json` |
| L8 calendar | -144 | `models/lgbm_r_all_v48cal.holdout.json` |
| L4 constant-leaf blend | +64 | `models/lgbm_r_all_v46const.holdout.json` |
| mean, standard deviation | +28, 89 | arithmetic |

### A5. Hand constants in the served path

`P24_ITY` 5/6; `ITY_SD_THRESHOLD` 70,000 s; `NORMAL_MEAN_LIRF` 1,150 s;
`GATE_SD` 14,400 s; 11 band edges; `FB_TOL` 60 s (head) and 5 s (reports);
`K_SMOOTH` 30; `MIN_COUNT` 20; OPDI lag 600 s; METAR tolerance 45 min;
turnaround tolerance 12 h; gap default and clip 3,600 s; ceiling default
25,000 ft; low visibility 5 km; very low visibility 1.5 km; low ceiling
500 ft; de-icing 3 °C; disruption clip -1,800 to 14,400 s; 6 hour bins;
stand gap window 0 to 1,500 s; tempo windows 30 and 60 min; order window
30 min; congestion windows 15, 30, 60 min and forward 10 min; plan residual
clip 3,600 s; `STOP_FRAC` 0.12; refit scale 1/0.88; 127 leaves for `R_norm`;
patience 100, 60 and 50; gates 200, 300, 400 and 500 MSE; blend 0.7/0.3;
clean window 30 to 7,200 s; 24-h threshold 80,000 s; label filter `y > 0`;
arrival validity 30 to 7,200 s; OPDI taxi validity 30 to 7,200 s.

### A6. Tuned parameters against their search bounds

`models/tune_lgbm_v43.tuned_params.json`, `src/tune_lgbm_v43.py:34-48`.

| parameter | value | bounds | at a bound |
|---|---|---|---|
| `num_leaves` | 436 | 64 to 440 | yes |
| `min_data_in_leaf` | 291 | 20 to 300 | yes |
| `min_gain_to_split` | 2.974 | 0 to 3 | yes, and inert |
| `lambda_l1` | 0.00107 | 0.001 to 10 | yes, and inert |
| `bagging_fraction` | 0.977 | 0.6 to 1.0 | yes |
| `lambda_l2` | 0.0274 | 0.001 to 10 | inert |
| `linear_lambda` | 0.0056 | 0.0001 to 10 | inert on features in seconds |
| `learning_rate` | 0.0183 | 0.01 to 0.08 | no |
| `feature_fraction` | 0.563 | 0.5 to 1.0 | no |

Trials: 30, one seed, unseeded sampler, stop RMSE 245.41 s on months 11 and
12 of every airport.

### A7. Served classes of the v46 base outside LIRF, hold-out scale

`models/lgbm_r_all_v46.holdout.json`, key `v46`.

| airport | clean | fallback | tail | 24-h |
|---|---|---|---|---|
| EDDF | 4,497 | 22 | 0 | 0 |
| EDDM | 2,881 | 64 | 0 | 0 |
| EGLL | 8,181 | 217 | 1,047 | 0 |
| EHAM | 4,374 | 20 | 460 | 0 |
| LEBL | 3,322 | 810 | 0 | 0 |
| LEMD | 3,243 | 124 | 0 | 0 |
| LFPG | 7,664 | 221 | 10,084 | 20,232 |
| LSZH | 2,414 | 27 | 451 | 0 |
| LTFM | 7,941 | 221 | 98 | 0 |
| total | 44,517 | 1,726 | 12,140 | 20,232 |

The same report scores the base on LIRF rows it does not serve: clean
16,836, fallback 8,172, tail 865, 24-h 12,426.
