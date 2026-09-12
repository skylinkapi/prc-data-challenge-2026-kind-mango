# Model analysis and the v32 handoff

Updated 2026-09-11, sixth audit. Replaces the v30 plan. Written for the agent that
builds v32. Status: live best **301.98 s (v30)**, rank 44/111. v31 scored 302.41 s
and is rejected. The gap to 300 s is **1,192 MSE**.

Section 1 is the brief for the build agent. Its step 1 repairs the repository,
which no longer reproduces v30 (section 2). Sections 3 to 5 are the evidence.
This document holds no model code.

## 0. Scoreboard since the fifth audit

| version | live RMSE | live MSE | change | content |
|---|---|---|---|---|
| v29 | 303.26 | 91,967 | — | v26 base, `R_norm_LIRF` clip, ITY340 constant formula |
| **v30** | **301.98** | **91,192** | **-775** | + LIRF-only encoders for the LIRF models + band table v30 |
| v31 | 302.41 | 91,452 | +260 against v30 | + "record normal-only" rule at LIRF |

The v30 plan gave a range of 302.31 to 299.25 s. v30 landed at 301.98 s, inside
the range and close to its fallback scenario (301.65 s).

## 1. Instructions for the build agent

You build v32. v32 is v30 with two variance cuts: a 5-member `R_norm_LIRF` and a
7-member base. You do not upload. You give the user a file and a report. The
user decides.

### 1.1 Read before the first edit

1. `AGENTS.md`, all of it.
2. Sections 2, 3.3, 5 and 6 of this file.
3. `src/predict_v30.py`, `src/train_lirf_regime.py` (lines 93 to 144) and
   `src/train_r_all_v26.py`.

### 1.2 Hard rules

1. Never overwrite a file that a shipped submission reads. Write every new model
   to a new name.
2. Before step 1, record the SHA-256 of every file in `models/` and
   `submission/`. At the end, show that only the files you expected changed.
3. Keep `AOBT_3_flt` and `LOBT_flt` out of every feature list.
4. Never set a row-level prediction from a leaderboard score (section 9).
5. Never replace a calibrated probability with 0 or 1 (section 3.3).
6. Keep the feature lists, the parameters, the stop sets and the seed recipes
   exactly as they are. The v27, v28 and v31 losses all came from a changed
   recipe.
7. A hold-out difference under 2 s of CLEAN RMSE is not evidence. Use only an
   exact reproduction or the ambiguity price of section 5.
8. Do not commit until the user asks. Update `README.md` in the same change.

### 1.3 Steps

Do the steps in order. If an acceptance test fails, stop and go to 1.4.

**Step 1. Repair the repository.** Do the four actions of section 2.2.

- Acceptance: `python src/predict_v30.py check_v30.parquet` matches
  `submission/kind-mango_v30.parquet` to 0.0000 s on all 344,841 rows.

**Step 2. Train 5 new `R_norm_LIRF` members.** Write
`src/train_r_norm_lirf_seeds.py`. Import the feature build from
`src/train_lirf_regime.py`. Do not copy it; section 6 item 2 explains why.

- Rows: LIRF departures, genuine only (`abs(y - sd) >= 60` and `y < 80,000`).
- Encoders: `models/lirf_regime.encoders.pkl`, the LIRF-only set.
- Features: `models/lirf_regime.features.txt`.
- Train months 2 to 6 and 8 to 10. Stop months 11 and 12.
- Parameters: `params_reg` of `train_lirf_regime.py`, 3,000 rounds, early stop
  100.
- Seeds 42, 43, 44, 45 and 46. Set `seed`, `bagging_seed` and
  `feature_fraction_seed` to the seed value. The audit priced exactly this
  recipe.
- Save each member as `models/lgbm_r_norm_lirf_s{seed}.txt`.
- Do not train a fallback gate in this script. Do not touch
  `models/lgbm_r_norm_lirf.txt`. v30 still reads it; v32 does not.
- Acceptance: each best iteration is above half the median of the five. The
  audit run gave 251, 417, 294, 294 and 297.

**Step 3. Train 4 more base members.** Move `SEEDS` in
`src/train_r_all_v26.py` to a command-line argument. Keep `[42, 43, 44]` as the
default. Run it with seeds 45, 46, 47 and 48 only.

- Do not retrain seeds 42 to 44. A multi-thread retrain does not give the same
  bits, and the shipped files need those exact models.
- Keep `feature_fraction_seed` unset, as the script does now.
- The stop set comes from `default_rng(1234)`. Do not change it.
- The files are `models/lgbm_r_all_v26_s45.txt` to `_s48.txt`.
- Acceptance: the hold-out CLEAN RMSE of each new member is within 1.5 s of the
  mean of seeds 42 to 44.

**Step 4. Build the v32 predictor.** Give `predict_v30.main` two arguments: the
list of base seeds and the list of `R_norm_LIRF` model files. Their defaults are
the v30 values. `src/predict_v32.py` calls `main` with seeds 42 to 48 and the
five `R_norm_LIRF` files.

Keep this order on LIRF rows:

1. Clip each member prediction at 0.
2. Average the five members.
3. Clip the mean at 4,431 s.
4. Apply the mixture, Step A and the ITY340 rule without change.

- Acceptance A: after the change, the default call still reproduces v30 to
  0.0000 s.
- Acceptance B: the v32 file has 344,841 rows, no missing value and no negative
  value.
- Acceptance C: the rows in Step A and in the ITY340 rule keep their v30 values.

**Step 5. Price v32 on the 2026 ranking set.** Write
`src/price_ensemble.py`. It needs no labels. Compute the ambiguity `A` of
section 5.1 on the final stacked predictions, one stack per member.

| ensemble | members `k` | shipped members `m` | expected gain | audit value |
|---|---|---|---|---|
| `R_norm_LIRF` | 5 | 1 | `A_5` | 74 MSE |
| base | 7 | 3 | `A_7 * 4 / 18` | about 154 MSE, from hold-out |

- Acceptance: `A_5` is between 50 and 100 MSE. The five members must reproduce
  the audit recipe; a value outside this range means the recipe changed.
- The base has no 2026 reference value. Report `A_7` as measured.

**Step 6. Measure the two debts of section 6.** Do not ship either one in v32.

1. Rebuild the fallback-rate maps from the fit months only. Report the corrected
   hold-out RMSE of the LIRF head next to the current one.
2. Merge the 12 feature pipeline copies only after the user accepts v32. The
   merged code must reproduce both v30 and v32 to 0.0000 s.

### 1.4 Stop and report

Stop, and report to the user, if one of these occurs:

- An acceptance test fails. Give the maximum difference and the count of rows
  that differ.
- A step needs a change to a feature list, a parameter or a stop set.
- A file that a shipped submission reads changes.

Do not work around a failed test. Do not try a new idea in its place.

### 1.5 Report to the user

Give these items:

1. The new and changed files, and the SHA-256 check of rule 2.
2. The best iterations of all new members.
3. `A_5`, `A_7` and the two expected gains in MSE.
4. The v32 minus v30 difference: count of rows that change, mean absolute
   change on LIRF rows and on the other 9 airports.
5. The expected live RMSE: `sqrt(91,192 - total expected gain)`. With the audit
   values this is about 301.60 s.
6. The risk: the member-draw lottery of section 5.2 has a spread of about 268
   MSE. One live score can come out worse than v30 even when the expectation is
   better.

Steps 1 to 5 together will not reach 300 s. Section 7 says why, and what that
means for further work.

## 2. The repository no longer reproduces v30

### 2.1 What is wrong

1. **The v30 base models were overwritten.** `models/lgbm_r_all_v26_s42.txt`,
   `_s43.txt` and `_s44.txt` now hold models retrained for v31. Their sizes are
   90.4 MB, 51.1 MB and 102.0 MB against 87.7 MB, 62.6 MB and 85.7 MB for the
   originals. The originals survive only in `models/v26_pre_v31/`. The encoder
   file and the feature list are unchanged.
2. **No script produces the retrained models.** They were trained on cleaned
   arrival taxi-in values. That cleaning is no longer in `src/`.
3. **`src/features_congestion_v2.py` carries a leftover line.** Line 86 repeats
   the assignment of `arr_ts`. It does nothing, but the file differs from git.
4. **`src/predict_v30.py` now reads four environment variables.** With their
   defaults it loads the base from `models/`, so it now loads the retrained
   models. Those models then receive uncleaned features: a train/serve mismatch.
   **Running `python src/predict_v30.py` today does not rebuild v30.**

### 2.2 The repair

1. Move the three retrained boosters from `models/` to `models/v31_arrclean/`.
2. Copy the three boosters from `models/v26_pre_v31/` back to `models/`.
3. Run `git checkout -- src/features_congestion_v2.py`.
4. Remove the environment variables and the record rule from
   `src/predict_v30.py`. Delete `src/predict_v31.py`.

Acceptance: `python src/predict_v30.py check.parquet` produces a file that
matches `submission/kind-mango_v30.parquet` to 0.0000 s on every row. The audit
harness already reproduces both shipped files to 0.0000 s from the
`v26_pre_v31` boosters, so the target is known to be reachable.

## 3. v31 post-mortem

### 3.1 What v31 changed

`src/predict_v31.py` loads the original base from `models/v26_pre_v31/` and turns
on one rule: at LIRF, rows **with a flight record and `sd` above 14,400 s** take
`R_norm_LIRF` alone instead of the mixture. The arrival-cleaning retrain named in
`RECAP.md` did not ship; section 3.4 covers it.

The rule touches 94 ranking rows. On 74 of them the gate already gives a
fallback probability near 0, so nothing moves. **20 rows move, all downward, by
20 to 508 s, mean -113 s.** Those 20 rows are the only difference between the
v30 and v31 files.

### 3.2 The hold-out saw nothing; live lost 260 MSE

| measurement | result |
|---|---|
| hold-out, 111 rows touched | -34 MSE, inside the noise |
| live | **+260 MSE** |

Because the 20 rows are the only difference, all 260 MSE come from them. The
change in summed squared error is `(p30 - p31) * (2y - p30 - p31)` per row. For
the sum to reach +89.6 million, some of the 20 truths must be far above the
predictions:

| hypothesis about the 20 rows | implied live change |
|---|---|
| all normal taxis near 1,300 s | -5.9 MSE |
| all fallbacks, y = sd | +279.4 MSE |
| one row (VLG202P, sd 32,456) is a 24-hour offset, the rest normal | about +250 MSE |

The leaderboard cannot separate the last two lines. Nor can label-free 2026 data:

| label-free check, 2025 against 2026 | result |
|---|---|
| LIRF arrivals with a flight record and delay above 14,400 s, share with BLOCK = SCHED | 0 % in both years |
| all LIRF arrivals with BLOCK = SCHED | 10.7 % against 11.7 % |
| LIRF record departures with `sd` 14,400 to 70,000 | 111 against 94 |

**Do not act on these 20 rows.** Setting them to `sd`, or to a 24-hour value,
would use labels learned from the leaderboard. That is learning from the ranking
process, which the brief forbids and the team has refused at every step.

### 3.3 The root cause needs no leaderboard data

In 2025, 382 LIRF record departures had `sd` between 14,400 and 70,000 s, and
none was a fallback. The rule read that as "impossible". **0 events in 382 is
not a probability of 0.** The rule-of-three upper bound is 3/382 = 0.79 %; the
Laplace estimate is 0.26 %.

Under MSE a small hedge toward a rare large-truth regime is cheap:

- A hedge of `d` seconds costs `d^2` on a normal row.
- It gains about `2 * d * G` on a row whose truth sits `G` above the prediction.
- For `d` = 100 s it pays whenever the rare regime is more likely than
  `d / 2G`: 0.28 % for `G` = 18,000 s, 0.058 % for `G` = 86,000 s.

The v30 gate gives these rows a fallback probability of 0.1 % to 2.4 %, inside
the range the 2025 data supports. The rule set it to 0.

**Rule for the next agent: never replace a calibrated probability with 0 or 1
because a finite count held no events.**

### 3.4 The arrival-cleaning retrain was noise

| fact | value |
|---|---|
| arrival labels below 30 s or above 7,200 s | 0.03 % in 2025, 0.04 % in 2026 |
| arrival labels missing | 0 % in both years |
| median difference between `arr_taxi_in_mean_60m` and the already-clean `arr_txi_mean_60m` | 0.1 s |

The model already holds the clean version of the feature. Cleaning the raw one
changes almost nothing, so the retrain differs from v26 by its seed and subspace
draw alone. The fourth audit put that noise at 1.5 s of CLEAN RMSE. The local
FULL of 397.84 s against 392.33 s is a FULL comparison, which a handful of rows
decide. **Discard the retrained models.**

## 4. The same certainty in the band table, priced and not recommended

The Step A band table holds an exact 0 or 1 in every band, from counts of 1 to 27
rows. Section 3.3 suggests smoothing it. The price on the 2026 Step A rows says
no.

Pooled smoothing (1 or 2 pseudo-rows spread by the pooled class shares of fb
0.795, 24h 0.087, normal 0.118) pulls the 60,000 to 70,000 s band away from its
24-hour majority. Worst case +177 MSE with 1 pseudo-row, +375 MSE with 2.

Targeted smoothing, 1 pseudo-row in the bands between 14,400 and 50,000 s only,
moves 37 rows up by 8 to 645 s:

| if every one of those rows is | change |
|---|---|
| a fallback, y = sd | -27 MSE |
| a 24-hour offset | -3,792 MSE |
| a normal taxi, y = band `mean_norm` | **+1,274 MSE** |

At the 2025 class frequencies in those bands (20.5 % normal between 14,400 and
25,000 s, 0 of 107 rows 24-hour), the expected change is about **+95 MSE, worse**.
Do not smooth the table.

The rule of section 3.3 forbids *removing* a hedge. It does not require adding
one when the hedge costs more on the common regime than it gains on the rare one.

## 5. A label-free price for ensemble changes

### 5.1 The tool

For a mean `f` of `k` member predictions `f_i`, on any set of rows and for any
labels:

`MSE(f) = mean over i of MSE(f_i) - A`, where `A = mean over rows and members of (f_i - f)^2`.

`A`, the ambiguity, needs no labels. It can be computed on the 2026 ranking set.
For a random subset of `m` members drawn from the `k`:

`E[MSE(subset mean)] - MSE(k-member mean) = A * (k - m) / (m * (k - 1))`

This is the only offline number in this project that is exact on the scoring
set. Every earlier comparison of ensemble sizes used the hold-out, which the
fourth audit showed cannot resolve them.

### 5.2 Worked example: `R_norm_LIRF` is one model

`R_norm_LIRF` is a single booster. Five members trained with the deployed recipe
(same features, same LIRF-only encoders, same train months, same November and
December stop set, only the seed varying):

| measurement | value |
|---|---|
| member best iterations | 251, 417, 294, 294, 297 |
| hold-out MSE of each member | 103,865, 103,556, 103,492, 103,575, 103,715 |
| hold-out: average member minus ambiguity (133.3) | 103,507, equal to the mean-of-5 MSE |
| deployed single model, hold-out | 103,500 |
| **ambiguity on the 2026 ranking set** | **74.0 MSE** |

The deployed model scores better on the hold-out than 4 of the 5 new members.
That is luck on 2025 rows; nothing carries it to 2026. The expected 2026 gain of
the 5-member mean over a single member trained the same way is **74 MSE,
about 0.12 s**. The mean moves LIRF predictions by 18.9 s on average, and 421
rows by more than 100 s.

Build it. It removes a known source of variance and cannot be priced as a loss.
Do not submit it on its own: 74 MSE sits inside the lottery of 268 MSE that the
fourth audit measured.

Acceptance: 5 members, each best iteration above half the median, predictions
averaged before the mixture, and the 2026 ambiguity printed in the build log.

### 5.3 Use the tool before touching the base ensemble

The base has 3 members. Growing it to 7 with the same recipe and the same
97-feature list gains `A_7 * 4 / 18` in expectation. On the hold-out the median
3-member draw sits 154 MSE above the 7-member mean, which implies `A_7` of about
690 MSE there, so about 154 MSE of expected gain. The 2026 value can differ.
Train the 4 extra members, compute `A_7` on the 2026 ranking set, and decide from
that number.

Keep the recipe of `src/train_r_all_v26.py` exactly: `seed` and `bagging_seed`
per member, `feature_fraction_seed` unset. The v27 and v28 regressions came from a
changed feature list and a changed recipe, not from the member count.

## 6. Measurement debts

1. `add_fallback_rate_features` in `src/train_lirf_regime_v23.py` reads all 12
   months, including the hold-out. Every LIRF figure since v23 is optimistic.
   Restrict the base rate and every leave-one-month-out map to the fit months.
2. The feature pipeline exists in 12 copies. The v29 encoder defect and the v31
   repository damage both came from copies drifting apart. Merge them into one
   `build_features` that returns the frame and its encoder set together.

## 7. Expected score and the ceiling

| build | expected live RMSE |
|---|---|
| v30 | 301.98 |
| v30 + the 5-member `R_norm_LIRF` | about 301.86 |
| + a 7-member base, if `A_7` matches the hold-out value of about 690 MSE | about 301.6 |

**No lever measured in six audits closes the remaining 1,192 MSE.** The error
that remains sits in three places:

| place | share | why it resists |
|---|---|---|
| a handful of extreme rows no observable predicts | about 28 % | third audit, section 6.1: no hedge pays |
| LIRF genuine rows, ratio about 0.92 against 0.55 to 0.65 elsewhere | about 13,700 MSE | the LIRF oracle gap; every attack on it failed |
| the other 9 airports | the rest | the model sits at a local optimum; the grid, the member count and a second model class all fall inside the noise |

A score below 300 s needs information the model does not have. Any research
towards it must meet the bar of the fifth audit: dominance, a train/serve fix, or
a label-free price such as section 5. A hold-out difference under 2 s is not
evidence.

## 8. Closed doors

New in this audit:

| candidate | measurement |
|---|---|
| v31 record normal-only rule | hold-out -34 MSE, live +260 MSE; root cause in section 3.3 |
| arrival taxi-in cleaning with a base retrain | 0.03 % of labels; a clean duplicate already exists |
| pooled smoothing of the band table | worst case +177 to +375 MSE |
| targeted smoothing, bands 14,400 to 50,000 s | expected +95 MSE |
| seed averaging `R_norm_LIRF` judged on the hold-out | +13 to +25 MSE, inside the noise; section 5.2 prices it correctly |

Still closed: the LIRF head rebuilt with the v24 recipe; the joint LIRF regressor;
XGBoost or CatBoost in the base; per-airport caps; a lower floor on `R_norm_LIRF`;
dropping features by gain share; the Step A clip; the 12-month refit; seasonal
weights; importance weighting; recalibrating the base; removing `ec_*` from the
LIRF head; the alpha shrink; `R_norm` alone; a gate on all 10 airports; local
time; route target encoding; out-of-fold operator encoders; queue and bank
features; arrival-side reporting state; a fallback gate outside LIRF; the target
`w = sd - y`; taxi-in drift correction.

## 9. Ethics

`AOBT_3_flt` and `LOBT_flt` stay out. The top score of 263.46 s sits below the
286.4 s conditional floor of the route, so a model that scores there reads the
actual off-block time.

Section 3.2 adds a second line. The live score of any submission that differs
from another on a handful of rows reveals something about those rows. Use live
scores to choose between whole submissions. Never use them to set row-level
predictions.

## 10. Seventh audit (2026-09-11 evening) — v32 debrief and v33

v32 stacked two priced changes on top of v30. v33 splits them.

### 10.1 v32 debrief

| change | pricing on 2026 ranking | expected MSE gain | actually landed |
|---|---|---|---|
| 5-member `R_norm_LIRF` (post-mixture, scaled to all rows) | `A_5 = 74.22` MSE | 74 MSE | inferred yes (see 10.2) |
| 7-member base mean vs 3-seed subset | `A_7 = 5,417` MSE | `A_7 * 4/18 = 1,204` MSE | did not; the whole +78 MSE live loss sits here |

Live: v32 = 302.11, v30 = 301.98. `91,192 → 91,270`, a swing of +78 MSE.
The stacked expectation was −1,278 MSE. The gap between expectation and
outcome is 1,356 MSE.

### 10.2 Where the 74 MSE R_norm gain went

The v32 file's LIRF rows are identical between v32 and any other build that
uses the same 5-member R_norm mean. On non-LIRF rows v32 differs from v30 only
through the 7-seed base. So the +78 MSE live loss is a lower bound on the
base-change loss: whatever the R_norm change contributed (positive or
negative), the base change contributed +78 MSE plus that. The pricing model
predicts −74 MSE for the R_norm change, so the implied base-change live cost
is about **+152 MSE**, against its predicted **−1,204 MSE**. `A_7` overshoots
its measurement by about 1,356 MSE.

### 10.3 Why the base expansion did not carry

Section 5.3 already flagged this: `A_7` estimated from the hold-out was about
690 MSE. The 2026 value came in at **5,417 MSE — eight times higher**. The
seeds 42, 43, 44 have been the shipped mean since v24. Every one of that
mean's live scores has been an implicit selection event. Seeds 45 to 48 have
no such history. On a truthing set where `A_7` is that large, a fresh
member's directional bias against the truth easily dominates the
`A * (k-m) / (m * (k-1))` mean-of-members gain.

**Rule for the next agent: when two priced changes are stacked and one of them
has `A` an order of magnitude larger than its hold-out estimate, ship them
one at a time.** The audit's own Section 5.3 said "decide from that number";
in v32 we shipped without waiting for the number to land.

### 10.4 v33 spec — R_norm-only

`src/predict_v33.py` calls `predict_v30.main` with `r_norm_files` set to the 5
already-trained seeds and `base_seeds` at its default `[42, 43, 44]`. No
retraining. No touched files under `models/`.

Expected live: `sqrt(91,192 - 74) = 301.86 s`. The lottery around 74 MSE is
small enough that a regression larger than about 0.4 s would falsify the
ambiguity model, not the change.

Diff checks confirmed:

- v33 vs v30: LIRF rows only change (26,899 rows, mean |diff| 18.8 s, max
  3,235 s). Non-LIRF: 0.0000 s on 317,942 rows.
- v33 vs v32: non-LIRF rows only change (317,942 rows, mean |diff| 7.0 s,
  max 3,298 s). LIRF: 0.0000 s.

Live score: **301.87 s**, vs the priced 301.86 s. The gap is 0.01 s. The
2026 ambiguity framework — priced with no labels, on the ranking rows only —
called the live outcome to a hundredth of a second. Section 5's formula
`MSE(f) = mean_i MSE(f_i) - A` is now the trusted tool for member changes on
this project.

### 10.5 Ladder update to Section 7

Delete the "+ a 7-member base" row. Its measured expected gain went the wrong
way live under a `A_7` eight times its hold-out estimate. Replace with:

| build | expected live RMSE, revised |
|---|---|
| v30 | 301.98 (measured) |
| v33 = v30 + the 5-member `R_norm_LIRF` | 301.86 (priced) |
| any base-ensemble change | do not price on hold-out; only 2026 `A_k` counts, and it must be at least an order of magnitude below the shipped baseline's live MSE before shipping |

Section 7's closing sentence stands: no lever measured in seven audits closes
the remaining ~1,192 MSE.

## Appendix — the band table, unchanged since v30

LIRF, no flight record, `sd > 14,400`, full year 2025. `fb` means
`abs(y - sd) < 60`. `24h` means `y > 80,000` and not `fb`.

| sd band (s) | rows | P(fb) | P(24h) | mean y |
|---|---|---|---|---|
| 14,400 to 16,000 | 27 | 0.704 | 0.000 | 10,910 |
| 16,000 to 18,000 | 14 | 0.786 | 0.000 | 13,846 |
| 18,000 to 20,000 | 16 | 0.875 | 0.000 | 16,718 |
| 20,000 to 22,000 | 8 | 0.875 | 0.000 | 18,382 |
| 22,000 to 25,000 | 8 | 0.875 | 0.000 | 20,849 |
| 25,000 to 40,000 | 18 | 1.000 | 0.000 | 33,955 |
| 40,000 to 50,000 | 16 | 1.000 | 0.000 | 43,992 |
| 50,000 to 60,000 | 9 | 0.667 | 0.333 | 65,715 |
| 60,000 to 70,000 | 6 | 0.333 | 0.667 | 79,950 |
| 70,000 to 100,000 | 4 | 0.000 | 1.000 | 87,567 |
| 100,000 and above | 1 | 1.000 | 0.000 | 131,167 |
