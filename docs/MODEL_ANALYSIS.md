# Model analysis and v21 architecture

Updated 2026-09-09. This audit supersedes the v20 blueprint of the same day.
Status: live best 345.70 s (v20), rank 49/84. Top team 263.46 s.

This document contains no model code. Section 8 is the ordered plan. Sections
2 to 7 are the evidence. Every number below comes from a measurement made in
this audit. Section 9 lists the closed doors. The appendix carries forward the
LIRF band table from the earlier audit.

## 0. Reference model for this audit

Sections 2 to 5 use one small model, not the v20 stack. The small model isolates
each effect and it runs in 40 s.

| item | value |
|---|---|
| inputs | 6 categorical, 7 numeric: airport, operator, stand, runway, aircraft type, destination, sd, sd mod 86400, mvt_eobt1, mvt_iobt, eobt1_sched, hour, dow |
| train | months 2 to 6 and 8 to 10, 1,412,355 rows |
| early stop | months 11 and 12, 327,968 rows |
| test | months 1 and 7, 344,336 rows |
| params | learning rate 0.05, 255 leaves, min_data_in_leaf 100 |
| result | FULL RMSE 470.63, CLEAN RMSE 309.00 |

The v20 stack reaches CLEAN 280.17 s with 143 inputs. With the tuned v21
parameters the small model reaches 302.27 s, so the 130 extra inputs are worth
22 s. Every delta in this document is a delta on the small model. A delta
measured on the small model transfers to the full stack only when the mechanism
is independent of the feature set. Section 8 states the transfer risk per item.

## 1. Conclusion

Four findings set the plan.

1. **The score lives in 33 rows.** On the Jan+Jul 2025 test set, 33 rows carry
   89 s of the 403.60 s RMSE. Section 2 gives the budget. The 2026 set is the
   same size, so it holds about the same count.
2. **The hold-out metric cannot measure a 2 s change.** The bootstrap interval
   of FULL RMSE is 131 s wide. Section 5 gives the numbers. Most accept and
   reject calls in `RECAP.md` sit inside that interval.
3. **`linear_tree` is worth 10.5 s of CLEAN RMSE.** The paired bootstrap
   interval is 7.3 s to 14.5 s. Section 4 gives the measurement. No feature
   family added since v9 comes close.
4. **`features_opdi_live.py` leaks the label.** The row's own OPDI record sits
   inside its own rolling window. Section 3.1 gives the measurement. The
   docstring calls the module leak-free.

The realistic target is 300 to 320 s. The pooled standard deviation of taxi-out
inside an (airport, stand, runway, hour-bin) cell is 286.4 s on genuine clean
rows. The top team scores 263.46 s on the full 2026 set, below that floor. A
model that scores below the conditional floor of the route uses the actual
off-block time. `AOBT_3_flt` supplies it. The team excludes that field. The
exclusion sets the floor near 300 s. Section 6 gives the per-airport floors.

## 2. Where the error is

### 2.1 MSE budget by target band

Test set, small model, 344,336 rows, total MSE 222,426.

| y band (s) | rows | share of rows | MSE contribution | share of MSE |
|---|---|---|---|---|
| 0 to 600 | 43,235 | 12.6 % | 5,512 | 2.5 % |
| 600 to 1,200 | 216,739 | 62.9 % | 30,023 | 13.5 % |
| 1,200 to 2,400 | 79,526 | 23.1 % | 32,738 | 14.7 % |
| 2,400 to 7,200 | 4,687 | 1.4 % | 26,517 | 11.9 % |
| 7,200 to 14,400 | 104 | 0.03 % | 5,123 | 2.3 % |
| 14,400 to 80,000 | 38 | 0.011 % | 23,743 | 10.7 % |
| above 80,000 | 7 | 0.002 % | 98,770 | 44.4 % |

Seven rows carry 44.4 % of the error. The 20 largest single-row errors carry
55 % of it. Exact predictions on those 20 rows move the score from 471.62 to
314.32 s.

### 2.2 MSE budget by airport

| airport | rows | RMSE | share of MSE |
|---|---|---|---|
| LIRF | 26,528 | 1,264.1 | 55.3 % |
| LFPG | 39,589 | 610.5 | 19.3 % |
| EGLL | 40,210 | 338.6 | 6.0 % |
| LTFM | 46,539 | 288.7 | 5.1 % |
| EHAM | 40,949 | 245.3 | 3.2 % |
| EDDF | 36,830 | 237.1 | 2.7 % |
| LEBL | 28,985 | 246.6 | 2.3 % |
| EDDM | 27,091 | 245.8 | 2.1 % |
| LSZH | 22,287 | 264.5 | 2.0 % |
| LEMD | 35,328 | 203.2 | 1.9 % |

LIRF and LFPG carry 74.6 % of the error with 19.2 % of the rows.

### 2.3 The structure of the extreme rows

All 2025 departures, 2,084,659 rows. 122 rows have `y > 14,400`.

| class | rows | airports |
|---|---|---|
| `abs(y - sd) <= 2` | 46 | LIRF 46 |
| `2 < abs(y - sd) <= 120` | 56 | LIRF 56 |
| no marker, `abs(y - sd) > 120` | 20 | LIRF 14, LFPG 4, LSZH 2 |

The doc of 2026-09-08 tested `y = sd - 86,400 * k`. That form matches 0 rows.
The form `y = sd + 86,400` matches 1 row, at LSZH. 12 of the 14 LIRF rows with
no marker hold `y` between 87,168 and 88,392 s, while their `sd` runs from
1,504 to 93,535 s. For those rows `sd` carries no information about `y`.

**The extreme regime is a null-flight-record condition, not a delay condition.**

| airport | flight record | rows | rows with y > 14,400 | rate |
|---|---|---|---|---|
| LIRF | none | 1,487 | 115 | 773 bp |
| LSZH | none | 2,054 | 2 | 9.7 bp |
| LFPG | none | 3,754 | 2 | 5.3 bp |
| 7 others | none | 14,878 | 0 | 0 bp |
| all 10 | present | 2,062,486 | 3 | 0.015 bp |

### 2.4 Oracle bounds

Start from the section 4.2 mixture that scores 403.60 s.

| oracle | rows | RMSE |
|---|---|---|
| exact on `y > 14,400` and `abs(y - sd) <= 1` | 12 | 398.14 |
| exact on `y > 14,400` and `abs(y - sd) > 1` | 33 | 314.51 |
| exact on all rows with `y > 14,400` | 45 | 307.47 |
| exact on the rows with `7,200 < y <= 14,400` | 104 | 398.66 |

**The whole remaining headroom on the full metric is 33 rows.** 30 sit at LIRF
and 3 at LFPG. The Step A band table already treats the LIRF group. Step 5 shows
the 3 LFPG rows are unreachable: no observable separates them from the other
5,800 null-record rows at LFPG and LSZH.

An oracle gate on the exact-fallback class `abs(y - sd) <= 1` is worth 10.2 s
(416.76 to 406.56). The calibrated soft gate of section 4.2 reaches 403.60 s,
below the hard oracle. A soft mixture beats a perfect hard switch, because it
also helps the rows near the class boundary. Do not build a hard switch.

## 3. Defects in the current code

### 3.1 `features_opdi_live.py` includes the row's own record

`_rolling_stats` selects OPDI records with `runway_entry_ts` in `[t - w, t]`,
where `t` is the row's own take-off time. A flight enters the runway before it
takes off. The row's own taxi-out therefore sits inside its own window. The
module has no flight key to exclude it.

Correlation of `opdi_live_taxi_mean_prev_30m` with the target, genuine clean
rows, 30-minute window:

| airport | no lag | 600 s lag | records in window |
|---|---|---|---|
| LSZH | +0.2962 | +0.1634 | 11.5 to 7.4 |
| EDDF | +0.2088 | +0.1575 | 8.3 to 5.7 |
| LEBL | +0.1604 | +0.1133 | 4.6 to 3.5 |

At LSZH 45 % of the feature's correlation with the target is the target itself.
The feature entered at v15. v15 improved the 2025 hold-out by 1.1 s and the live
score by 0.3 s. The gap is the leak.

Fix: subtract a 600 s lag from the window end. The lag removes the own record,
because the runway entry of a flight falls within 600 s of its take-off.

### 3.2 The OPDI coverage collapses between 2025 and 2026

`taxi_out_v2.parquet` record counts:

| airport | 2025 | 2026 |
|---|---|---|
| EDDF | 29,021 | 42,035 |
| LSZH | 113,252 | 70,863 |
| EDDM | 11,512 | 1,789 |
| LEBL | 23,267 | 174 |
| LEMD | 14,375 | 92 |
| EHAM | 1,312 | 170 |
| LFPG | 675 | 4 |
| EGLL | 64 | 19 |
| LIRF | 9 | 6 |
| LTFM | 0 | 0 |

18 inputs (9 OPDI counts and 9 OPDI live-taxi) are dense in training and empty
in scoring at LEBL, LEMD, EDDM, LFPG and EHAM. LightGBM routes a missing value
down one branch. A feature that is present in training and absent in scoring
shifts every row at those airports to that branch. Those five airports hold
44 % of the ranking rows.

Fix: keep the OPDI families at EDDF and LSZH. Drop them elsewhere, or add a
per-airport coverage input so the model can learn the two regimes.

### 3.3 `train_lirf_noflt_detector.py` and `predict_v20_final.py` disagree

| file | line | expression |
|---|---|---|
| `src/train_lirf_noflt_detector.py` | 59 | `STAND_mvt.str.extract(r"^([A-Za-z]+)")` |
| `src/predict_v20_final.py` | 73 | `STAND_mvt.astype(str).str.slice(0, 1)` |

LIRF stands are numeric. The regex returns "UNK" for every LIRF row, so the
detector trained on a constant input. The prediction path sends "6", "8", "3"
and other digits. The detector never saw those categories, so it routes them as
missing. The 6.3 rule therefore runs on an input the model does not know.

Fix: use `str.slice(0, 1)` in both files. Retrain the detector.

### 3.4 `features_operator.py` fits the target encoders in-fold

`fit_encoders` computes the group median of the target over the training slice.
`apply_encoders` writes that median onto the same rows. Each training row
contributes to its own encoded value. The finest key is
(operator, airport, stand) with a floor of 20 rows, so the row's own weight is
at most 1/20. The model over-trusts the encoders in training and loses that
trust at inference.

Fix: fit the encoders out of fold. Use 5 month-blocked folds. Use the full
training fit for the ranking set.

### 3.5 `features_turnaround.py` rebuilds an observed field

The module computes the in-block time of the linked arrival as
`landing + clip(TAXITIME_SEC_mvt, 0, 7200)` and fills a missing taxi time with
0. `BLOCK_TIME_UTC_mvt` holds the same value exactly, and it is present on
100 % of arrival rows in both the training files and the ranking file. The
identity `TAXITIME_SEC_mvt = BLOCK_TIME_UTC_mvt - MVT_TIME_UTC_mvt` holds on
every arrival row with a difference of 0 s.

The clip corrupts every arrival with a taxi-in above 7,200 s. The fill writes a
wrong in-block time instead of a missing one.

Fix: read `BLOCK_TIME_UTC_mvt` directly. Leave a missing value missing.

### 3.6 `train_lgbm_v21.py` early-stops on the hold-out

Lines 143 and 151 pass the Jan+Jul frame as `valid_sets`. The reported CLEAN 283.85 s
is therefore an optimistic estimate, and `lgbm_v21.txt` is still the base of the
Step A rule inside `predict_v20_final.py`. `train_r_all_v20.py` fixes the split
for `R_all`. Fix `v21` the same way, or drop `v21` from the deployed path.

### 3.7 Two items that look wrong and are not

- **Categorical code order.** `load_training_categories` returns
  `pd.Index(list(set))`. Python randomises the iteration order of a string set
  per process. `astype("category")` in training sorts the categories. The two
  orders differ. LightGBM stores the training category list in the model file
  and realigns the input by value, so the predictions match. Verified: two
  permutations of the same category list give identical output. The call is
  dead work, and it becomes a real defect the day anyone feeds a numpy array.
- **The window boundary on the ranking set.** The ranking file holds January
  and July 2026 only. Rolling windows at the start of each month see no prior
  rows. The affected share is under 2 % of rows. Leave it.

## 4. Measured levers

### 4.1 `linear_tree`

A LightGBM leaf holds a constant. The model must therefore reproduce
`y ≈ mvt_eobt1` and `y ≈ sd` through many axis-aligned splits. A linear tree
holds a linear model in each leaf, so one leaf reproduces an identity exactly.

Small model, test on Jan+Jul:

| variant | leaves | FULL | CLEAN |
|---|---|---|---|
| constant leaves, learning rate 0.05 | 255 | 470.63 | 309.00 |
| constant leaves, the tuned `BEST_PARAMS` of v21 | 440 | 467.00 | 302.27 |
| constant leaves, `min_data_in_leaf` 500 | 127 | 480.93 | 312.87 |
| `linear_tree`, `linear_lambda` 1.0, untuned | 127 | **427.84** | **298.42** |
| `linear_tree` plus signed-log copies of sd and the OBT deltas | 127 | 419.57 | 298.42 |

The untuned linear tree beats the tuned constant-leaf model by 3.9 s of CLEAN
RMSE and by 39 s of FULL RMSE. It also runs in half the time, because it needs
half the leaves.

Paired bootstrap of the CLEAN difference against the 255-leaf constant-leaf
model, 300 resamples: **-10.53 s, interval -14.48 to -7.25 s.**

**Guard against extrapolation.** A leaf-linear term in `sd` extrapolates without
bound. The effect is visible on the `R_norm` linear tree of section 4.2 at EHAM:
on rows with `sd` at -38,147 s and +74,428 s it predicts 2,859 s and 5,316 s,
and on one row with `sd` at -27,045 s it predicts 28,561 s where the truth is
1,390 s. Eight such rows carry 113 % of the 32.3 s EHAM gap.

Signed-log copies of `sd`, `mvt_eobt1`, `mvt_iobt` and `eobt1_sched` cost
nothing on CLEAN and cut the EHAM RMSE of the all-row linear tree from 248.6 to
245.6 s. Add the copies and keep the raw columns for the split decisions. Test
the same fix on `R_norm`, where the damage is larger.

### 4.2 A calibrated fallback gate, applied at all 10 airports

Build a classifier for `abs(y - sd) <= 1`. Mix its probability with a regressor
trained on the complement:

```
y_hat = p(x) * sd + (1 - p(x)) * R_norm(x)
```

That expression is the conditional mean. The conditional mean minimises MSE.
A calibrated `p` therefore cannot lose in expectation.

Small model, `R_norm` with `linear_tree`, test on Jan+Jul:

| variant | FULL | CLEAN |
|---|---|---|
| `R_norm` alone | 416.76 | 299.47 |
| mixture, raw probability | 400.62 | 300.60 |
| mixture, one global isotonic calibrator | 399.04 | 300.82 |
| mixture, per-airport isotonic calibrators | **396.24** | 301.03 |

Reliability of the per-airport gate on the test set, by decile of `p`:

| predicted | actual |
|---|---|
| 0.0000 | 0.0001 |
| 0.0007 | 0.0009 |
| 0.0020 | 0.0026 |
| 0.0034 | 0.0031 |
| 0.0048 | 0.0056 |
| 0.0073 | 0.0075 |
| 0.0161 | 0.0180 |
| 0.0784 | 0.0737 |

Three rules govern the gate.

1. **Select the gate on log loss, not on AUC.** AUC ranks. The mixture needs a
   calibrated probability. The AUC-selected gate stopped at 40 rounds and scored
   403.60 s with a global calibrator. The log-loss-selected gate ran to 68
   rounds and scored 399.04 s with the same calibrator.
2. **Calibrate per airport.** The base rate runs from 0.18 % at EDDF and LSZH to
   4.48 % at LIRF. A global calibrator over-fires at the low-rate airports.
3. **Fit the calibrator on months 11 and 12, never on the test months.**

The earlier audit rejected mixtures. Section 9 states why the earlier form
failed and this one does not.

### 4.3 The per-airport comparison is a comparison of 8 rows

The `R_norm` linear tree loses 32.3 s at EHAM against the all-row constant-leaf
model. Eight rows carry 113 % of that gap. All eight hold a huge `sd` and a
normal `y`, and they are the extrapolation failure of section 4.1. The gate is
not the cause: it almost never fires at EHAM, and the mixture scores the same
276.7 s as `R_norm` alone.

The bootstrap interval of the EHAM difference runs from -91.2 to +5.7 s. It
covers 0. On this sample EHAM cannot tell the two models apart.

**Never accept or reject a component on a per-airport hold-out RMSE.** Use the
row-level squared-error difference and a paired bootstrap. Section 5 gives the
protocol.

## 5. The measurement protocol is the largest process defect

Bootstrap intervals on the Jan+Jul 2025 test set, 300 resamples:

| metric | point | 5 % to 95 % | width |
|---|---|---|---|
| FULL RMSE | 470.6 | 403.6 to 534.5 | 131 s |
| CLEAN RMSE | 309.00 | 302.63 to 316.01 | 13 s |
| paired CLEAN difference, two models | -10.53 | -14.48 to -7.25 | 7 s |

`RECAP.md` records 16 accept and reject calls on hold-out differences of 9.8 s
or less. Each call used the same two months, and the interval above covers all
16. The team has run about 30 comparisons on one 344,336-row sample. The
selected winner is therefore the winner of the noise.

Three changes fix the protocol.

1. **Report the paired difference, not two RMSE values.** Compute the per-row
   squared-error difference. Bootstrap it. State the interval. Accept a change
   only when the interval excludes 0.
2. **Use 12-fold month-blocked cross-validation for every selection.** Every
   month becomes a fold. The estimate then uses 2,084,659 rows instead of
   344,336, and it averages over the season instead of sampling two months. The
   out-of-fold predictions also feed the ensemble weights and the isotonic
   calibrators without a second split.
3. **Report three numbers, never one.** CLEAN RMSE on `30 <= y <= 7,200`, the
   count and the identity of every row above 7,200 s, and the full RMSE. 44 % of
   the full RMSE is a report on 7 rows.

## 6. What is left in the clean rows

Pooled within-cell standard deviation of taxi-out, genuine clean rows,
1,895,341 rows:

| conditioning set | pooled sd |
|---|---|
| airport | 362.5 |
| airport, runway | 347.8 |
| airport, stand, runway | 297.1 |
| airport, stand, runway, hour-bin | 286.4 |
| airport, stand, runway, hour-bin, month | 278.1 |

Per airport, conditioned on stand, runway and hour-bin:

| airport | total sd | within cell |
|---|---|---|
| EGLL | 412.8 | 372.8 |
| LIRF | 437.3 | 380.2 |
| LFPG | 391.3 | 324.4 |
| LTFM | 427.6 | 318.0 |
| EDDM | 305.3 | 271.0 |
| EDDF | 331.9 | 232.7 |
| LEBL | 325.4 | 227.4 |
| LSZH | 295.4 | 224.6 |
| EHAM | 317.6 | 222.3 |
| LEMD | 313.3 | 217.3 |

The v20 stack scores about 280 s CLEAN, below the 286.4 s cell floor, because it
also holds the delay, the queue and the weather. The remaining clean signal is
the time-varying state of the airport. The turnaround family and the disruption
meters attack exactly that state, and together they are worth 14 s on the small
model per the earlier audit. Expect 10 to 25 s more from that direction, against
89 s from the 33 rows of section 2.4.

**Order the work by that ratio.**

## 7. Data to add

### 7.1 Temperature, from a file already on disk

`external/metar/*.csv` holds `tmpf`, `dwpf`, `relh`, `p01i` and
`ice_accretion_1hr`. `features_weather.py` reads `p01i` and drops it. It never
reads the temperature.

De-icing gates on temperature. Mean taxi-out, 2025 genuine clean rows:

| airport | -5 to 0 °C, precipitation | -5 to 0 °C, dry | above 7 °C, precipitation |
|---|---|---|---|
| LFPG | 2,354 | 1,372 | 1,046 |
| EGLL | 1,950 | 1,416 | 1,419 |
| EDDF | 1,456 | 934 | 907 |
| EDDM | 1,469 | 944 | 835 |
| EHAM | 1,393 | 988 | 743 |
| LSZH | 1,196 | 837 | 739 |

Precipitation at 0 °C adds 400 to 1,000 s. The same precipitation at 15 °C adds
under 60 s. The current inputs hold `wx_precip` and `wx_freezing` but no
temperature, so the model cannot separate the two cases. It falls back on
`month`, which is a weak proxy.

January is 44 % of the ranking rows.

Add `tmpc`, the dew-point spread, `p01i` and `ice_accretion_1hr`. Add the
product `wx_precip * (tmpc < 3)`. Expected gain: 3 to 8 s of CLEAN RMSE, all of
it in January.

### 7.2 An aircraft-type filter on the turnaround link

`add_turnaround` links a departure to the last arrival on the same stand within
12 h. The link is ambiguous on 11 % of rows. `AIRCRAFT_TYPE_mvt` is present on
both rows. An aircraft that departs is the aircraft that arrived. Require the
two types to match. Keep the unmatched link with a flag.

The link is the input to the six turnaround features, and `slack_sched` ranks
13 of 143 in `R_all` v20. A cleaner link raises all six.

### 7.3 Nothing else

The challenge file, the METAR archive, OSM, OurAirports, the Eurocontrol dailies
and OPDI already cover the physics. The data is rich enough. Sections 2 and 6
show the limit is not information; the limit is 33 rows and a measurement
protocol.

## 8. Ordered plan

Each step states its acceptance test. Run the test once. Use the paired
bootstrap of section 5.

### Step 1 — Fix the six defects. No new features.

Fix 3.1, 3.2, 3.3, 3.4, 3.5 and 3.6. Retrain `R_all` and the LIRF detector.

Acceptance: CLEAN RMSE does not rise. A rise means the leak of 3.1 was carrying
the hold-out score, which confirms the finding.

Transfer risk: none. These are corrections.

### Step 2 — Turn on `linear_tree`.

Add signed-log copies of `sd`, `mvt_eobt1`, `mvt_iobt`, `eobt1_sched`,
`eobt1_iobt`. Set `linear_tree` true and `linear_lambda` 1.0. Halve
`num_leaves` to 220.

Acceptance: the paired CLEAN interval against the current `R_all` excludes 0.
Target: -6 s or better on the full stack.

Transfer risk: medium. The measured -10.5 s comes from a 13-input model where
the identity terms carry more of the signal. The full stack holds 143 inputs,
so expect less.

### Step 3 — Retune the hyper-parameters on the linear tree.

`BEST_PARAMS` comes from an Optuna run of 2026-09-03 on 97 inputs, a filtered
target and a hold-out early stop. Those values hold up: on the small model they
score CLEAN 302.27 against 309.00 for a plain 255-leaf setting, and a
conservative 127-leaf, 500-row setting is worse at 312.87. **Do not loosen the
current parameters.** The 440 leaves are not the problem.

The tuning is still needed, because `linear_tree` changes the optimum. The
untuned linear tree already beats the tuned constant-leaf model. Run 60 Optuna
trials over `num_leaves`, `learning_rate`, `min_data_in_leaf`, `linear_lambda`
and the feature and bagging fractions, with the months 11 and 12 objective.

Include `cat_smooth`, `cat_l2` and `max_cat_threshold` in the search.
`STAND_mvt` holds 1,898 categories, `ADES_mvt` 1,560 and
`AIRCRAFT_OPERATOR_flt` 676; the defaults are set for tens of categories.
A single test of `cat_smooth` 50, `cat_l2` 25 and `max_cat_threshold` 64 moved
CLEAN RMSE by 0.6 s, which is inside the noise. Treat these as search
dimensions, not as a lever on their own.

Acceptance: the paired CLEAN interval against Step 2 excludes 0.

### Step 4 — Replace the rules with a calibrated regime head.

Build three components.

1. `R_norm`: the regressor of Step 3, trained on rows with `abs(y - sd) > 1`
   and `y < 80,000`.
2. `p_fb(x)`: a classifier for `abs(y - sd) <= 1`, selected on log loss,
   calibrated by a per-airport isotonic fit on months 11 and 12.
3. `q_ext(x)`: the probability of `y > 14,400`, and `E[y | y > 14,400, x]`.
   Fit both on the null-flight-record rows at LIRF, LSZH and LFPG only.
   Section 2.3 shows the rate is 0 at the other 7 airports.

Combine as one conditional mean:

```
y_hat = q_ext * E[y | extreme] + (1 - q_ext) * (p_fb * sd + (1 - p_fb) * R_norm)
```

This expression replaces the Step A band table, the 6.3 rule and the ITY340
rule with one calibrated statement. Keep the band table as the estimator of
`E[y | extreme, LIRF, sd]`; it already holds that conditional mean.

Acceptance:
- the reliability table of `p_fb` matches within 0.005 in every decile;
- the full RMSE improves and no airport loses more than its paired bootstrap
  interval allows;
- every submitted row above 7,200 s is listed with the component that produced
  it.

Do not build a hard switch. Section 2.4 shows a soft mixture beats a perfect
hard switch.

### Step 5 — Do not extend the extreme head to LSZH and LFPG.

The arithmetic says the extension cannot pay. LSZH and LFPG hold 5,808
null-record rows per year and 6 extreme rows among them, so `q_ext` is 1.03e-3.
The MSE-optimal addition to every one of those rows is therefore
`1.03e-3 * 85,000 = 88 s`. The net gain over a year is 45 million of summed
squared error, which is 22 MSE over 2,084,659 rows, or 0.03 s of RMSE.

An 88 s addition also does almost nothing for a real extreme row: it moves the
error from 86,100 s to 86,012 s.

**The LFPG and LSZH extreme rows are unreachable.** No observable separates them
from the other 5,800 null-record rows. Section 9 records the four markers tested.
Accept them as the irreducible variance of the leaderboard, and spend the effort
on LIRF, where `q_ext` is 773 bp and the band table already works.

Skip to Step 6.

### Step 6 — Move every selection to 12-fold month-blocked cross-validation.

Section 5. This step costs 12 times the training time and it removes the
selection noise that produced 16 of the entries in `RECAP.md`.

Run it before any further feature work.

### Step 7 — Add the temperature family and the type-matched turnaround link.

Sections 7.1 and 7.2. Expected gain 4 to 10 s of CLEAN RMSE.

### Step 8 — Rebuild the ensemble from different model classes.

The v25 NNLS collapsed to one member because the five members shared a feature
set, a loss and an algorithm. Real diversity needs a different bias:

- the constant-leaf model and the linear-tree model;
- a CatBoost model, which uses ordered target statistics for the 1,898-category
  stand instead of a gradient-sorted split;
- a per-(airport, stand, runway, hour-bin) shrinkage estimator as the floor.

Fit the weights on the out-of-fold predictions of Step 6, not on Jan+Jul.

Acceptance: the NNLS weight vector holds at least three non-zero entries.

## 9. Closed doors

Items measured in this audit. Do not retry them.

| item | measurement | reason |
|---|---|---|
| target `w = sd - y`, then `y_hat = sd - w_hat` | CLEAN 498.73 against 309.00 | On a normal row `w` tracks `sd`, so the tree must learn an identity over the full delay range. The reparameterisation moves the problem, it does not remove it. |
| `y = sd - 86,400 * k` as an extreme marker | 0 of 122 rows | Wrong sign. `y = sd + 86,400` matches 1 row at LSZH. |
| arrival-side fallback share as an instrument | day-level correlation 0.009 to 0.153, hour-level -0.017 to 0.046 | Departure and arrival reporting fail independently. Confirms the earlier audit. |
| LIRF day-level outage state | daily null count against daily extreme count, r = 0.566; but the rate conditional on a null row is flat, 6.25 % to 9.80 % across quintiles | The day-level count adds nothing beyond the row-level null flag. |
| hour of day as an extreme-row marker | uniform across all 24 hours | No signal. |
| taxi-in drift as a 2026 correction | mean arrival taxi-in moves -5.7 % to +4.1 % per airport between 2025 and 2026 | The observable ground truth shows no large operational drift. The live-to-hold-out gap is the tail, not a clean-row shift. |

Items closed by the earlier audit and still closed: hard-swap mixture (v22,
v27); a detector mixed with the `R_all` or `v21` output; Step A extended to
`sd > 3,600`; an EGLL detector; runway configuration as a category; the stand
occupancy bound; the EOBT, IOBT and LOBT exact-copy detectors; the seed and
feature-fraction ensemble; the 12-month refit alone; SSL drift features;
per-row submission variants.

The earlier audit rejected mixtures on the evidence of v22, v27 and v17. Those
three used a hard swap, an uncalibrated probability, or a mixture against the
all-row regressor. Section 4.2 uses a soft mixture, a log-loss-selected gate,
a per-airport isotonic calibration fitted off the test months, and `R_norm` as
the complement. The failure modes do not carry over. Read the acceptance test in
Step 4 before deploying it.

## 10. Ethics, unchanged

`AOBT_3_flt` and `LOBT_flt` stay out. Section 1 gives the arithmetic: the top
score of 263.46 s sits below the 286.4 s conditional standard deviation of the
route, which no model without the off-block time can reach. The exclusion
therefore costs about 37 s, the distance from the 300 s floor to the top score.
The remaining 46 s, from 345.70 s to 300 s, is ours to take. The team decision
stands. The README records it. Keep it there.

## Appendix — the Step A band table

Carried forward from the audit of 2026-09-08. LIRF, no flight record,
`sd > 14,400`, full year 2025. `fb` means `abs(y - sd) < 60`. `24h` means
`y > 80,000`.

| sd band (s) | rows | P(fb) | P(24h) | mean y |
|---|---|---|---|---|
| 3,600 to 7,200 | 748 | 0.373 | 0.000 | 2,925 |
| 7,200 to 14,400 | 500 | 0.590 | 0.002 | 6,380 |
| 14,400 to 25,000 | 73 | 0.795 | 0.000 | 14,654 |
| 25,000 to 40,000 | 18 | 1.000 | 0.000 | 33,955 |
| 40,000 to 50,000 | 16 | 1.000 | 0.000 | 43,992 |
| 50,000 to 70,000 | 15 | 0.533 | 0.467 | 71,409 |
| 70,000 to 100,000 | 4 | 0.000 | 1.000 | 87,567 |

The table took the live score from 430.45 to 372.40 s. Step 4 keeps it as the
estimator of `E[y | extreme, LIRF, sd]` inside the regime head.
