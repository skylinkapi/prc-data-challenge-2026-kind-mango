# Model analysis — fourteenth pass: the plan for a model under 290 s

Status on 2026-09-13: live best **299.31 s (v41)**, rank inside the top 40
of 111 teams. Top score 263.46 s. This file is the source of truth for the
next model. It replaces every earlier pass. The twelfth and thirteenth
passes, with their diagnostic tables T1 to T17 and the v39 debrief, stay in
git at `git show b193868:docs/MODEL_ANALYSIS.md`. A table id cited below
points there unless the appendix of this file repeats it.

Evidence rules, unchanged:

- A number carries its source: a live score, a paired hold-out run, a
  label-free diagnostic, or arithmetic. A number without one is marked
  **unmeasured**.
- A lever is priced by the class MSE it moves on the paired hold-out, never
  by a quick test on a small column set. The quick tests of the twelfth
  pass promised 9 to 11 s for the tempo family; the deployed frame gave
  2.55 s across v40 and v41.
- One change per upload. Five uploads per UTC day.

## 1. The deployed model, exactly

v41 runs `src/predict_v41.py`, which calls `src/predict_v30.py:main` with
the v40 base members, the v41 LIRF regressor members, no cap, and the 13
extra columns joined from the `src_v2` ranking frame by movement id. Every
default of `predict_v30.main` still rebuilds the v33 file to 0.0 s.

| component | what it serves | files | trained by |
|---|---|---|---|
| base | every row outside LIRF; the normal term at LIRF inside Step A | `models/lgbm_r_all_v40_s{42,43,44}.txt`, 110 columns | `src/train_r_all_v40.py`: v26 recipe (`linear_tree`, 220 leaves, `BEST_PARAMS`), months 2 to 6 and 8 to 12, random 12 % stop, all `y > 0` rows |
| LIRF regressor `R_norm` | genuine LIRF rows | `models/lgbm_r_norm_lirf_v41_s{42..46}.txt`, 166 columns | `src/train_r_norm_lirf_v41.py`: rows with `abs(y - sd) >= 60` and `y < 80,000`, stop months 11 and 12, 127 leaves |
| LIRF gate `p_fb` | probability that `BLOCK_TIME` equals the schedule | `models/lgbm_p_fb_lirf_v23.txt` and its isotonic map | `src/train_lirf_regime_v23.py`: target `abs(y - sd) < 60`, 14 fallback-rate columns |
| mixture | `p_fb * sd + (1 - p_fb) * R_norm` on LIRF rows | `predict_v30.py` | |
| Step A table | LIRF rows with a null flight record and `sd > 14,400` | `models/lirf_band_table_v30.json`, 11 bands, 127 rows | `src/build_lirf_band_table_v30.py`, all 12 months |
| ITY340 rule | LIRF rows with `sd > 70,000` outside Step A | `predict_v30.py:36-39` | one 2025 row |
| post-processing | clip at 0, left-merge on the template | `predict_v30.py` | |

The 13 extra columns (`src_v2/frame.py`): median, mean and count of other
departures' `MVT - EOBT_1` in the previous 30 and 60 min at the airport and
in the previous 30 min on the same runway; the backward take-off order pair;
the stand re-occupation gap; the count of departures between `EOBT_1` and
`MVT`.

## 2. The budget and the target

### 2.1 Live and hold-out

| quantity | value | source |
|---|---|---|
| v41 live MSE | 299.31² = 89,589 | `submission/kind-mango_v41.result.json` |
| target 290 s | 84,100 MSE | arithmetic |
| gap to close | **5,489 MSE** | arithmetic |
| 1 s of live RMSE at this level | about 599 MSE | arithmetic |
| member-draw lottery on a seed change | 268 MSE, about 0.45 s | ninth pass, validated by v33, v37 |
| v33 stack on the 2025 hold-out | 321.74 s, 103,517 MSE | `models/v33.holdout.json` |
| v41 stack on the same hold-out | about 102,300 MSE, from v33 minus the paired prices of v40 and v41 | arithmetic |

The price-landing record: v33 priced -74 MSE and landed; v40 priced -805
and landed -1,146; v41 priced -415 and landed -391. v32 and v37 priced
seed-only changes and missed. Prices from the paired harness land; prices
from the seed lottery do not.

### 2.2 Where the hold-out MSE sits after v41

| block | MSE on the 344,336-row scale | rows | movable |
|---|---|---|---|
| clean rows, 9 airports outside LIRF, served by the base | 46,381 | 290,000 | yes |
| LIRF clean rows, served by `R_norm` | 14,031 | 22,000 | yes |
| LIRF fallback rows, served by the gate | 5,913 | 4,053 | four gate refits regressed; low |
| 24-h LFPG row, `sd = 1,740`, null record | 20,191 | 1 | no observable |
| tail LFPG row, `y = 58,206`, `sd = 2,043`, null record | 9,461 | 1 | no observable |
| tail, other rows | 4,200 | 50 | at most 250 MSE |
| fallback rows, 9 airports, served by the base | 1,753 | 9,700 | ceiling 1,753; heads did not beat the base |
| LIRF 24-h rows, served by Step A | 1,273 | 5 | hedge is MSE-optimal per band |

The 2026 set scored 12,392 MSE lower than the 2025 hold-out for the v33
stack. The two LFPG rows alone are 29,652 on the hold-out, so the 2026 set
carries about 17,000 MSE of its own error that no hold-out shows: its own
extreme rows and the 2025-to-2026 drift. That block is not addressable from
2025 labels.

### 2.3 The arithmetic of 290

The gap is 5,489 MSE. The two blocks that can move are the clean rows,
60,412 MSE together. A 9 % cut of the clean error closes the gap. Section 4
prices the levers; their graded sum is 2,200 to 8,400 MSE. Under 290 s is
inside the range and not assured. Section 7 gives the stop rule.

### 2.4 Ceiling of the ethics stance

`AOBT_3_flt` matches the actual off-block on 17 to 24 % of 2025 rows and
`LOBT_flt` equals it on 4.3 %. Both stay out. OPDI records the scored
flight's own pushback event; that event is the label by another route and
stays out for the same reason. OPDI events of other flights are context and
stay in. The organiser reserved a second phase "if reverse engineering the
ranking is too easy" (RECAP, Discord 2026-09-11). No target below what a
model with these exclusions can reach is set.

## 3. What the data is, in ten facts

1. The label is exact: `TAXITIME = MVT_TIME - BLOCK_TIME` on 100 % of rows
   (T1). Every label error is a `BLOCK_TIME` error at the airport.
2. Four label classes: clean (90.9 %), fallback (`abs(y - sd) <= 5`,
   `BLOCK_TIME` equals the schedule), tail (`7,200 < y <= 80,000`), 24-h
   (`y > 80,000`). Definitions in `src_v2/config.py`.
3. The fallback spike exists at 7 airports: LIRF 16.6 %, LEBL 5.8 %, LTFM
   4.7 %, EDDM 4.2 %, LEMD 3.9 %, EGLL 3.8 %, LFPG 3.2 %. EDDF, EHAM and LSZH
   have none (T8). The arrival side shows the same rates in 2026 (T9).
4. At LIRF, `sd > 70,000` means `y` between 87,002 and 131,167 on all 6
   rows of 2025, with or without a flight record (T16). At the 9 other
   airports the same `sd` means a normal taxi (251 rows, median 1,111).
   One missed row of this class costs 21,000 MSE, +34 s live; v39 paid it.
5. Rows with a null flight record and `sd > 14,400` outside LIRF are normal
   taxis: 0 fallback and 0 24-h among 411 rows of 2025 (fourteenth-pass
   check). EHAM shows 138 such rows in January 2026 against 91 in all of
   2025. Drift to monitor, not a head to build.
6. `mvt_eobt1 = MVT_TIME - EOBT_1_flt` is the strongest single column and
   the anchor of the tempo family. Coverage 98.5 % on the 2026 rows.
7. Other flights' `EOBT_1` and `MVT_TIME` are present on every 2026 row and
   the organiser permits features built from them (RECAP, Discord
   2026-09-09). Arrivals carry their `BLOCK_TIME` in 2026.
8. Structure is stable from 2025 to 2026 (T10) except the EHAM null-record
   share (1.97 % to 3.16 %) and EDDM stands unseen in 2025 (4.55 % of rows).
9. External coverage in 2026: METAR complete; the Eurocontrol pre-departure
   family is 0 % in July 2026; OPDI taxi records exist in every month of
   both years at LSZH only (appendix A3).
10. Multi-threaded `linear_tree` training is not bit-exact, but the paired
    control of `train_r_all_v40.py` reproduced the shipped v26 members to
    the iteration. The cached frame and the fixed split make a paired run
    a 25-minute job.

## 4. Levers, priced and reasoned

Each lever states the mechanism, the evidence, the expected MSE on the
hold-out scale, the cost, the risk and the gate. Grades: A measured on the
paired harness or live; B measured on a partial frame; C arithmetic ceiling;
D reasoning only.

### 4.1 Information levers

**L1. Purified retune of the base.** Mechanism: `BEST_PARAMS` came from a
tuner that clipped `sd`, censored `y` and scored on the hold-out; the leaf
count was halved by hand for the linear tree; `linear_lambda` never moved.
The base has 110 columns now and a different loss surface. Evidence: no
paired tune exists (M3 of the twelfth pass). Expected: 600 to 1,800 MSE,
grade D. Cost: 20 to 40 trials of 5 min on the cached frame, one seed each,
blocked stop months, then 3 seeds paired. Risk: a tuned model that fits the
stop months. Gate: paired against v40 on months 1 and 7, at least 500 MSE on
the clean class outside LIRF, no airport worse by 500.

**L2. Refit on all 12 months for the final ship.** Mechanism: the base has
never seen January or July; the scoring months are January and July. Winter
holds, de-icing, summer peaks and the runway use of those months exist only
in the hold-out. Evidence: v14 tried a 12-month refit in the v13 era and
landed inside the noise; that stack is gone. Expected: 500 to 2,000 MSE,
grade D. Cost: one refit of every member with the iteration counts of the
paired run scaled by 1/0.88. Risk: no hold-out remains for the refit itself.
Gate: a proxy on the harness first: train on months 3 to 6 and 9 to 12
against the current 2 to 6 and 8 to 12, both scored on months 1 and 7; the
gain from the adjacent months bounds the gain from the same months. Ship
the refit as one upload and read the live delta as the price.

**L3. Richer tempo from the same anchors.** Mechanism: the 13 columns hold
medians and means; the shape of the neighbour distribution carries more.
Candidates: 25th and 75th percentiles of neighbour `mvt_eobt1`; the same on
`mvt_iobt`; tempo per stand prefix (terminal); tempo per operator over 2 h
(the quick test gave it 0.7 % of gain share); the arrival taxi-in residual
against the airport's 2025 median (drift-corrected live tempo); runway-use
shares over the previous 30 min (configuration state). Evidence: the family
delivered 2.55 s live; the residual clean error is 46,381. Expected: 400 to
1,200 MSE, grade D. Cost: one column build in `src_v2/frame.py`, one paired
run per group. Risk: collinearity with the served columns; the v42 forward
order gave -93 MSE. Gate: paired, at least 300 MSE per group or drop it.

**L4. Model-class diversity in the base.** Mechanism: a constant-leaf
booster and a linear-leaf booster make different errors; the mean of two
unbiased models with correlation 0.8 cuts the variance term by 10 %.
Evidence: the v39 constant-leaf base scored CLEAN 264.26 alone against
260.13, on a different column set; no paired mean exists. The ninth pass
closed "XGBoost or CatBoost in the base" without a recorded number.
Expected: 300 to 1,200 MSE, grade D. Cost: 3 members of a second class on
the cached frame, 30 min, then the paired mean. Risk: the weaker class
drags the mean; use a fixed 0.7 / 0.3 weight, never a weight fitted on the
hold-out. Gate: paired mean against v40, at least 400 MSE.

**L5. LIRF regressor on the 5-second definition.** Mechanism: `R_norm`
trains on rows with `abs(y - sd) >= 60`, which drops 3.8 points of LIRF
rows that are punctual genuine taxis (T8) and keeps a biased sample.
Evidence: the definition change is measured on the label (D1 of the twelfth
pass); its effect on the regressor is not. Expected: 100 to 500 MSE, grade
D. Cost: one paired run of `train_r_norm_lirf_v41.py` with `FB_TOL = 5` on
the genuine mask only, 5 min. Risk: the gate keeps its 60-second target;
the mixture stays consistent because the gate serves probabilities, not
rows. Gate: paired LIRF head, at least 200 MSE.

**L6. OPDI live tempo at LSZH.** Mechanism: actual taxi-out times of other
flights in the previous 30 to 120 min, from ADS-B events, with a 600 s lag
so the row's own event stays out. Evidence: `features_opdi_live.py` exists;
the base dropped it in v26 because coverage collapsed at other airports.
Appendix A3: LSZH holds 5,630 to 12,588 records in every month of both
years; every other airport has months near zero. Expected: 200 to 500 MSE,
grade D, on LSZH's 2,691 clean MSE. Cost: one column group gated to LSZH,
NaN elsewhere, paired run. Risk: a 2026 coverage change at LSZH; the
coverage monitor (H4 below) guards it. Gate: paired, at least 200 MSE, and
the LSZH coverage in January and July 2026 within 20 points of 2025.

**L7. Weather at the estimated off-block time.** Mechanism: every weather
column joins at `MVT_TIME`, the end of the taxi. De-icing and low
visibility act at pushback. `EOBT_1` estimates the pushback within about 20
min on most rows. Expected: 100 to 400 MSE, grade D, on winter rows. Cost:
one extra as-of join at `EOBT_1` for temperature, visibility, precipitation
and the de-icing gate. Gate: paired, January hold-out rows at least 300 MSE.

**L8. Calendar.** Mechanism: public holidays and school holidays change the
traffic mix; the model reads weekday and month only. Source: the
`holidays` package (MIT) for the 7 countries. Expected: 0 to 300 MSE, grade
D. Cost: minutes. Gate: paired, 200 MSE or drop.

**L9. Planned-time residual from `ARVT_1_flt`, clipped.** Mechanism: the
planned block time minus the route median isolates the planned taxi. The raw
form collapsed a seed in v38. Expected: 0 to 500 MSE, grade B from the
twelfth-pass quick test. Cost: one column already in `src_v2/frame.py`.
Gate: paired, no seed under half the median iteration, 300 MSE or drop.

### 4.2 Structural levers

**S1. Seeds and members.** Seed averaging beyond 3 base members and 5 LIRF
members is priced by the ambiguity rule and lost twice live (v32, v37).
Closed. The final refit keeps 3 and 5.

**S2. Per-airport specialists.** A LIRF specialist matched the pooled model
(RECAP). The pooled base reads the airport as a categorical and as a key of
every encoder. Closed unless L1 shows per-airport parameter sensitivity.

**S3. Stacking.** A meta-learner over the base and the head adds a fitted
weight and a leak path. Closed.

### 4.3 Label and regime levers, set aside with numbers

| lever | price | status |
|---|---|---|
| gate `p_fb` refit, any form | +94 to +396 MSE on four attempts (v34, v35, v37, thirteenth pass) | closed |
| fallback heads at the 6 non-LIRF spike airports | ceiling 1,753; the v39 heads scored 1,850 on the same rows | closed until a paired gain |
| hold rule from `mvt_eobt1` | -254 MSE at `> 7,200`, losses below | set aside |
| forward take-off order on the base | -93 MSE | set aside |
| tail head at the 9 airports | 0 fallback and 0 24-h rows in the null-record class outside LIRF | no target |
| Step A smoothing or extension | +95 to +375 MSE (ninth pass) | closed |
| the two LFPG rows | no observable | accepted |

### 4.4 The ledger

| lever | expected MSE, low | high | grade | cost |
|---|---|---|---|---|
| L1 retune | 600 | 1,800 | D | 3 h machine |
| L2 12-month refit | 500 | 2,000 | D | 1 h, live-priced |
| L3 richer tempo | 400 | 1,200 | D | 2 h |
| L4 second model class | 300 | 1,200 | D | 1 h |
| L5 `R_norm` on the 5-second mask | 100 | 500 | D | 10 min |
| L6 OPDI at LSZH | 200 | 500 | D | 1 h |
| L7 weather at off-block | 100 | 400 | D | 1 h |
| L8 calendar | 0 | 300 | D | 20 min |
| L9 plan residual | 0 | 500 | B | 30 min |
| sum | 2,200 | 8,400 | | |

The gap is 5,489. The midpoint of the ledger is 5,300. Two levers, L1 and
L2, carry half of the range and neither has a measured number yet. Run
those two first.

## 5. The build programme

Order by expected MSE per hour, with the dependencies. Every step ends in
a paired report on the cached frame and, for a ship, in a label-free 2026
check. One upload per step.

### 5.1 Harness, before any lever

- H1. Rebuild the cached frame with the movement id inside it, all 110
  served columns, the label, `sd`, the month and the airport. Today the id
  sits in a side file (`models/v36_tune_cache.ids.parquet`) joined by a
  composite key that leaves 169 rows unmatched.
- H2. One script scores any stack end to end on months 1 and 7 and writes
  the class table per airport. `src/eval_v33_holdout.py` does it for v33;
  generalise it to read a member list per component.
- H3. One script runs the label-free 2026 checks on any file against the
  best file: per-airport mean shift, count of rows over 7,200 and 80,000,
  rows at the clip, rows that differ per airport. `predict_v30.py` prints
  parts of it; the check exists only as session commands today.
- H4. The coverage monitor of the twelfth pass: per column, non-null and
  non-zero share for 2025, January 2026 and July 2026, per airport; fail on
  a drop over 20 points. Extend it to the OPDI record counts of appendix A3.
- H5. A test that the default `predict_v30.main` still rebuilds v33 to
  0.0 s, run before every upload.

### 5.2 Steps

1. **L1 retune** (3 h). Optuna, 30 trials, one seed, `linear_tree` on and
   off in the space, blocked stop months 11 and 12, hold-out untouched.
   Then 3 seeds paired. Ship as v43 if the gate passes.
2. **L5 `R_norm` mask** (10 min). Paired LIRF head. Ship as its own upload
   if over 200 MSE, else fold into the next LIRF change.
3. **L3 richer tempo** (2 h). Build the groups in `src_v2/frame.py`, one
   paired run per group, keep the groups over 300 MSE, ship the union.
4. **L4 second model class** (1 h). Paired mean at 0.7 / 0.3. Ship if over
   400 MSE.
5. **L6, L7, L8, L9** (3 h together). Each paired; ship the union of the
   ones that pass, as one upload, with the per-lever prices recorded.
6. **L2 12-month refit** (1 h). Last, because it removes the hold-out. Run
   the proxy of L2 first. Refit every member of the shipped stack on all
   12 months with the scaled iteration counts. Upload. The live delta is the
   price.
7. **Stop or continue** by section 7.

### 5.3 What every step records

The paired report JSON under `models/`, the price in `RECAP.md`, the gate
log in this file, the live score in `submission/`. The README names the
current best and the scripts that build it.

## 6. Data sources

Every source is open, declared here, and cited in the README. The
organiser permits any open dataset that the documentation declares (RECAP,
Discord 2026-09-09). Raw OpenSky state vectors through Trino are not
permitted (Discord 2026-09-11).

| source | content | licence | URL | status |
|---|---|---|---|---|
| PRC Data Challenge 2026 | 2025 movements at 11 airports, 2026 ranking set | challenge terms | https://ansperformance.eu/study/data-challenge/dc2026/data.html | in use |
| OPDI, Open Performance Data Initiative (PRC and OSN) | ADS-B derived flight events: pushback, taxiway and runway entries, taxi-out records | CC BY 4.0, confirmed by the organiser | https://www.opdi.aero/ | other flights only; LSZH tempo is L6; the scored flight's own events stay out |
| Iowa Environmental Mesonet, ASOS archive | METAR per station, 2025 and 2026 complete at every airport | public domain | https://mesonet.agron.iastate.edu/request/download.phtml | in use; L7 adds a second join time |
| OurAirports | runway ends, headings, coordinates | public domain | https://ourairports.com/data/ | in use |
| OpenStreetMap through Overpass | parking positions, taxiways, runways; stand-to-runway paths | ODbL | https://overpass-api.de/ | in use |
| EUROCONTROL PRU performance data | daily ATFM delay, pre-departure delay, airport traffic per airport | EUROCONTROL open data | https://ansperformance.eu/data/ | dropped from the base in v26 for 0 % July 2026 coverage; the arrival ATFM column stays |
| EUROCONTROL Aviation Data for Research | flight lists with off-block and take-off times, 2015 to 2022, four months per year | research licence, registration | https://www.eurocontrol.int/dashboard/rnd-data-archive | candidate for taxi-time priors per airport, stand area and hour; does not cover 2025 or 2026; grade D |
| Copernicus ERA5 | hourly reanalysis: precipitation, snow, wind, visibility proxies on a grid | Copernicus licence, free | https://cds.climate.copernicus.eu/ | candidate only where METAR has gaps; METAR has none at the 10 airports |
| `holidays` Python package | public holidays per country and region | MIT | https://github.com/vacanza/holidays | L8 |
| VRS StandingData | aircraft type metadata | open | https://github.com/vradarserver/standing-data | tried, redundant with the type categorical |
| OpenSky Network | raw ADS-B, Trino access | | https://opensky-network.org/ | not permitted for the challenge |

What no open source gives: the actual pushback time of the scored flight,
the ATC sequence at the holding point, the CTOT of a regulated flight, and
the ground-handling state of the stand. The tempo columns are the closest
proxy for all four.

## 7. Risks and stop rules

- **The hold-out is not the scoring set.** The 2026 set holds about 17,000
  MSE of error that no 2025 row shows (section 2.2). A hold-out gain
  transfers when it comes from the clean class; the price-landing record
  supports that. A gain on the tail or 24-h class does not transfer by
  itself.
- **Base changes carry the v27, v28, v32 record.** Those were seed-count
  changes. Feature changes on the paired harness landed twice. Keep the
  paired control in every run.
- **The stop months.** Every early stop reads months 11 and 12 or a random
  12 %. A retune that reads the same months can overfit them. L1 keeps
  the hold-out untouched and reports it once.
- **Stop rule.** After steps 1 to 5 of section 5.2, sum the paired prices
  of the shipped changes. If the sum is under 3,000 MSE, the model does not
  reach 290 s from 2025 labels and the levers of section 4 with these data
  sources; record that, ship L2 as the last change, and spend the remaining
  slots on the documentation and the paper, not on the lottery.
- **Ethics.** No `AOBT_3_flt`, no `LOBT_flt`, no own-flight OPDI event, no
  row value from the leaderboard. A change that moves a handful of rows by
  tens of thousands of seconds needs a 2025 row that supports it, as the
  ITY340 rule has.

## Appendix

### A1. Label classes and fallback share per airport, 2025

| airport | rows | fallback `<= 5 s` | clean median | clean p99 |
|---|---|---|---|---|
| EDDF | 230,141 | 0.6 % | 838 | 1,823 |
| EDDM | 167,334 | 4.6 % | 775 | 1,929 |
| EGLL | 239,546 | 4.5 % | 1,320 | 2,701 |
| EHAM | 247,951 | 0.7 % | 742 | 1,781 |
| LEBL | 179,705 | 5.1 % | 906 | 1,916 |
| LEMD | 212,242 | 4.1 % | 985 | 1,931 |
| LFPG | 239,552 | 3.8 % | 955 | 2,448 |
| LIRF | 160,704 | 16.6 % | 1,018 | 2,634 |
| LSZH | 134,907 | 0.6 % | 713 | 1,711 |
| LTFM | 272,965 | 4.7 % | 964 | 2,631 |

Tail rows in 2025: 139. 24-h rows: 14, all with a null flight record, 12 at
LIRF. Source: T2, T6, T8 at `b193868`.

### A2. Paired prices measured on 2026-09-13, hold-out scale

| change | class moved | MSE | live |
|---|---|---|---|
| 13 tempo columns on the base (v40) | clean outside LIRF | -906; net -805 | -1,146 |
| same columns on `R_norm_LIRF` (v41 members) | LIRF clean | -255; net -228 | shipped with the next row |
| no 4,431 s cap on `R_norm_LIRF` | LIRF tail and clean | -187 | v41 live -391 for the pair |
| tempo columns on the gate | LIRF | +94 | rejected |
| forward take-off order on the base (v42 members) | clean outside LIRF | -191; net -93 | set aside |
| hold rule `max(pred, 0.85 * mvt_eobt1)` at `mvt_eobt1 > 7,200` | tail | -254 | set aside |

Sources: `models/lgbm_r_all_v40.holdout.json`, `models/lirf_regime_v41.holdout.json`,
`models/lirf_regime_v41.gate_holdout.json`, `models/lgbm_r_all_v42.holdout.json`.

### A3. OPDI taxi-out records per airport and month

Records in `external/opdi/taxi_out_v2.parquet`, a pushback event paired with
a runway entry. Months of 2025 January to December, then 2026 January to
July:

| airport | 2025 | 2026 |
|---|---|---|
| LSZH | 7,542; 8,083; 7,725; 9,747; 9,306; 11,226; 11,770; 10,568; 11,550; 7,344; 8,638; 9,753 | 5,630; 8,174; 10,198; 11,305; 11,350; 11,618; 12,588 |
| EDDF | 7; 10; 18; 34; 21; 852; 615; 22; 4,979; 6,065; 7,695; 8,703 | 5,582; 859; 23; 3,853; 10,122; 10,208; 11,388 |
| EDDM | 1,762; 1,149; 872; 508; 481; 1,547; 1,547; 1,470; 410; 756; 1,010; 0 | 1; 1; 0; 296; 421; 75; 995 |
| LEBL | 6,912; 7,115; 6,089; 632; 509; 680; 458; 430; 363; 62; 12; 5 | 6; 23; 10; 13; 25; 39; 58 |
| LEMD | 0; 0; 1; 1; 0; 0; 0; 1; 1; 3,832; 10,090; 448 | 1; 1; 2; 2; 4; 18; 64 |
| EHAM, EGLL, LFPG, LIRF, LTFM | under 310 in every month | under 65 in every month |

Runway-entry events alone are broader (T14) and were redundant with the
movement counts (v18). Only LSZH supports a live-taxi tempo in every
scoring month.

### A4. The 2026 ranking set

344,841 departure rows at the 10 airports, 152,719 in January and 192,122
in July; every template row matches one of them. `BLOCK_TIME` and the label
are null on all of them. LIRF rows with `sd > 70,000`: 3. LIRF rows in the
Step A cell: 43. Rows outside LIRF with `sd > 70,000`: 51, normal taxis by
fact 4.
