# Audit prompt: sixteenth pass, the path from 281.87 s toward 224.88 s

Give this file to the auditor as the full task. The auditor is a person or
an AI agent with read access to this repository and the data machine. The
auditor writes the result as the sixteenth pass of
`docs/MODEL_ANALYSIS.md`. Follow `AGENTS.md` for code and prose.

## 1. Goal

Find the changes that move the live score of kind-mango from 281.87 s
(v67) toward the leaderboard top of 224.88 s. Rank each change by its
expected live gain. For each change, give a test that can reject it before
an upload.

## 2. Units and the gap

- Live MSE is RMSE squared on 344,841 rows. 281.87 s is 79,448 MSE.
  224.88 s is 50,571 MSE. The gap is 28,876 MSE.
- At 282 s, 1 s of RMSE is about 564 MSE.
- One row missed by 1,000 s costs 2.9 MSE. One 24-h row costs about
  21,600 MSE.

## 3. Hard rules

1. Do not read `AOBT_3_flt` or `LOBT_flt` in any model, feature or rule.
   This is a team rule (`README.md`, Ethics; `MODEL_ANALYSIS` 4.9, MX1).
   The organiser allows the fields (ruling of 2026-09-18). Only the team
   can change the rule. Section 6, task E prices that change as a scenario.
   Do not build it.
2. Do not set a row value from a leaderboard score. Do not select between
   files that differ only on a few rows. Do not probe the ranking.
3. Price every change on the 2025 hold-out (January and July) before an
   upload. Compare arms at equal fixed rounds. An early-stopped control
   can stop short and inflate the price (v71: -1,861 against -560 MSE).
4. Set the acceptance bar before the run: 150 MSE on the hold-out scale
   (344,336 rows). A pass is necessary, not sufficient.
5. Run one full-frame fit at a time. Two parallel fits exhaust 32 GB of RAM.
6. Timestamps in the organiser parquets are `datetime64[us]`. Convert with
   `(ts - Timestamp(0, tz="UTC")) // Timedelta(seconds=1)`, never with
   `astype("int64") // 10**9`.
7. The deadline is 2026-10-11, 23:59:59 CET. Upload before 21:00 UTC on
   that day. The limit is 5 uploads per UTC day.

## 4. Inputs

Read these first, in this order:

1. `docs/MODEL_ANALYSIS.md`: the Progress section (every lever since v48
   with its live result) and sections 4 and 5 (findings and measures).
2. `docs/WINNING_PLAN.md`: the ledger, the gap analysis, competitor code
   and the policy tracks A, M and B.
3. `RECAP.md`: one row per upload with its mechanism and result.
4. `REPRODUCE.md`: the v67 stack and the producer of each artefact.
5. `src_v3/predict_v67.py`, `src_v3/predict_v57.py` and `src/predict_v30.py`:
   the served path.

Saved evidence for fast tests:

- `models/v3/catboost_pair_holdout.parquet`: out-of-sample predictions of
  the v46 LightGBM members and the v67-recipe CatBoost on the 317,808
  non-LIRF hold-out rows (`y`, `v46`, `cat_a`, `cat_b`, `MVT_ID_mvt`).
- `models/v3/*.holdout.json`: the priced levers of 2026-09-24 to 26.
- `submission/kind-mango_v*.check.json`: label-free checks of each upload.

## 5. What is known

### 5.1 The served stack (v67)

- Base outside LIRF: 0.5 x mean of three LightGBM members (v65 recipe,
  118 columns, MB3 rows, 12 months) + 0.5 x CatBoost (depth 8, 7,385
  rounds).
- LIRF: `p_fb * sd + (1 - p_fb) * R_norm`, with the v56 gate and the
  out-of-fold isotonic map of v64; Step A on null-record rows with
  `sd > 14,400` reads `R_norm` (v63); the ITY340 hedge on `sd > 70,000`.
- Post-processing: MS1 per-airport caps, MS2 floors, MS4 checks.

### 5.2 Error budget of the served base on the hold-out

The served-style base is 0.5 v46 + 0.5 CatBoost A, on non-LIRF rows. Total
76,632 MSE, hold-out scale.

| class | rows | MSE | RMSE in class |
|---|---|---|---|
| clean | 293,832 | 41,368 | 220 s |
| 24-h (`y > 80,000`) | 1 | 20,235 | one LFPG row |
| tail (`7,200 < y <= 80,000`) | 35 | 12,100 | 10,004 of it at LFPG |
| fallback (`abs(y - sd) < 60`) | 23,841 | 2,836 | 202 s |
| low (`y < 30`) | 99 | 94 | |

In the clean class, rows with an error above 1,800 s hold 11 % of the MSE.
The rest is spread over normal rows. A clean-class RMSE of 150 s instead of
220 s is worth about 22,100 MSE on the hold-out scale
(`293,832 x (220^2 - 150^2) / 344,336`).

### 5.3 Closed levers: do not repeat

| lever | result |
|---|---|
| anchored offset on `mvt_eobt1` (MB1) | +35.86 s live |
| 12-month or out-of-fold operator encoders (MB8, MD4) | +1.29, +2.25 s |
| calendar block (MB4), arrival drift (MF4), weather at EOBT_1 (MF1) | +0.11, +0.37, +0.33 s |
| revert of the Optuna retune (L4) | +0.99 s |
| MS3 member median on the MB3 base | moves 0 rows |
| LIRF head without drifted columns (MD2), CatBoost in `R_norm` | +0.42, +0.03 s |
| converged or second base CatBoost | +92 and +71 MSE hold-out, under the bar |
| MF2 runway-queue columns | hold-out -560 MSE, live -0.015 s |
| MH2 fallback heads outside LIRF | +292 MSE hold-out, worse at all airports |
| tail rules that pull outliers to the 2025 tail | +1,639 MSE hold-out |
| CatBoost blend weight 0.6 to 0.8 | -80 to +97 MSE hold-out |

### 5.4 Transfer from hold-out to live

| change | hold-out, MSE | live, MSE |
|---|---|---|
| base CatBoost (v67) | -2,079 | -1,627 |
| out-of-fold gate map (v64) | calibration only | -652 |
| `plan_nm_taxi` (v65) | not priced | -695 |
| CatBoost in `R_norm` (v68) | -435 | +17 |
| MF2 columns (v71) | -560 | -8 |

A new model class and defect fixes transferred. Feature additions and
LIRF-head model changes did not.

## 6. Tasks

### A. Confirm the budget

Recompute table 5.2 from the saved predictions. Add the LIRF rows with the
v67 head. Report the budget per airport and per class on the live scale.
State which part of the 28,876 MSE gap each class can close at best.

### B. Audit every served rule

For each rule below, state the current value, its source, the evidence
that set it, and one hold-out test that could change it. Mark each rule
"keep", "test" or "change".

- MS1 caps (2025 class maximum plus 600 s; the `mvt_eobt1 > 5,400` bar).
- MS2 floor (2025 clean 0.1 % quantile per airport).
- The fallback tolerance of 60 s in the gate label and the `R_norm` mask.
- The Step A band table (11 bands, `sd > 14,400`, null record only).
- The ITY340 constants (5/6, 86,400 + 1,150 s).
- The MB3 row filter (not LIRF, `y <= 80,000`).
- The blend weight 0.5 and the round-count scaling by row ratio.
- The clip at 0 and the NaN fill with the global median.
- The 169 null ids of the H1 side file (MODEL_ANALYSIS R2), kept for parity.

### C. Search for new signal inside the team rule

The clean class needs information about the actual off-block time that the
model does not see. For each candidate, state the data source, the 2026
coverage, the leak risk and a hold-out test. Start with these:

1. Other departures' take-off sequence on the same runway after the row's
   own EOBT_1 (backward windows only; MODEL_ANALYSIS L4 and MF6).
2. Arrivals in the ranking file (block and runway times are not blank):
   stand turnaround of the inbound aircraft, gate conflicts at the stand.
3. OSM taxi route: runway crossings and hold points (MF2 remainder).
4. Local hour, night restrictions and daylight saving (MF3).
5. Per-airport target transforms or class weights for the clean class.
6. Any open data source in `external/` that no served feature reads.

### D. Price the unmodelable rows

The one 24-h row and the 35 tail rows hold 32,335 of 76,632 hold-out MSE.
Rows with a null flight record hold 31,480 of it. Two LFPG null-record rows
(y 84,240 and 58,206 s, `sd` about 2,000 s, no `mvt_eobt1`) hold 29,722 and
show no observable. Other null-record tail rows sit near `sd` (EGLL y 11,167
against `sd` 10,563; LSZH 9,652 against 9,946), as at LIRF under Step A.

1. State whether any observable separates the tail rows in 2025.
2. Price a Step A analogue outside LIRF: null-record rows with a large `sd`.
3. State the MSE floor of the rows that stay unpredictable, and remove it
   from the reachable gap.

### E. Price the team-rule scenarios

Use `docs/WINNING_PLAN.md` sections 4, 6 and 8. Give the expected live MSE
and the risk of each scenario. Do not build a model for tracks M or B.

| track | reads | plan range, live |
|---|---|---|
| A (current) | no NM actual clocks | 279 to 284 s |
| M | other flights' `MVT - AOBT_3` only, backward windows | 273 to 281 s |
| B | the scored flight's `AOBT_3_flt` and `LOBT_flt` | 254 to 271 s |

State what evidence supports the gap between track B and 224.88 s.

## 7. Output

Write the sixteenth pass at the top of `docs/MODEL_ANALYSIS.md`, below the
Progress section. Include:

1. The budget of task A, per class and per airport, on the live scale.
2. The rule table of task B.
3. A ranked lever table: id, mechanism, track, evidence grade (A to D),
   hold-out test, pre-set bar, expected live MSE (low, high), hours, uploads.
4. The floor of task D and the reachable gap on each track.
5. The scenario table of task E, with the decision left to the team.
6. An upload plan to 2026-10-11 that respects the stop rule in
   `docs/WINNING_PLAN.md` 11.4.

Give every number a source: a file and line, a JSON key, a `RECAP.md` row
or arithmetic. Mark a number without a source "unmeasured".
