# Model analysis: progress to v137, the twenty-third pass (budget of v135, last probes), the twenty-second pass (coverage audit: no-record rows linked to their tracks), the twenty-first pass (live judging, neural-forward blend), the twentieth pass (timestamp precision, the LFPG off-grid regime), the nineteenth pass (audit of v115 and v116, NM join rules), the eighteenth pass (new observables, TabM), the seventeenth pass (CatBoost gate) and the sixteenth pass (error budget of v97)

Status on 2026-10-13, 09:25 UTC: the twenty-third pass gives the budget of v135 and closes the last probes; candidate v137 (member columns in the stack, hold-out -76 MSE) is built, not uploaded. Status on 2026-10-13, 08:55 UTC: live best **232.45 s (v135)**, -3.09 s against v132, accepted: the first-gap ADS-B segments in the stack, the TabM families and the plain tree members (one upload for K7, K7b and K7c). Status on 2026-10-13, 03:50 UTC: candidates v134 (TabM families refit on the first-gap tables, hold-out -523 MSE against v133) and v135 (plain tree members refit, -58 MSE more) are built with their final-frame files; no file is uploaded. The first-gap rows carry -527 of the -581 MSE (RECAP rows v134, v135). Status on 2026-10-12, 17:50 UTC: candidate v133 (not uploaded) gives the stack the ADS-B segments that end up to 3,600 s before the take-off; 4.3 % more ranking departures get a track and the hold-out price is -1,081 MSE (RECAP row v133). Status on 2026-10-12, 02:45 UTC: the final file is built with the v132 recipe (`release_final`, REPRODUCE section 8); on the shared January and July departures it differs from v132 by 6.6 s on average, the size of the new timestamp noise. Status on 2026-10-11, 19:58 UTC: live best **235.54 s (v132)**, -0.41 s against v131, accepted: the plain tree members refit on the v130 tables. Status on 2026-10-10, 21:55 UTC: live best **235.95 s (v131)**, -2.30 s against v130, accepted: the TabM families refit on the K8 tables. Status on 2026-10-10, 09:15 UTC: live best **238.25 s (v130)**, -1.47 s against v129, accepted: the lift-off link (F22.4). Status on 2026-10-10, 01:15 UTC: live best **239.72 s (v129)**, -0.87 s against v128, accepted: the TabM families refit on the K7 tables. The final phase (organiser e-mail of 2026-10-09) adds February and June 2026; the deadline is 2026-10-14T10:00Z. Status on 2026-10-09, 05:20 UTC: leaderboard best **240.60 s (v128)**, the stack tuned for the 2026 coverage mix, -0.11 s against v127. Status on 2026-10-09, 05:15 UTC: live best **240.70 s (v127)**, -0.95 s against v126, accepted: departures without an NM record now link to their own ADS-B and OPDI tracks by flight number (twenty-second pass, F22.1). Status on 2026-10-09, 00:10 UTC: leaderboard best **241.66 s (v126)**, a fifth TabM family, -0.01 s against v124; v125 (NM clocks in the LIRF gate) 242.35 s, rejected (21.3). Status on 2026-10-08, 20:50 UTC: leaderboard best **241.67 s (v124)**, a fourth TabM family, -0.12 s against v123. Status on 2026-10-08, 17:45 UTC: leaderboard best **241.79 s (v123)**, the TabM share 0.65, -0.06 s against v122. The twenty-first pass (below the Progress section) records the user directive of 2026-10-08: the live score judges each upload, no hold-out price first. Status on 2026-10-08, 17:00 UTC: leaderboard best **241.84 s (v122)**, `tabm_j` as a third TabM family without J3, -0.18 s against v121, 0.11 s under the bar (20.8). Status on 2026-10-08, 08:45 UTC: live best **242.03 s (v121)**, -0.49 s against v119, accepted (20.6). Status on 2026-10-08, 07:00 UTC: the twentieth pass (below the Progress section) finds one new regime: at LFPG a take-off time off the minute grid marks a departure with a long label. Rule P1 prices -250 hold-out MSE on 17 rows; 19 ranking rows carry the flag. v121 serves P1 with six TabM seeds per family (N10). No measured lever reaches 220 s (20.5). Status on 2026-10-08, 01:05 UTC: v120 (J3 with a third TabM member) 242.91 s, +0.40 s against v119, rejected; best stays **242.51 s (v119)** (19.9). Status on 2026-10-07, 23:05 UTC: leaderboard best **242.51 s (v119)**, J2 bundled with a periodic-embedding TabM member; -0.295 s against v118, 0.005 s short of the acceptance bar. Status on 2026-10-07, 16:00 UTC: live best **242.81 s (v118)**, -0.85 s against v116, accepted; team rank 34 of 235. v118 serves J1 of the nineteenth pass (the NM join window) on the v116 recipe: hold-out -371 MSE, live -411. Earlier the same day, 15:00 UTC: leaderboard best **243.65 s (v116)**, team rank 36 of 235; the top is 213.52 s. The nineteenth pass (below the Progress section) audits the served path and measures where the error of v115 sits. It finds one new structural rule, the NM join window `|BLOCK - LOBT| <= 3,606` s, that the pipeline does not use: hold-out -564 MSE, no retrain (`python -m src_v3.measure_nm_join`). No measured lever reaches 220 s (19.6). Status on 2026-10-06, 18:40 local time: leaderboard best **243.65 s (v116)**, X-Plane stands at five airports; -0.035 s against v115, inside noise. Earlier: live best **243.69 s (v115)**, X-Plane stands at EDDM and EHAM and ATFM slot columns (18.7). Earlier: leaderboard best **244.02 s (v114)**, the 32-epoch TabM member with three seeds at 0.4; -0.27 s against v112, under the 0.3 s acceptance bar. Earlier: live best stays **244.29 s (v112)**; v113 (TabM gate member) 247.40 s, rejected. Before: live best **244.29 s (v112)**, the 24-epoch TabM member at 0.35 plus the Step A date-crossing fix (18.5). Before: **245.23 s (v111)**, a TabM member at 0.3 on top of the v105 base (eighteenth pass). Status on 2026-10-05, 20:20 local time: live best stays **246.48 s (v105)**. The seventeenth pass names the strong and weak points of v105 and prices a CatBoost member in the LIRF gate (N7): hold-out -484 MSE, live v109 247.37 s, rejected; and NNLS member weights per coverage class (N3): hold-out +139 MSE, live v110 246.57 s, rejected. Status on 2026-10-04, 04:00 local time: live best **246.48 s (v105)**, LIRF gate g3 with 2026 arrival information plus the MLP on the v97 columns (Progress table). Status on 2026-10-03, 12:00 local time: live best **247.89 s (v97)**, team
rank 31; the leaderboard top is **216.57 s** (API, 09:25 UTC). The
sixteenth pass below the Progress section measures where the v97 error
sits. `docs/WINNING_PLAN.md` third edition turns it into the plan toward
210 s (lever ids G, K, N, Q).
Status on 2026-10-03, morning: live best **247.89 s (v97)**: every base member refit
with the OPDI and ADS-B pushback columns of 350 days of 2025, the v95 stack
on top. v98 (serve-time parameters re-chosen on the hold-out) landed 248.25 s.
Status on 2026-10-01, evening: live best **249.39 s (v95)**, the v94 recipe
with the stack on all 62 January and July 2025 ADS-B days. v96 (ADS-B
specialist column) landed 249.92 s and is rejected. Every section 5
measure now has a final state (Progress section).
Status on 2026-09-30, evening: live best **250.51 s (v94)**, v93 (stack on
21 ADS-B days of 2025, 251.44 s) plus the MLP member at 0.15.
Status on 2026-09-30: live best **253.82 s (v92)**, the v89 blend plus a
LightGBM stack that reads the pushback time of the scored flight from
OPDI (v90, 263.17 s) and ADSB.lol (v91, 259.39 s; v92 on all 62 ranking
days), floored at the last time at the assigned stand (A5). Lever ids A1
to C1 refer to the second edition of `docs/WINNING_PLAN.md`; L1 to L15
refer to the first edition (`git show 2a4ec1b:docs/WINNING_PLAN.md`).
Status on 2026-09-28: live best **264.34 s (v89)**, per-airport LightGBM
members mixed 50/50 into both LightGBM terms. v85 264.55 s: forward
windows of the NM clocks (L8) in both LightGBM bases. Before that, **265.27 s (v76)**, NM clocks of the
scored flight in both base classes plus an anchored member in each
(track B), LightGBM parameters from the MB5 study on the clock-era
columns. v75 266.15 s, v74 266.84 s, v73 270.58 s; before track B,
281.87 s (v67). Leaderboard top **224.88 s** (2026-09-26). The gap from
v76 is 19,799 MSE (70,370 against 50,571). This file is the
source of truth for the next model.

- The Progress section below tracks every measure of the fifteenth pass
  and every WINNING_PLAN lever to v98.
- The sixteenth pass (2026-10-03) follows the Progress section. It runs
  tasks A, C and D of `docs/AUDIT_PROMPT.md` on v97 instead of v67.
- Sections 0 to 9 and the appendix are the fifteenth pass (audit of v47,
  2026-09-17). They stay as the record of the findings that the measures
  cite.

Earlier passes stay in git:

- fourteenth pass (levers L1 to L9, plan under 290 s):
  `git show acb157a:docs/MODEL_ANALYSIS.md`
- twelfth and thirteenth passes (diagnostic tables T1 to T17, v39 debrief):
  `git show b193868:docs/MODEL_ANALYSIS.md`

## Progress, 2026-09-23: section 5 measures merged with `WINNING_PLAN.md`

Live best **247.89 s (v97)**, team rank 31 on 2026-10-03. On 2026-09-23 the rank was 57 of 178 at 287.33 s (v57). `docs/WINNING_PLAN.md` ranks
levers L1 to L15; this table maps them onto the measures of section 5.
Live deltas come from the official API (WINNING_PLAN section 3). On
2026-09-26 the team moved from track A to track B: the base reads
`AOBT_3_flt` and `LOBT_flt` of the scored flight from v73 on (README
stance section). Section 4.9 and MX1 record the earlier stance.

### Gap to the top, 2026-09-26

Error budget of the served-style base (0.5 v46 + 0.5 CatBoost A) on the
317,808 non-LIRF rows of the 2025 hold-out, hold-out scale. Source:
`models/v3/catboost_pair_holdout.parquet`, arithmetic.

| class | rows | MSE | RMSE in class |
|---|---|---|---|
| clean | 293,832 | 41,368 | 220 s |
| 24-h (`y > 80,000`) | 1 | 20,235 | one LFPG row |
| tail (`7,200 < y <= 80,000`) | 35 | 12,100 | 10,004 of it at LFPG |
| fallback (`abs(y - sd) < 60`) | 23,841 | 2,836 | 202 s |
| low (`y < 30`) | 99 | 94 | |
| total | 317,808 | 76,632 | |

1. The clean class is the only large mass that a model can move. A clean
   RMSE of 150 s instead of 220 s is worth about 22,100 MSE. Rows with an
   error above 1,800 s hold only 11 % of the clean MSE.
2. The 24-h row and the 35 tail rows hold 32,335 MSE; rows with a null
   flight record hold 31,480 of it. Two LFPG rows (y 84,240 and 58,206 s,
   `sd` about 2,000 s, no `mvt_eobt1`) hold 29,722 and have no observable:
   that is a floor. Some other null-record tail rows sit near `sd` (EGLL
   y 11,167 against `sd` 10,563; LSZH 9,652 against 9,946), the LIRF Step A
   pattern outside LIRF (audit task D).
3. On the team rule (track A) the levers left are small: WINNING_PLAN puts
   the track at 279 to 284 s, and v67 sits inside that range. Five levers
   ran after v67: two reached an upload and moved the live score by +0.03 s
   (v68) and -0.015 s (v71); three failed their hold-out bar (v69, v70,
   MH2).
4. The WINNING_PLAN range for track B (the scored flight's `AOBT_3_flt`) is
   254 to 271 s. The five public repos above rank 40 read that field. The
   rule stays a team decision (section 4.9, MX1).

The sixteenth pass must find new signal for the clean class or price the
team-rule scenarios. `docs/AUDIT_PROMPT.md` sets the tasks, the hard rules
and the output format.

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
| MH2 fallback head outside LIRF: `base + alpha p (sd - base)` with a calibrated classifier for `abs(y - sd) < 60` | v72, not built | hold-out +292 (alpha 1) / +72 (alpha 0.5) MSE | rejected, closed: worse at all 9 airports; outside LIRF these rows are mostly on-time pushbacks that the base already serves |
| L7 check: day-level correlation of departure fallback share and arrival schedule-grid share, 2025 | | | passes only at LIRF (0.43) and EDDM (0.34); the other 8 airports sit at 0.02 to 0.19 |
| L1 NM clocks of the scored flight in the LightGBM base (track B), MS3 on its members, 0.3 v67 CatBoost | v73 | -11.29 | accepted; hold-out -5,652 MSE against an equal-round control, all 9 airports; L2 neighbour clocks -5,026 (LFPG +733 against L1), not served |
| Clocks in the CatBoost member, and L5 anchored LightGBM member (`y - clip(MVT - AOBT_3, 0, 7,200)`) at 0.5 inside the LightGBM term | v74 | -3.73 | accepted; L5 hold-out -818 MSE against the clock member at 0.5, all 9 airports |
| L6 clocks in the LIRF `R_norm` term | not built | hold-out -39 to -118 MSE | closed: under the 150 MSE bar |
| Anchored CatBoost member at 0.5 inside the CatBoost term | v75 | -0.70 | accepted; no hold-out price, served by analogy with the L5 LightGBM member |
| Clock bias: median of `y - (MVT - AOBT_3)` per stand, stand-runway and operator | v76, not built | in-sample map +1,531, leave-one-month-out -51 MSE | closed: the categoricals already carry it; in-sample medians leak the label |
| `ARVT_3` residual; arrival `MVT - ARVT_3` offset per airport-hour | not built | correlation with the clock error 0.005 and -0.04 to 0.09 | closed: no signal |
| MB5 study on the clock-era LightGBM base (`linear_lambda` 15.1, 613 leaves), both bases refit | v76 | -0.88 | accepted; hold-out -965 MSE against the served-style member, all 9 airports |
| Per-airport LightGBM members, 50/50 with the global member in both LightGBM terms | v89 | -0.22 | accepted; hold-out mean -657 at all 9 airports; extended forward windows (5, 60 min) closed |
| L8 forward windows in both CatBoost members too | v87 | +0.31 | rejected; not priced on the hold-out first; lower CatBoost weights also closed |
| L8 forward windows: NM clocks of departures taking off in the next 15 and 30 min, both LightGBM bases | v85 | -0.72 | accepted; hold-out -604 member, -208 in the blend; queue lengths on top closed (-216 against -208) |
| Second MB5 study, v76 parameters as trial 0; XGBoost as a fifth member; blend weights with the MB5 members | not built | -539 member (LTFM -449, 5 airports worse); XGBoost +37; weights +63 | closed: the LightGBM parameters sit near the optimum and the member classes are saturated |
| Arrival artefact link in the LIRF gate (previous arrival at the stand on the schedule grid) | v83 | +0.14 | rejected; hold-out -277 MSE did not transfer (LIRF head, as v49 and v68) |
| Stand-sequence bound: next arrival in-block at the stand | v81, not built | +301 member, -2 blend | closed |
| CatBoost with `l2_leaf_reg` 10; CatBoost Lossguide | v80, not built | +54 MSE in the blend; Lossguide too slow | closed: v67 CatBoost parameters stay |
| MB5 study on the anchored LightGBM target; round count of v76a | v79, not built | same best parameters as v76; 1,202 rounds +124 MSE against 2,284 | closed: v76a is at its optimum |
| MB5 on the LIRF `R_norm` regressor (`linear_lambda` 44, 387 leaves) | v78, not built | -251 MSE on genuine rows, upper bound | held: LIRF-head model changes have not transferred (v49, v68) |
| Blend weights of the four clock-era members, grid-fitted per month | v77, not built | cross-fitted -2 MSE | closed: v75 weights at the optimum |
| Constant-leaf LightGBM as a fifth member | v77, not built | fitted weight 0 | closed: residual correlation 0.983 with the blend |
| Clocks in the LIRF gate | not built | hold-out mixture +108 MSE | closed: at LIRF `MVT - AOBT_3` is near 900 s on fallback and genuine rows alike |
| L15 final 12-month refit | not needed | | every served component already fits on 12 months; the target encoders stay at 10 months by the L15 rule |
| L2 neighbour clocks, backward 30 and 60 min | not built | hold-out +626 MSE against L1; -137 without 7 single-member blow-up rows | closed: under the 150 MSE bar |
| Discord tail rules: blend heavy-hold rows toward the 2025 curve of y on `mvt_eobt1`; cap normal rows | not uploaded | hold-out +1,639 / -3 MSE | rejected on the 2025 hold-out (`src_v3/measure_tail_rules.py`); see below |
| A1 OPDI pushback clock of the scored flight, LightGBM stack on `y - base` | v90 | -1.17 | accepted; hold-out -566 MSE, LSZH carries it |
| A2 ADSB.lol pushback clocks in the same stack, 24 July days | v91 | -3.78 | accepted; hold-out -937 MSE on 7 days of 2025 |
| A2 on all 62 ranking days, plus A5 stand bound `min(adsb_own_last, adsb_push)` after MS1 | v92 | -5.57 | accepted; bound hold-out -15.9 MSE, 3.3 % of 2025 labels below it; 1,184 rows lifted |
| A2 stack on 21 ADS-B days of 2025 (Jan-Jul cross-fit) | v93 | -2.38 | accepted; hold-out stack -1,575 against -937 MSE on 7 days |
| A2 base retrain: every base member with the OPDI and ADS-B columns (4.5, last step) | v97 | -1.50 | accepted; hold-out blend plus stack -983 MSE against v95's recipe |
| Serve-time parameters re-chosen on the v97 hold-out members | v98 | +0.36 vs v97 | rejected; hold-out about -120 MSE |
| N1 stack on stand-clock rows only | v99 | +0.32 | rejected; hold-out -175 MSE; the stack helps segment rows live |
| K2 ADS-B columns in the LIRF `R_norm` | v100 | +0.12 | rejected; hold-out -267 MSE on genuine rows |
| G1 airline arrival artefact shares in the LIRF gate | v101 | -0.20 | accepted; hold-out -52 MSE |
| G1 plus the v83 stand link in the gate | v102 | -0.55 | accepted; hold-out -323 MSE |
| 63-leaf stack | v103 | +0.20 | rejected |
| gate g3: plus operator arrival shares | v104 | -0.19 | accepted; hold-out -235 MSE |
| N4 MLP on the v97 columns at 0.2 | v105 | -0.48 | accepted, best 246.48 s; hold-out -64 MSE |
| gate g4: plus exact-copy and stand-day shares | v107 | +0.21 | rejected; hold-out -379 MSE |
| K1 extra ADSB.lol feeds, K3 route distance, K4 long holds, N5 pseudo-labels, N2 study | not uploaded or pending | | K1 +0.6 points coverage, K3 -28, K4 0 of 4 rows, N5 +7 MSE; N2 refit uploads as v108 |
| N8 TabM member on the v97 columns at 0.3 on top of the v105 base (eighteenth pass) | v111 | -1.25 | accepted; hold-out full path -231 MSE |
| N8b TabM with a 24-epoch schedule at 0.35, plus the Step A date-crossing fix | v112 | -0.94 | accepted, best 244.29 s; hold-out -89 MSE against v111, plus one LIRF row |
| N9 TabM member in the LIRF gate (g3t), mean with g3 | v113 | +3.12 | rejected, closed: hold-out -319 MSE; LIRF rows mean +37.1 s |
| N8c TabM 32 epochs, three seeds, at 0.4 | v114 | -0.27 | under the 0.3 s bar; hold-out -60 MSE against v112 |
| K5 X-Plane stands at EDDM and EHAM, ATFM slot columns in TabM | v115 | -0.33 | accepted, best 243.69 s; hold-out -33 MSE against v114 |
| K5b X-Plane stands also at LEMD, LEBL, LFPG | v116 | -0.035 | inside noise; 2026 stand-clock share LEMD +4.4, LEBL +2.1 points |
| K5c tree members refit on the v116 clock tables, MS3 guard on the per-airport mix | v117 | +0.09 | rejected: hold-out blend -232 MSE against v97 members, but the stack already read the new clocks |
| J1 NM join window: LIRF gate 0 where a schedule copy breaks the window, then the LOBT window as the last step (nineteenth pass) | v118 | -0.85 | accepted, best 242.81 s; hold-out -371 MSE, live -411 (1.1 times); 1,123 rows moved |
| J2 arrival-side `LOBT` for LIRF rows without a record, bundled with N10 `tabm_plr` (periodic embeddings) for rule 12 | v119 | -0.295 | leaderboard best 242.51 s, under the 0.3 s bar by 0.005 s; hold-out -193 (J2) and -83 (member), live -143 |
| J3 anchor on rows with `LOBT` 2 to 4 h before take-off, bundled with `tabm_j` at the TabM share 0.5 | v120 | +0.40 | rejected, closed: hold-out -255 and -17, live +194 |
| ADS-B specialist (4.5 step 1) as a stack column, trained on 291 days outside months 1 and 7 | v96 | +0.53 | rejected; hold-out -302 MSE did not transfer (a derived column, like v66, v71, v83) |
| v96 study on 62 days: 63 leaves, residual clip 7,200 s, A5 offsets, A4 columns | not built | -92, +19, +29 to +523, +37 MSE | closed: under the bar or worse |
| A2 stack on all 62 January and July 2025 ADS-B days | v95 | -1.12 | accepted; hold-out stack -4,048 MSE; unattended release |
| MLP member at 0.15 in the base outside LIRF | v94 | -0.93 | accepted; hold-out about -224 MSE, live -470 |
| A3 floor at the first moving ADS-B point | not built | hold-out +4,875 to +7,954 MSE | closed: the first point often moves before the reported off-block (EHAM 41 %); the stack already reads it |
| A4 neighbour pushback clocks, backward 30 and 60 min, forward 15 and 30 min | not built | +2 (columns), +62 (neighbour-only rows) | closed |
| A6 pushback clock in the LIRF term | not built | | reopened 2026-10-03 as third-edition K2: the 0 % came from one day; over 62 days 29 % of 2026 LIRF departures carry an ADS-B segment (F16.7) |
| B1 Step A analogue outside LIRF | not built | 0 of 412 null-record rows with `sd > 14,400` sit on `sd` | closed on measurement |
| B2 3 seeds for the per-airport members | not built | hold-out -5.2 MSE in the blend | closed: under the 150 MSE bar; airports split (EGLL +3.1, LTFM -4.2) |
| MF3 local hour, MD5 stand and runway scoped to the airport, MD6 drift flags, as residual-stack columns on all hold-out rows | not built | Jan-Jul cross-fit: control stack +535, with the columns +723 to +1,267 MSE | closed: no signal the base misses |
| C1 reproduction path | | | `test_parity.py` (v67, v92), `fetch_metar.py`, v85 forward files, `REPRODUCE.md` for v92 |
| MC4 pin library versions | | | `requirements.txt` (Python 3.13.3); fixed thread counts in `config.py` |
| MC6 parity test for the best file | | | `src_v3/test_v67_parity.py`: v67 rebuilds bit for bit against `submission/kind-mango_v67.parity.json`; `REPRODUCE.md` rewritten for v67 (two tiers) |
| R2 id side file of the H1 frame | | | `tune_lgbm_v36.py build` now writes `v36_tune_cache.ids.parquet` with the 169 historical null rows (`v36_tune_cache.null_ids.json`); a scratch rebuild matched the historical cache, feature list and ids exactly |
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

### Final status of every section 5 measure, 2026-10-01

No section 5 measure is open. Each one is served, rejected live, closed on
a hold-out or measurement, or set aside with the reason below. The best
file is v95 (249.39 s).

| measure | state | evidence or reason |
|---|---|---|
| MP1 month-blocked folds | partial | `src_v3/folds.py`; the stack cross-fits January against July. The base members keep the (1, 7) hold-out plus the live reading. |
| MP2 decide on intervals | replaced | a pre-registered 150 MSE hold-out bar per lever; the null spread of near-null arms is 89 MSE (A4) |
| MP3 same-run control | served | equal-round controls, `deterministic`, fixed threads |
| MP4 artefacts inside the fold | partial | out-of-fold gate map (v64); out-of-fold encoders rejected live (v60) |
| MP5 select on served rows | served | MB3 rows; MB5 study on the served objective (v76) |
| MP6 arrival mirror | built | `src_v3/arrival_mirror.py`, used as a drift meter |
| MP7 label-free gates | served | `check_submission_2026.py`; `auto_release.py` blocks an upload on a failed gate |
| MP8 strict coverage monitor | partial | `src_v3/coverage.py` |
| MP9 read every score | served | |
| MP10 one module for the classes | set aside | reports compute classes per script; no score effect |
| MD1 fallback class per airport | measured, no change | residual grid 2025: spikes at 0 to 6 s and at exactly 60 s (about 2.5 times the neighbours) at 7 airports, flat at EDDF, EHAM, LSZH. The LIRF `< 60` label takes the 46 to 59 s band (2.2 % of LIRF rows) and misses the 60 s spike (0.26 %). A relabel changes the LIRF head only; LIRF-head changes did not transfer (v49, v68, v83). |
| MD2 drifted columns out of the head | rejected | v49, +0.42 s |
| MD3 NaN for absent context | set aside | needs four base refits (about 10 h); MB5 raised `linear_lambda` to 15.1, which damps the extrapolation that M2 describes |
| MD4 out-of-fold encoders | rejected | v60, +2.25 s |
| MD5 airport-scoped stand and runway | closed | residual stack, Jan-Jul cross-fit: worse than the control |
| MD6 drift flags | closed | same test |
| MD7 one taxi-in meter | set aside | arrival drift features rejected live (MF4, v59) |
| MD8 one tempo pool | set aside | six columns, train-serve pool difference only; no base refit for it |
| MD9 replace `ades_arr_atfm_delay_today` | set aside | 4 to 10 % row coverage |
| MD10 month-start context | set aside | at most the first 12 h of 1 July and 1 January; 30 June 2026 is not in the ranking file |
| MB1 anchored offset on `mvt_eobt1` | rejected | v54, +35.86 s; the `AOBT_3` anchor (L5) is served instead |
| MB2 conditioned linear leaves | partial | MB5 `linear_lambda` 15.1 (was 0.0056); no standardisation |
| MB3 served rows | served | v51 |
| MB4 calendar | rejected | v52; MF3 closed too |
| MB5 retune per class | served | v76 |
| MB6 one round count per recipe | set aside | rounds from the MB5 study scaled by the row ratio |
| MB7 model diversity | served | CatBoost (v67), anchored members (v74, v75), per-airport members (v89), MLP (v94); XGBoost and constant leaves closed |
| MB8 12-month refit | served | every served component fits 12 months |
| MH1 rebuilt LIRF head | partial | out-of-fold calibration served (v64); MD2 part rejected |
| MH2 fallback heads outside LIRF | closed | +292 MSE |
| MH3 day-level artefact share (L7) | reopened 2026-10-03 at the airline level | 2025 check passes at LIRF only (0.43); the stand-and-day form lost live (v83, +0.14 s). Per airline the 2025 correlation is 0.77 across months and 0.58 within an airline (F16.6); third-edition lever G1 |
| MH4 smoothed tail table | set aside | the expected gain rests on one 2026 row (`sd` 111,654 s); a file that differs from the best on one row learns from the ranking |
| MH5 normal term reads `R_norm` | served | v63 |
| MH6 keep the supported rules | served | ITY340 hedge; MB3 |
| MH7 hold rule | closed | Discord tail rules +1,639 MSE; the anchored members and the A5 stand bound carry the holds |
| MF1 weather at pushback | rejected | v66 |
| MF2 runway queue | inside noise | v71 |
| MF3 local hour | closed | residual stack test |
| MF4 drift feature | rejected | v59 |
| MF5 planned taxi | served | v65 |
| MF6 backward mirrors | replaced | forward windows served by decision (L8, v85) |
| MS1 to MS4 | served | v48 onward; the A5 bound overrides MS1 on supported rows (v92) |
| MC1 one package | set aside | versioned scripts stay for the paper trail |
| MC2 constant sweep | set aside | |
| MC3 artefacts with hashes | partial | `models/v9x.manifest.json` (SHA-256); pickles remain |
| MC4 pinned versions | served | `requirements.txt` |
| MC5 one-to-one merges | partial | `validate="1:1"` in every new merge |
| MC6 reproduction path | served | `test_parity.py` (v67, v92, v94, v95), `fetch_metar.py`, `REPRODUCE.md` |
| MC7 reconcile README and RECAP | served | |
| MX1 organiser ruling | resolved | 2026-09-18 |

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

Track B: L1 (v73), CatBoost clocks and L5 (v74) are served. The anchored
CatBoost member is served (v75). L2, L6, the gate clocks, the clock bias
and `ARVT_3` are closed on the hold-out; L15 needs no refit. MB5 is
served (v76). A second study, an XGBoost class, CatBoost with more L2,
fitted blend weights, the stand sequence and the LIRF arrival link are
closed (v77 to v84). No measured signal is left above the 150 MSE bar. On
the 2025 hold-out, 45 % of the remaining error outside LIRF sits in 36
tail and 24-h rows that no column in the files predicts.
The stop rule of WINNING_PLAN 11.4 replaces the stop rule of section 6.
The final status table above gives the state of every measure.

## Twenty-third pass, 2026-10-13: budget of v135 and the last probes

### 23.1 How this pass was made

The pass uses the hold-out path of v135: the tree blend `blend_v133_holdout.parquet`, the K9 neural term at 0.72, the
served stack (15 leaves, 400 rounds, coverage weights) cross-fitted by month on the v133 tables, then the stand
bound. The functions are `stack_pushback_v90.frame`, `coverage_weights`, `crossfit` and `stand_bound`. The path
gives 62,347 MSE outside LIRF on the hold-out scale (344,336 rows). The logs are `logs/budget_v135.log`,
`logs/price_tw.log`, `logs/price_probe2.log` and `logs/price_probe3.log`.

### 23.2 Budget

Hold-out rows outside LIRF with a label up to 7,200 s, and the same error per row on the 2026 coverage mix
(ranking scale, 344,841 rows; grade C, the method of 16.2):

| coverage class | hold-out rows | hold-out RMSE, s | 2026 rows | projected live MSE |
|---|---|---|---|---|
| clock at the assigned stand | 64,698 | 93.4 | 96,221 | 2,854 |
| ADS-B segment, last point within 300 s of `MVT` | 85,565 | 164.9 | 104,808 | 8,699 |
| ADS-B segment that ends earlier (first gap) | 18,637 | 169.8 | 13,948 | 1,102 |
| OPDI only | 988 | 176.8 | 1,815 | 75 |
| no clock | 147,865 | 217.0 | 101,150 | 15,251 |
| sum | | | 317,942 | 27,981 |

The live MSE of v135 is 54,034, so LIRF and the labels above 7,200 s hold about 26,050 (48 %). The rows without a
clock hold 7,406 at LTFM, 5,149 at LFPG, 1,080 at EDDF, 527 at LEMD and 504 at EGLL. On the hold-out, 36 rows
above 7,200 s hold 32,073 of 62,347 MSE.

### 23.3 Findings

**F23.1. OPDI taxiway events add no clock.** V, medium.
- Evidence: 2,666 EDDF ranking departures without a clock have an `entry-taxiway` event of their OPDI flight
  (median 249 s before `MVT`). As stack columns on the rows without a clock the events price +5 MSE; on every row
  -33. The 986 EDDM hold-out rows move from 163.3 to 160.2 s RMSE. Closed.

**F23.2. The blend weights and the stack size sit at their optimum.** V, medium.
- Evidence: neural weight 0.65 and 0.80: -20 and +55 MSE. TabM share 0.8 and 1.0: +42 and +118. Stack with 31
  leaves and 600 rounds: +46; 800 rounds: +51; mean of 5 bagged stacks: -8. A second stack on the rows without a
  clock: +187 with member columns, +231 without. Closed.

**F23.3. Two member columns in the stack price -76 MSE.** V, low.
- Evidence: `d_nn_tree` alone -49 (January -27, July -22); with `d_tabm_mlp` -76 (January -35, July -41). The
  spread of the five families -68; each family as a column -15. Without coverage weights the first column prices
  -73 against its own control. Candidate v137.

**F23.4. LEBL labels in July 2025 run long on a few days.** V, medium; no lever.
- Evidence: 450 LEBL clock rows with a prediction more than 600 s below the label hold about 1,420 hold-out MSE.
  435 of them lie in July; 2025-07-09 and 2025-07-18 hold 83. The label is 1,075 s (median) above `MVT - AOBT_3`.
  The medians of each clock column equal those of the other LEBL rows, so no column separates the rows.

**F23.5. EDDF has no ground coverage from 2026-01-29 to 2026-02-28.** V, high; risk of the final file, no fix.
- Evidence: the share of EDDF departures with an ADS-B segment is 0.29 in February against 0.74 to 0.79 in
  January, June and July. The OPDI stand events stop on 2026-02-05 (1 to 33 per day against 190 to 900), so
  10,367 February rows have no clock. The v89 members, which read no ADS-B column, serve the EDDF clock rows of
  the hold-out at 176 s against 126 s for the v135 path; that gap puts the cost near 230 MSE on the final scale
  (670,790 rows). The mean February prediction lies 1 s above the median of `MVT - AOBT_3`; June and July lie 31
  and 42 s below it, so the rows show no bias.

### 23.4 Result

No measured lever beyond v137 remains for the final file. Earlier closed: a lookback of 7,200 s (-1,053 against
-1,081 MSE at 3,600 s), each change of the LIRF head (v100, v109, v113, v125) and the tree refit alone (v117).

## Twenty-second pass, 2026-10-09: coverage audit

### 22.1 Coverage per airport, 2026 ranking departures

| airport | rows | stand has a position | stand clock | segment only | no ADS-B |
|---|---|---|---|---|---|
| EDDF | 36,317 | 0.961 | 0.058 | 0.622 | 0.320 |
| EDDM | 25,966 | 0.999 | 0.719 | 0.243 | 0.038 |
| EGLL | 39,840 | 0.982 | 0.460 | 0.396 | 0.144 |
| EHAM | 38,182 | 0.994 | 0.755 | 0.196 | 0.049 |
| LEBL | 30,080 | 0.984 | 0.364 | 0.550 | 0.086 |
| LEMD | 36,954 | 0.983 | 0.179 | 0.527 | 0.295 |
| LFPG | 39,872 | 1.000 | 0.001 | 0.121 | 0.878 |
| LIRF | 26,899 | 0.985 | 0.020 | 0.274 | 0.706 |
| LSZH | 23,150 | 0.965 | 0.361 | 0.524 | 0.114 |
| LTFM | 47,581 | 0.989 | 0.000 | 0.000 | 1.000 |

Every 2026 ranking day is on disk (`adsb_day_loaded` 1.0). Stand positions
are not the gap: segment rows without a stand position are 0 to 3 % per
airport. EDDF and LEMD stay at segment level because the receivers miss the
aircraft at the stand.

### 22.2 Findings

**F22.1. Departures without an NM record never linked to their own tracks.** V, high.
- Evidence: the ADS-B and OPDI builders link on `CALLSIGN_flt`, an NM
  field. It is empty on all 5,283 ranking departures without a record, so
  none had an ADS-B or OPDI clock. On the 2025 hold-out, these rows outside
  LIRF hold 34,350 of 64,512 MSE; their clean rows serve at 441 s RMSE
  against 182 s for record rows.
- `FLIGHT_mvt` is the fallback key. EDDF, EHAM and LSZH report IATA flight
  numbers (`FLIGHT_mvt` equals the callsign on 13, 12 and 34 % of record
  rows), so an IATA airline code takes the ICAO code that the record rows
  pair with the same number: 98 codes with at least 20 pairs and a purity of
  0.9 (`LY358` is `ELY358`).
- Result on the 2026 rows without a record: 737 stand clocks and 975
  segments (1,712 of 5,290 rows); OPDI adds 157 clocks (LSZH 133). Rows with
  an NM callsign keep the served values (`--keep`). The 2025 tables gain
  6,082 ADS-B and 456 OPDI rows, so the stack learns them.

**F22.2. The stop order of the ADS-B builder is not stable.** V, medium; reproducibility.
- Evidence: a rebuild of the 2026 table with today's code swaps
  `adsb_stop1_*` and `adsb_stop2_*` on 1,855 to 4,072 rows against the v116
  table. `--keep` holds the served values, so v127 does not depend on it. Q2
  must pin the order before a clean-clone rebuild.

**F22.3. The stack was never tuned for the 2026 coverage mix.** V, medium.
- Evidence: the v93 search covered rounds and leaves on 21 days. The
  hold-out holds the 2025 mix: EGLL had a stand clock on 4 % of departures
  (46 % in 2026), LEMD on 0.3 % (18 %). v128 weights each hold-out clock row
  by the 2026-to-2025 share of its airport and coverage class (capped at 0.2
  and 5; no label) and tunes leaves, rows per leaf, L2 and rounds on the
  weighted cross-fitted objective (`tune_stack_v128.json`).

**F22.4. The flight-number link misses airlines whose callsign differs from the flight number.** V, high.
- Evidence: after v127, 3,578 of 5,290 ranking departures without a record still had no track; EHAM 909, LTFM 720, LFPG 645, LSZH 481. Many airlines broadcast alphanumeric callsigns (`EZY12AB`).
- The ADSB.lol points hold the climb-out up to 5,000 ft, so each aircraft has a lift-off: the first airborne point within 60 s of a ground point. On 2026 callsign links it follows `MVT` by a fixed offset per airport (EHAM -24 s, LFPG +30 s; std 9 to 22 s), and the aircraft type agrees on 99 %. Time and type alone pick the callsign aircraft on 99.4 % of record rows; with the lift-offs of callsign-linked aircraft skipped, on 100 % of a 1-in-6 day sample.
- Result: 1,016 more ranking rows with a segment (EHAM 697, LSZH 234, EDDF 85), 537 of them at the stand; 2,832 rows of 2025. On 2025 stand clocks the label minus `adsb_own_move` has a median of -5 s, p10 -46 s, p90 201 s (callsign links +25, -22, 217 s).

### 22.3 Uploads

| upload | change | live |
|---|---|---|
| v127 | F22.1: no-record rows linked to their ADS-B and OPDI tracks; stack and stand bound on the v127 tables | **240.70 s, -0.95 s (-461 MSE), accepted** |
| v128 | F22.3: stack with coverage weights; 15 leaves, 100 rows per leaf, L2 50, 400 rounds | 240.60 s, -0.11 s (-52 MSE) |
| v129 | the five TabM families refit on the K7 tables, so the networks also see the no-record rows with their clocks | **239.72 s, -0.87 s (-419 MSE), accepted** |
| v130 | F22.4: lift-off link; stack and stand bound on the v130 tables | **238.25 s, -1.47 s (-703 MSE), accepted** |
| v131 | the five TabM families refit on the K8 tables | **235.95 s, -2.30 s (-1,092 MSE), accepted** |
| v132 | plain LightGBM, plain CatBoost and per-airport LightGBM refit on the v130 tables; the anchored members keep the v97 fits and the v97 table; MS3 guard on the per-airport mix; stack on the hold-out blend of the same members | **235.54 s, -0.41 s (-194 MSE), accepted** |
| v135 | K7: ADS-B segments that end up to 3,600 s before the take-off, in the stack (v133 tables), in the five TabM families (K9) and in the plain tree members (tag `v133`); one upload for the three steps | **232.45 s, -3.09 s (-1,445 MSE), accepted** |

Reading of v135: coverage paid a fourth time. The first-gap rule gives 14,749 ranking departures (4.3 %) a track. The live gain is -1,445 MSE against hold-out prices of -1,081 (stack), -523 (networks) and -58 (trees): 0.87 of the sum landed. V133 and v134 have no live score, so the score does not split the stack, the networks and the trees. Reading of v132: the links also pay in the tree base, but less than in the stack and the networks (-194 MSE against -703 and -1,092). A departure without an NM record has no `AOBT_3` anchor, so only the plain members read its new track. On the hold-out the refit members gain 366 to 435 MSE each and the blend 415; after the stack 144 remain (stand-clock rows -28, segment rows -136, rows without a track +22), because the stack already read the tables. The v117 refit of every tree member (+0.09 s) predates the links. Reading of v130 and v131: the lift-off link paid twice again, and the network refit paid more than the stack (-1,092 against -703 MSE). Reading of v127 to v129: the coverage fix paid twice. The stack learned the newly linked rows in v127 (-461 MSE); the networks
learned them in v129 (-419 MSE). Together with v128 the fix moved the score from 241.66 to 239.72 s.

## Twenty-first pass, 2026-10-08 evening: audit with live judging only

### 21.1 Gap and rules

1. v122 scores 241.84 s = 58,489 MSE. 220 s = 48,400 MSE. The gap is
   **10,089 MSE** (arithmetic). Leaderboard API, 17:33 UTC: charming-eel
   213.25 s, jolly-lobster 213.29 s, strong-crane 213.52 s; rank 8 is
   220.45 s; kind-mango is rank 35.
2. User directive of 2026-10-08: no hold-out price before an upload; the
   live score judges each version. This overrides AGENTS.md rules 11.1
   and 11.2. Only the format checks (rows, ids, float64, no NaN) block an
   upload, because the scorer rejects a file that fails them. Rule 12 (no
   per-row probing, no file that differs from the best on a few rows)
   stays.

### 21.2 Findings

**F21.1. The stack is calibrated where the hold-out has few clocked rows.** V, high.
- Evidence: the 2025 clock bias `adsb_own_move - y` is stable per airport
  over the months (EGLL -89 to -150 s, LEMD -92 to -115 s, EHAM +11 to
  +14 s). On 2026 stand-clock rows with a good track, v122 minus
  (`adsb_own_move` minus that bias) has a median of -10 to +5 s at EGLL and
  LEMD, and -5 to +36 s elsewhere (LEBL July +36 s).
- Reading: the scarce EGLL and LEMD rows of January and July 2025 did not
  bias the stack. No lever.

**F21.2. The stack respects the first-moving-point floor.** V, high.
- Evidence: where the first ADS-B point moves at 3 m/s or more, the served
  value lies below `adsb_seg` minus 30 s on 0.3 % of hold-out rows (label
  0.7 %). A hard floor costs +955 MSE: towed aircraft fall far below it.

**F21.3. A second departure of one flight number within 1 h is long.** V, medium; rule 12.
- Evidence: 127 such rows in 2025; median label 1,791 s, 34 % above
  3,600 s. The ranking file has 58 second departures within 4 h. Too few
  rows for a file of their own; not pursued.

**F21.4. The LIRF gate and `R_norm` read no `AOBT_3` or `LOBT` term.** V, high.
- Both date from before track B. For the gate, `MVT - AOBT_3` (AUC 0.64)
  and `LOBT - SCHED` (0.66) separate copies at 1 to 2 h of delay; L15
  priced +108 hold-out MSE and was never uploaded. For `R_norm`, the clock
  is weak at LIRF: correlation with the genuine label 0.39 (0.53 to 0.72
  at most airports) against 0.815 for `R_norm`, so L6 stands.

**F21.5. The LIRF rows without a record and a long delay follow the 2025 shares.** V, high.
- Evidence: 2025 outcomes per delay band: 4 to 6 h 50 copies, 14 normal;
  6 to 12 h 33 copies, 1 normal; 12 to 18 h 16 copies, 5 day faults;
  18 to 24 h 1 copy, 5 day faults. v122 serves `D` on the 6 to 12 h band
  and hedges in the others. Between 50,000 and 61,000 s of delay, 2025
  has 5 day faults and 6 copies; three 2026 rows lie there.
- A public note of team resilient-kiwi reads its own leaderboard deltas as
  evidence that 2026 has fewer day faults. That evidence is a live score
  on 26 rows; rule 12 excludes it, so the hedge stays.

**F21.6. The gap to the top sits in lottery rows.** D.
- The same notes put 34 % of a validation error in five rows above
  20,000 s, and record a 15 s step from a change on 26 rows chosen with
  leaderboard deltas. With rule 12 in force, this part of the gap stays
  closed to kind-mango.

### 21.3 Strategy: a neural-forward blend

Neural members are the only model change that kept transferring live (2 to
5 times their hold-out price: v94, v105, v111, v112, v115, v122). Two
uploads, one change each:

| upload | change | live |
|---|---|---|
| v123 | TabM share 0.5 to 0.65 (neural weight 0.72) on the v122 families | 241.79 s, -0.06 s (-27 MSE) |
| v124 | fourth family `tabm_plr_j` (periodic embeddings on the J column set) | 241.67 s, -0.12 s (-57 MSE) |
| v125 | LIRF gate g5: the g3 columns plus the five NM clock columns (F21.4) | 242.35 s, +0.67 s (+326 MSE), rejected |
| v126 | fifth family `tabm_s`: the J column set plus the structure terms of the record (`build_struct_terms.py`) | 241.66 s, -0.01 s (-6 MSE) |

Reading of v123: the live gain per unit of share is small at 0.65; more
share is not the next step, more diversity is.

Reading of v124 to v126: the fourth family gained -57 MSE, the fifth -6. The
neural term saturates at four to five families. v125 confirms the LIRF-head
pattern of v109, v113 and K2: a gate change that moves every LIRF row lost
live (+326 MSE). The hold-out priced it +108, so the hold-out sign was right;
the hold-out underrates neural members, not every change.

### 21.4 Reachable score

220 s needs 10,089 MSE. No lever of this pass or of 20.7 reaches it within
the open data and rule 12.

## Twentieth pass, 2026-10-08: timestamp precision, and new strategies toward 220 s

### 20.1 Gap

1. v119 scores 242.51 s = 58,812 MSE. 220 s = 48,400 MSE. The gap is
   **10,412 MSE**, 18 % of the current error (arithmetic).
2. Leaderboard API, 2026-10-08 04:30 UTC: charming-eel 213.25 s,
   jolly-lobster 213.29 s, strong-crane 213.52 s, vigorous-whistle
   214.93 s, versatile-watermelon 218.77 s; rank 6 is 220.58 s. kind-mango
   is rank 34 at 242.51 s (v119).
3. Uploads left: 4 on 8 October (v120 used one), 5 on each of 9 and
   10 October, 5 on 11 October before 21:00 UTC.

### 20.2 How this pass was made

1. A scratch script saved the nineteenth-pass hold-out serve path
   (`measure_nm_join.outside_lirf` and `lirf_head`, with J1 and J2) per
   row. Every probe below reads the residual of that path. Hold-out scale:
   344,336 rows. Outside LIRF 64,512 MSE, LIRF 21,465 MSE.
2. New probes: the second of each time column, the precision of the NM
   times, the order of the row ids, the missing fields, and the NM default
   taxi times. Public sources: the GitHub search for 2026 challenge repos;
   one new README (`joris-decombe/taxiout`, team gentle-octopus, 274.1 s).
   Only ideas were read; no code was copied.
3. `python -m src_v3.mvt_precision measure` repeats the lever price and
   writes `models/v3/mvt_precision.holdout.json` (about 1 min, no fit).

### 20.3 Findings

**F20.1. The organiser times sit on a minute grid with noise at seven airports.** V, high.
- Evidence: at EDDM, EGLL, LEBL, LEMD, LFPG, LIRF and LTFM, the second of
  `BLOCK` takes the 13 values 54 to 59 and 0 to 6 only (each about 8.3 %;
  54 and 6 less often). `MVT` does the same at EDDM, EGLL, LFPG, LIRF and
  LTFM, in 2025 and in the 2026 ranking file. At EDDF, EHAM and LSZH both
  times have second resolution. NM writes `AOBT_3`, `LOBT`, `EOBT_1` and
  `IOBT` at minute resolution; `SCHED` sits on a 5 min grid.
- Reading: the source times are whole minutes, and the organiser added
  noise of at most 6 s. The noise limits `y mod 60`. That helps only on
  rows already served within 30 s, so it is not a lever.

**F20.2. At LFPG, a take-off time off the grid marks a long label.** V, high.
- Evidence: 97 LFPG departures of 2025 have an `MVT` second from 7 to 53.
  Every one of them has an NM record and a `BLOCK` on the grid. Their
  label has the quantiles 1,396 s (5 %), 4,453 s (50 %) and 10,069 s
  (95 %). In the cell `MVT - LOBT` > 2,500 s and `MVT - AOBT_3` < 1,800 s,
  the 22,275 on-grid rows have a median label of 1,191 s; the 73 off-grid
  rows have 6,100 s.
- Mechanism: the take-off time of these rows comes from a second source.
  The airport off-block stayed near `LOBT` (median `BLOCK - LOBT` 242 s),
  so the label is `MVT - LOBT` minus a small offset. NM's `AOBT_3` follows
  the take-off (median `MVT - AOBT_3` 916 s), so the trees read a normal
  taxi there.
- Cost today: on the 17 hold-out rows the served RMSE is 2,602 s; 334
  hold-out MSE.
- Rule P1 (`src_v3/mvt_precision.py`): raise the value of a flagged row to
  `MVT - LOBT` minus the median offset of the 2025 flagged rows; never
  lower it. Offset fit on the ten months outside the hold-out: 152.5 s.
  Hold-out **-250.1 MSE** (17 rows; body -26.4); leave-one-month-out RMSE
  over all 97 rows 1,293 s, against 3,277 s for a constant. The ranking
  file has **19 flagged rows**, all at LFPG; v119 serves 1,000 to 6,800 s
  on them, and the rule raises 18 of them.
- Other airports: EGLL has 7 off-grid rows of 2025 with labels under 31 s
  (data errors) and none in 2026; EDDM, LIRF and LTFM have none. The rule
  stays at LFPG.

**F20.3. J3 failed because it anchored on-grid rows.** V, medium.
- Evidence: J3 (v120) moved 137 ranking rows by more than 1 s; 5 of them
  are P1 rows. It raised the other 132 rows from a mean of 4,438 s to
  5,447 s. F20.2 shows that on-grid rows with a large `MVT - LOBT` taxi
  normally (median 1,191 s).
- Reading, grade D: this explains the +0.40 s of v120 without a live
  score per row. The precision flag separates the two regimes that J3
  mixed.

**F20.4. `AOBT_3` is often an NM default taxi.** V, high; no lever.
- Evidence: on clean hold-out rows, `MVT - AOBT_3` equals 1,200 s
  (within 6 s) on 37 % of the LTFM rows, and 720 s or 660 s on 40 % of the
  EDDM rows. On these rows `AOBT_3` is the take-off minus a fixed time and
  carries no off-block information.
- Test: the networks smooth numeric inputs, so they could miss the exact
  values. They do not: on the default rows the TabM RMSE is equal to or
  lower than the LightGBM RMSE at every airport (LTFM 221 against 226 s).

**F20.5. Probes closed in this pass.** V, high.
- `AOBT_3` with seconds (EHAM 16 %, others under 1.1 %): no closer to
  `BLOCK` than whole-minute values; served residual mean within 30 s of 0.
- Row ids: `MVT_ID_mvt` follows the day, not the order inside a day
  (rank correlation with `MVT` and with `BLOCK` equal to 5 decimals). No
  leak.
- Missing fields per airport, cross-fitted January against July: best
  gain -4.9 MSE.
- On-grid `MVT` at EHAM and LSZH: rows with `y > 3,600` s are 7 times more
  frequent among LSZH grid rows, but the rate stays at 0.16 %.
- The new public README (gentle-octopus) repeats the LOBT window, the
  `AOBT_3` anchor and the LIRF schedule mixture.

### 20.4 Ranked levers

| rank | id | lever | evidence | hold-out | live (low, high) | grade | uploads |
|---|---|---|---|---|---|---|---|
| 1 | P1 | raise flagged LFPG rows to `MVT - LOBT` minus 242 s | F20.2 | -250 | -100, -400 | B | 1 |
| 2 | N10 | six TabM seeds per family | v111 to v115: neural changes transferred at 2 to 5 times their price | seed price in 20.6 | 0, -100 | C | with P1 |

P1 moves 19 rows, so rule 12 bundles it with N10 (as J2 rode with
`tabm_plr`). Arithmetic for P1 live: 19 rows at the hold-out RMSE drop
(2,602 to 1,305 s) give about -280 MSE.

### 20.5 Reachable score and the 220 s target

1. P1 and N10 together: about -100 to -500 live MSE, 242.3 to 241.5 s.
2. **220 s needs 10,412 MSE. No measured lever reaches it.** The blocks of
   19.6 stand: the LIRF rows without a record and a delay above 1 h, LTFM
   and LFPG without a clock, and the tail without an observable.
3. The format of a field is an observable the earlier passes did not
   test. It found one regime at one airport. The top teams moved in steps
   of 10 to 100 s (19.1); the cause of those steps stays unmeasured.

### 20.6 Execution, 2026-10-08

Prices on the hold-out scale, bar -150 MSE for the bundle, set before the run:

| part | rows moved | hold-out, MSE | body | source |
|---|---|---|---|---|
| P1 | 17 hold-out, 19 ranking | -250.1 | -26.4 | `mvt_precision.holdout.json` |
| N10 seeds: each family as the mean of seeds 42 and 45 against seed 42 | all outside LIRF | +34.2 | +28.3 | `tabm_seeds_v121.holdout.json` |
| bundle | | -215.9 | | arithmetic |

The seed price is a proxy: each hold-out member is one seed, and the served
file averages six seeds per family against three. Averaging seeds cannot
raise the error of the member mean in expectation. The +34 comes from the
stack refit, which is cross-fitted on the hold-out and moved earlier TabM
variants by 16 to 30 MSE per month (`tabm_variants_v120.holdout.json`).
Grade D. P1 carries the expected gain of v121.

Live: v121 scored **242.03 s**, -0.49 s against v119 (-235 MSE against -216 priced, 1.1 times). The bar of AGENTS rule
11.5 holds (best minus 0.3 s is 242.21 s), so v121 is accepted. Every later upload builds on it. The split between P1 and
the seeds is not measured; a split upload would probe the 19 rows.

### 20.7 Second sweep after v121, 2026-10-08 afternoon

Goal: a lever toward 220 s. Each probe reads the 2025 hold-out serve path of
20.2 or the 2026 ranking file. None reaches the -150 MSE bar; no upload
follows.

| probe | idea | result | state |
|---|---|---|---|
| year drift | predict 2026 clocked rows with the own-flight ADS-B and OPDI columns masked (v97 LightGBM, three seeds); compare with the stand clock minus its 2025 airport bias; 2026 against 2025 | median drift per airport-month -56 to +37 s (one cell with a single 2025 row aside) | closed: no level shift to correct |
| time-local correction | median gap of other clocked rows within 30 or 60 min, cross-fitted on random halves | correlation 0.12 (2026) and 0.08 (2025); 1.4 % of the variance; 2025 price -46 MSE | closed |
| de-icing days | 2026 airport-days with the largest gap | worst day medians -31 s; day spread 24 to 37 s | closed: the served path follows the day level |
| `AOBT_3` copy rows | rows with `BLOCK = AOBT_3` (12 to 13 % at LEBL, LEMD, EDDM) | perfect detector at most about 960 hold-out MSE; served RMSE there 67 to 137 s at most airports | closed |
| rows without a record outside LIRF | cap of the served value | cap at 3,000 s: -22 MSE; errors run both ways | closed |
| LIRF copies in time | arrival artefact share within 15 to 120 min | AUC 0.51 to 0.52; day share 0.53 | closed: copies are not outages |
| LIRF copies in the track | ADS-B columns against the fallback label (153,576 rows) | AUC 0.45 to 0.50; fallback rate 0.19 to 0.22 with or without a track | closed: a copy is a reporting artefact |
| `SCHED` precision | schedule off the 5 min grid | 0 to 0.3 % of rows | closed |
| row ids inside a day | id order against `MVT`, `BLOCK`, `SCHED` | equal correlation to 0.01 | closed: no leak |
| format and mismatch flags | time precision classes, missing NM fields, `ADES`, `ADEP`, type, rule and callsign mismatches, string shapes, order of times | only P1 and the rows without a record stand out; every other flag below 8 MSE of bias | closed |
| LIRF rows without a record | served bias +503 s mean | gate over-predicts in July (0.31 against 0.19), not in January | closed: not stable across months |

Reading: the served path has no static, monthly, daily or format bias left
that a 2025 or 2026 observable can price. The largest block, LIRF schedule
copies, is invisible in every observable tested (time, track, arrivals of
the hour, schedule precision). 220 s stays out of reach with the open data
and rules of this team (20.5).

### 20.8 v122: `tabm_j` without J3

F20.3 puts the v120 loss on J3, so the stop rule of WINNING_PLAN 6 leaves
`tabm_j` open: a written defect explanation exists for its bundle. v122
serves the v121 recipe with `tabm_j` as a third TabM family, one third per
family, at the TabM share 0.5 of v120. Hold-out full path against the v121
neural term: -41.9 MSE, body -40.6, both months negative. The user rule of
2026-10-03 applies: the bar does not stop the upload.

Live: **241.84 s**, -0.18 s against v121 (-88 MSE, 2.1 times the price).
This is under the 0.3 s bar of AGENTS rule 11.5. The accepted recipe stays
v121; v122 is the leaderboard best. The live gain confirms F20.3: the
member gains alone, so J3 caused the v120 loss.

## Nineteenth pass, 2026-10-07: audit of v115 and v116, and new strategies toward 220 s

### 19.1 Gap

1. v116 scores 243.65 s = 59,365 MSE. 220 s = 48,400 MSE. The gap is
   **10,965 MSE**, 18 % of the current error (arithmetic).
2. Leaderboard API, 2026-10-07 14:16 UTC: strong-crane 213.52 s,
   jolly-lobster 213.71 s, vigorous-whistle 215.81 s, charming-eel 216.53 s,
   versatile-watermelon 218.77 s; rank 6 is 220.58 s. Twelve teams are
   below 225 s. kind-mango is rank 36 of 235 teams.
3. The top teams moved in large steps: bold-ladder 329.8 to 230.4 s in one
   upload (2026-10-01), strong-crane 253.2 to 235.9 s in one day
   (2026-09-26), adventurous-radio 263.2 to 228.2 s within 14 h
   (2026-09-30). Steps of this size come from a few rows with large
   errors, not from tuning (grade D).
4. Uploads left: 4 on 2026-10-07 (v117 used one), 5 on each of 8, 9 and
   10 October, 5 on 11 October before 21:00 UTC: 24 in total.

### 19.2 How this pass was made

1. Scripts in the session scratchpad rebuilt the v115 serve path on the
   January and July 2025 hold-out: outside LIRF the v97 member blend, the
   MLP and the v115 TabM member at 0.48, 0.12 and 0.40, the stack
   cross-fitted January against July on the v115 clock tables, and the
   A5 bound. At LIRF: the out-of-fold g3 gate with the nested isotonic
   map, the out-of-sample v41 `R_norm`, Step A with the date fix, and
   ITY340. One module in the repo repeats the part that the levers need:
   `python -m src_v3.measure_nm_join` writes
   `models/v3/nm_join.holdout.json` (about 3 min, no fit).
2. Public sources, read on 2026-10-07: the leaderboard API; the READMEs and
   research notes of two public GPLv3 repos, `dhsdb010/OPDI` (team
   quirky-honey, 302.01 s) and `EnioAguiar/prc-taxiout-2026` (team
   outgoing-boat, 243.95 s). Only ideas were taken; no code was copied.
3. Grades as in 0.2. A number names a JSON key, a scratch measurement of
   this pass, or arithmetic.

### 19.3 Budget of the served path (task A)

Hold-out outside LIRF, v115 path, hold-out scale (317,789 rows):

| class | rows | MSE |
|---|---|---|
| clean | 292,319 | 29,951 |
| fallback | 25,339 | 2,450 |
| tail | 34 | 12,086 |
| 24-h | 1 | 20,251 |
| low | 96 | 88 |
| total | 317,789 | 64,826 |

| coverage class | rows | MSE | RMSE, s |
|---|---|---|---|
| clock at the assigned stand | 63,636 | 1,618 | 93.6 |
| ADS-B segment only | 85,347 | 8,403 | 184.1 |
| OPDI only | 2,460 | 173 | 155.5 |
| no clock | 166,346 | 54,632 | 336.3 |

The v115 clock tables add X-Plane stands, so 13,336 more hold-out rows
carry a stand clock than in the v97 table (16.2).

LIRF hold-out, current head (26,528 rows): **21,774 MSE**; with a perfect
gate 11,340.

| delay `sd`, record | rows | fallback rate | gate mean | MSE | MSE, perfect gate |
|---|---|---|---|---|---|
| up to 1,800 s, record | 14,897 | 0.275 | 0.279 | 1,681 | 1,258 |
| 1,800 to 3,600 s, record | 7,443 | 0.085 | 0.091 | 3,224 | 1,800 |
| 3,600 to 7,200 s, record | 3,047 | 0.065 | 0.065 | 4,569 | 1,762 |
| 7,200 to 14,400 s, record | 632 | 0.017 | 0.016 | 624 | 507 |
| 3,600 to 7,200 s, no record | 194 | 0.227 | 0.320 | 1,694 | 362 |
| 7,200 to 14,400 s, no record | 136 | 0.353 | 0.482 | 4,499 | 310 |
| 14,400 to 70,000 s, no record (Step A) | 42 | 0.857 | 0.662 | 4,842 | 4,842 |

The 375 LIRF rows without an NM record and with a delay above 1 h hold
11,036 MSE, **51 % of the LIRF error**. They are 1.4 % of the LIRF rows.

Live scale, projected onto the 2026 coverage mix (grade C, the method of
16.2):

| block | live MSE | share of 59,384 (v115) |
|---|---|---|
| outside LIRF, no clock | 18,032 | 30.4 % |
| outside LIRF, segment only | 9,715 | 16.4 % |
| outside LIRF, stand clock | 2,646 | 4.5 % |
| outside LIRF, OPDI only | 208 | 0.4 % |
| LIRF and the tail outside LIRF (remainder) | 28,784 | 48.5 % |

Normal classes outside LIRF by airport: LTFM 7,423, LFPG 6,472, EGLL 4,389,
LEBL 3,137, EDDF 3,115, LEMD 2,640, EHAM 1,620, LSZH 998, EDDM 806.

### 19.4 Findings

**F19.1. The NM join holds every reported off-block within 1 h of `LOBT`.** V, high.
- Evidence: all 2,062,577 departures of 2025 with an NM record have
  `|BLOCK - LOBT| <= 3,606` s (minimum -3,606, maximum +3,606, no
  exception). Arrivals carry the same window on landing against
  `ARVT_3` (-3,623 to +3,606 s). No other NM time has a hard window.
- Mechanism: the organiser joined a movement to its NM flight only inside
  that window. So the label of a record row lies in `MVT - LOBT` plus or
  minus 3,606 s.
- Cost today: the pipeline does not use the window. `mvt_lobt` is clipped
  at 7,200 s, so the trees cannot see the window on 94 ranking rows.
  Projection of the served value onto the window: hold-out **-256 MSE**
  outside LIRF (75 rows, of which -234 on rows with `y > 3,600` s; LFPG
  -83, EHAM -78, LSZH -50, EGLL -29) and -28 at LIRF (`nm_join.holdout.json`, keys
  `outside_lirf_window`, `lirf.window`). On v116, 54 ranking rows lie
  outside the window. Their projection guarantees a live gain of at least
  **80.5 MSE**: a projection onto an interval that holds the label cannot
  raise the error of a row (key `served`).
- Public confirmation: team outgoing-boat reports that the window held on
  the 2026 ranking file (README of `EnioAguiar/prc-taxiout-2026`, their
  diagnostic upload v10). Rule 12 forbids such an upload for us; the
  published result needs none.

**F19.2. At LIRF a schedule copy is impossible outside the window.** V, high.
- Evidence: a fallback row has `BLOCK = SCHED`, so on a record row it
  needs `|SCHED - LOBT| <= 3,606` s. 1,355 LIRF hold-out record rows have
  `|SCHED - LOBT| > 3,606` s; none of them is a fallback row. The served
  gate gives them a mean share above 0.
- Cost: gate set to 0 on those rows: hold-out **-71 MSE** (824 rows move;
  body -162, tail +91; key `lirf.copy_record`). 1,550 LIRF ranking rows
  are in this cell.

**F19.3. A missing NM record means the reported off-block missed the window.** V, high.
- Evidence: 929 departures of 2025 without a record link to the arrival
  row of the same flight number at another of the ten airports, landing
  10 min to 14 h after the take-off. That arrival row carries the NM
  record of the flight. 97.1 % of these departures have
  `|BLOCK - LOBT| > 3,606` s. Among their fallback rows 85 % have
  `|SCHED - LOBT| > 3,606` s; among their genuine rows 10 %.
- Mechanism: the departure join failed because the reported off-block was
  more than 1 h from `LOBT`. If `|SCHED - LOBT| <= 3,606` s, a schedule
  copy would have joined, so the row is not a copy.
- Cost: at LIRF, 21 linked hold-out rows meet that test and none is a
  fallback row. Gate set to 0 there: hold-out **-193 MSE** (20 rows move;
  body -187; key `lirf.copy_linked`). 20 LIRF ranking rows are in this
  cell; 36 LIRF ranking rows without a record link at all. Outside LIRF
  the same link gives no usable bound: a cap at `MVT - LOBT - 3,606` s
  fails on 3.2 % of the rows and costs +38 MSE.
- All three NM join rules at LIRF together: **-309 MSE** (key `lirf.all`),
  -352 of it on rows with `y <= 3,600` s.

**F19.4. The remaining error has a time-local part that no observable carries.** V, high.
- Evidence: per airport and hour, group means of the clean residual remove
  3,120 hold-out MSE in sample against 1,065 for permuted residuals. The
  mean residual of the other departures within 15 min, read from the
  labels, removes 2,092 (oracle). Read only from neighbours with a stand
  clock it removes 38; on rows without a clock, from neighbours with a
  clock or a segment, 137.
- Tested carriers, none passes: the taxi-in excess of arrivals within 15,
  30 or 60 min (correlation -0.019 to 0.044 per airport, 0 MSE); the
  arrival and departure runway configuration (cross-fitted group means
  +226 to +314 MSE).
- Reading: the local error sits on rows without a clock (LTFM, LFPG), and
  their neighbours have no clock either.

**F19.5. No static bias is left.** V, high.
- Evidence: residual group means fitted on January and applied to July,
  and back, raise the MSE for every key tested (airport, hour, weekday,
  runway, stand, operator, airline prefix, type, destination, flight
  number, coverage, record status, runway and hour): +136 to +573.

**F19.6. The LIRF gate is calibrated; the rows without a record are ambiguous, not mispriced.** V, high.
- Evidence: on rows without a record and a delay above 1 h, the fallback
  rate depends on the airline (ISR, LAV, ETH, TKJ, TWB, MSC 1.0; KLM, EWG,
  ELY 0.0; RYR 0.33, WMT 0.49). The served gate follows it: log loss 0.366
  against 0.476 for a leave-month-out airline rate. The 2025 corrupted
  flight numbers at LIRF (airline code plus a repeated tail of it, for
  example `ITYTY672`; 2,906 training and 466 ranking departures) sit in
  cells where the gate mean matches the fallback rate.
- A second LIRF artefact, the block equal to `EOBT_1`, is real (2,148 rows
  within 60 s against about 1,010 expected among late departures) but a
  perfect classifier gains only 403 MSE.

**F19.7. The label-derived daily delay series does not help the LIRF gate.** V, medium.
- Evidence: a day-level shift of the gate logit, solved so that the
  served head matches the daily mean of `max(0, BLOCK - SCHED)` (the
  quantity of EUROCONTROL's daily pre-departure delay, simulated from the
  2025 labels), raises the LIRF hold-out MSE by 523 over 61 days.
- Reading: the daily mean reflects the genuine taxi as much as the
  fallback share. With the airport-day ceiling of 572 MSE outside LIRF
  (RECAP, EUROCONTROL row), the label-derived series are worth at most
  about 1,000 live MSE even if the organiser allows them.

**F19.8. The OPDI fetch drops the de-icing events.** V, medium.
- Evidence: `src/fetch_opdi_events.py` keeps ten event types. OPDI
  `flight_events` v002 also holds `entry-deicing_pad` (research note of
  outgoing-boat, read from the Parquet footer of the 2026-07-09 file).
- Limit: OPDI comes from OSN receivers. The organiser's coverage table
  (`euctrl-pru.github.io/opensky-airport-coverage`, sample 5 to 7 June
  2026) gives a median received taxi-out share of 0.996 at LSZH, 0.797 at
  LEBL, 0.622 at EDDF, 0.286 at EHAM, 0.175 at LEMD and 0.000 at EDDM,
  LFPG, EGLL, LIRF and LTFM. The events add nothing at the airports
  without a clock, and ADSB.lol already shows the stops at the others.

**F19.9. Drift checks on 2026 find coverage changes, not defects.** V, medium.
- LSZH July: the served value sits a median 277 s above the last time at
  the stand, against 82 s for the 2025 label. The plain LightGBM member
  predicts 693 s on these rows against 577 s in July 2025 (label 575 s);
  `adsb_own_move` explains +159 s of it (input swap). In 2026 the tracks
  show the pushback start: the gap from first move to the last point at
  the stand grew from 44 to 189 s, as at EHAM. In 2025 the label followed
  `adsb_own_move` within a median 28 s in both months. The model follows
  the same clock, so no fix is set. The risk is about 450 MSE if the 2026
  label kept the 2025 relation to the last stand point instead.
- January 2026: predictions sit 134 to 201 s above the January 2025 labels
  at EDDF, EDDM, EHAM, LSZH and LTFM. The NM clocks rose by 31 to 129 s,
  and the EUROCONTROL check of 2026-10-06 put the January means near the
  implied levels.
- EDDF: segment share 0.873 in 2025 against 0.680 in 2026, stand clocks
  0.242 against 0.061. The stand names did not change; the receivers did.

**F19.10. Re-tested and closed.** V, high.
- The next arrival at the stand as a hard floor: +1,530 MSE (LTFM +1,340),
  as v81.
- Off-block copies of `LOBT`, `EOBT_1` or `IOBT` outside LIRF: no spike at
  0 s on late departures.
- A fallback gate outside LIRF: late fallback rows spread over many days
  (LEBL: 547 rows on 182 days, the top 10 days hold 111).

**F19.11. A better transfer predictor exists in public.** V, decision.
- Evidence: outgoing-boat prices a change by the gain on rows with
  `y <= 3,600` s only (the body share). Over their six uploads with a
  known live delta, the body share predicted the live delta with a ratio
  of 1.03 and a mean absolute error of 0.15 s, against 0.44 s for the
  full hold-out RMSE (their README, section 2.4).
- Reading for us: the NM join rules at LIRF gain mostly in the body
  (-352 of -309), so they are likely to transfer. The window outside LIRF
  gains in the tail; its floor is the guaranteed 80.5 MSE.

### 19.5 Ranked levers

Live MSE on the 344,841-row scale; negative is a gain. Bars are on the
hold-out scale and set before the run.

| rank | id | lever | evidence | hold-out | live (low, high) | grade | hours | uploads |
|---|---|---|---|---|---|---|---|---|
| 1 | J1 | project every served value onto the LOBT window; at LIRF set the gate to 0 where a schedule copy breaks the window | F19.1, F19.2 | -371 | -80, -500 | B | 3 | 1 |
| 2 | J2 | at LIRF, set the gate to 0 on rows without a record whose arrival row gives a `LOBT` within 1 h of `SCHED` | F19.3 | -193 | -50, -300 | B | 2 | 1 |
| 3 | N10 | neural members: more TabM seeds, a second network family, a higher neural share | v111 to v115 transferred at 2 to 5 times their price | not priced | -100, -400 | C | 6 GPU | 1 |
| 4 | K6 | OPDI `entry-deicing_pad` events as stack columns | F19.8 | not priced | 0, -200 | D | 6 | 0 to 1 |
| - | Q3 | price levers by their body share and a 2026 importance weight | F19.11 | process | 0 | C | 2 | 0 |
| - | E1 | EUROCONTROL label-derived series | F19.7, RECAP | outside LIRF only | 0, -1,000 | decision | 3 | 1 |

J1 needs no fit. The copy test changes the gate inside `predict_v30.main`
(a gate hook, as `r_norm_post` changes `R_norm`). The window is the last
step, after the stack, the A5 bound and `step_a_date`. Only rows outside
the window, and LIRF rows with a copy test, may move. The label-free gate:
no row may leave the window, and no row without `LOBT` may move.

### 19.6 Reachable score and the 220 s target

1. J1 and J2 together: -130 to -800 live MSE, 243.4 to 242.0 s
   (arithmetic). With N10 and K6 at their high end: about 240.8 s.
2. E1, if the organiser allows it: about 1,000 MSE more, near 238.7 s.
3. **220 s needs 10,965 MSE. No measured lever and no identified data
   source reaches it.** The error that blocks it sits in three places:
   - LIRF rows without an NM record and a delay above 1 h: 51 % of the
     LIRF error. F19.3 resolves them where the `LOBT` of the flight is
     known; only 35 of 348 such ranking rows link to an arrival row.
   - LTFM and LFPG: 13,895 live MSE in the normal classes, nearly all on
     rows without a clock. No open archive has ground coverage there
     (F19.8, 18.6).
   - Tail and 24-h rows outside LIRF without a record: no observable.
4. The top teams reached 213 to 216 s in steps of 10 to 100 s (19.1).
   With the information of this pass, the cause of those steps is
   unmeasured for us.

### 19.7 Upload plan to 11 October

| UTC day | work | uploads |
|---|---|---|
| Wed 7 Oct | J1 build, parity check of the unchanged rows, label-free gate, upload | 1 |
| Thu 8 Oct | J2 on top of the best file; N10 hold-out prices on the GPU overnight; K6 probe on one January day, close if no gain | 1 to 2 |
| Fri 9 Oct | N10 full-year members if the price passes; Q2 manifest and parity for the best file | 1 to 2 |
| Sat 10 Oct | Q2 reproduction on a clean clone; README and REPRODUCE for the final recipe | 0 to 1 |
| Sun 11 Oct | no new model; last upload before 21:00 UTC | 0 to 1 |

### 19.8 Corrections to earlier statements

| earlier statement | this pass |
|---|---|
| 16.4 and F16.10: the tail rows outside LIRF have no observable | rows with an NM record do: the LOBT window lifts 9 hold-out tail rows (`y > 7,200` s) and gains 214 MSE on them (F19.1) |
| WINNING_PLAN third edition 2.4: a perfect LIRF gate is the ceiling of gate work | for rows without a record, the arrival-side `LOBT` decides the gate exactly where it exists (F19.3) |
| 18.6: no source beats ADSB.lol at LFPG; OSN state vectors stay out by team rule | the organiser's OSN coverage table gives 0.000 at LFPG, LIRF, LTFM, EGLL and EDDM, so the team rule costs nothing there |

### 19.9 Execution, 2026-10-07

| upload | change | hold-out, MSE | live |
|---|---|---|---|
| v118 | J1: LIRF copy test, then the LOBT window as the last step | -371 | 242.81 s, -0.85 s (-411 MSE), accepted |
| v119 | J2 (20 rows) bundled with `tabm_plr` for rule 12 | -193 and -83 | 242.51 s, -0.295 s (-143 MSE), 0.005 s under the bar |
| v120 | J3 (93 rows) bundled with `tabm_j` at the TabM share 0.5 | -255 and -17 | 242.91 s, +0.40 s (+194 MSE), rejected |

Measured for v120, on the full serve path against the v119 recipe:

1. J3 is the second gain of the window family. Rows whose `LOBT` lies 2
   to 4 h before the take-off get no window signal from the trees,
   because `mvt_lobt` is clipped at 7,200 s. The 2025 label sits
   `MVT - LOBT` minus 634 to 1,040 s there. Served at that value: -255 MSE
   on 49 rows (`nm_join.holdout.json`, key `j3`).
2. `tabm_j`, TabM on the v116 columns plus the unclipped LOBT terms, is
   the best single member (hold-out RMSE 266.0 s; -526 MSE alone at 0.4).
   It correlates 0.9955 with `tabm_v116`, so as a third member it adds
   -1 MSE at the share 0.4 and -17 at 0.5; a swap for `tabm_v116` gives
   -8 (`tabm_variants_v120.holdout.json`). The neural term is saturated on
   the hold-out.
3. J4, the arrival-side `AOBT_3` as a clock for departures without a
   record, is worse than the served value (+38 to +139 MSE). Closed.

Reading of v120, grade D: J3 holds about 90 % of the squared change
against v119 (93 rows, RMS 1,837 s), and its whole hold-out price sits on
rows above 3,600 s. Outgoing-boat found that share of a price
anti-correlated with the live delta (F19.11). J1 and J2 transferred; their
gains sat on rows below 3,600 s (J1 at LIRF) or came from a hard bound (the
window). A rule fitted to the rare 2025 tail rows did not. Both parts close.

## Eighteenth pass, 2026-10-06: new observables tested, and the TabM member (v111)

### 18.1 Gap

1. v111 scores 245.23 s = 60,137 MSE. 220 s = 48,400 MSE. The gap is
   11,737 MSE, 20 % of the current error (arithmetic).
2. Leaderboard API, 2026-10-05 22:05 UTC: top 213.92 s (strong-crane),
   rank 5 220.08 s, rank 10 225.85 s. kind-mango was rank 35 of 132 teams
   at 246.48 s. 34 teams are ahead, so the missing error is not one rare
   trick.

### 18.2 Measurements without an upload

Data: the 2025 training files, the saved v97 hold-out members and the
ADSB.lol points on disk. Scripts ran from the session scratchpad; the
numbers below are their outputs.

| question | method | result | state |
|---|---|---|---|
| Does `AOBT_3_flt` copy the airport off-block on some rows? | block minus `AOBT_3` in whole minutes, 2025 departures | a smooth peak, no spike at 0 min (EDDF: 0.118 at 0 min, 0.139 at +1 min) | closed |
| Does the inbound arrival of the same airframe tell a delay at the stand from a delay on the taxiway? | link each departure and each arrival to its ADS-B `icao` by callsign near `MVT`; join the previous arrival of the same `icao` at the airport (79 to 96 % of departures at 9 airports; LTFM has no ADS-B point in 2025); LightGBM on `y - final`, January against July | control stack +814 MSE, with the link +831; residual mean by inbound slack within 10 s in every bin | closed: the stand link of the base already carries it |
| The same link as a LIRF gate column | logistic layer on the out-of-fold g3 score, fit outside January and July | on delayed rows the fallback rate is 24.8 % after an inbound artefact and 6.7 % otherwise, but the served mixture loses 215 MSE | closed: g3 already reads the airline and stand shares |
| Position of the first ADS-B point of a segment | east and north offset from the assigned stand as stack columns, v105 path | +366 MSE against +237 for the control stack | closed, as K3 |
| Stand positions learned from ADS-B (OSM has no position for the assigned stand on 42 % of EDDM and 32 % of EHAM 2026 departures) | hold-out RMSE of segment rows with an unknown stand and a stationary first point | EHAM 107 s and EDDM 118 s, against 92 s and 68 s on stand-clock rows: about -100 hold-out MSE | not built |
| Can the 24-h artefact occur without a local date crossing? | 2025 LIRF null-record rows above the Step A gate | no crossing: 50 fallback, 14 genuine, 0 24-h; crossing: 51 fallback, 11 24-h, 1 genuine; above 25,000 s no genuine row | `src_v3/step_a_date.py` drops the 24-h term on such rows; it moves 1 row of 2026 (64,814 to 53,520 s), so rule 12 bars an upload of it alone |

### 18.3 TabM member (v111)

`src_v3/tabm_member.py`: TabM (`tabm` 0.0.3) with k = 16 submodels, 2
blocks of 512, piecewise-linear embeddings (`rtdl_num_embeddings` 0.0.12,
version B, 32 quantile bins, size 16) of the z-scored signed-log numeric
columns, a missing flag per column with gaps, and embeddings of the 10
categorical columns. AdamW, learning rate 1e-3, weight decay 3e-4, one
cycle over 16 epochs, batch 1,024, mixed precision on an RTX 2070. The
target is the label in thousands of seconds, clipped at 80,000 s.

| quantity | TabM | MLP (v105) | v97 blend |
|---|---|---|---|
| best stop-month RMSE (months 11, 12) | 163.5 s | 171.0 s | |
| hold-out RMSE (January, July) | 266.8 s | 274.6 s | 265.6 s |
| residual correlation with the blend | 0.980 | 0.950 | |

| TabM share on top of the v105 base | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| base only, MSE | -130 | -214 | -252 | -244 | -191 |
| full serve path (stack, A5 bound), MSE | | -227 | -231 | -191 | |

The share 0.3 was set before the full-year fit. Live: 245.23 s, -1.25 s
(about -614 MSE), 2.7 times the price.

### 18.4 Variants, the LIRF gate member and v112

| member | stop-month RMSE | on the v105 base at 0.3 / 0.4, MSE | state |
|---|---|---|---|
| `tabm`, 16 epochs (v111) | 163.54 s | -252 / -244 | served in v111 |
| `tabm_e24`, 24 epochs | 162.68 s | -343 / -360 | served in v112 at 0.35; full path -320 |
| mean of the two | | -307 / -319 | residual correlation 0.997: no gain |
| `tabm_k32`, k = 32, width 768 | 162.88 s | -203 / -188 | closed |

The TabM gate member (`src_v3/tabm_gate.py`, g3t) prices -319 MSE at the pre-registered share 0.5 (January -16, July -303), against -484 for the CatBoost member that lost 0.89 s live (January +71). It has no target statistics and reads the 2026 share columns directly. Uploaded alone on top of v112
as v113: 247.40 s, +3.12 s. Its 2026 fallback probabilities sit above those of g3 (LIRF rows mean
+37.1 s). A new model class in the LIRF head has now failed live five times (v49, v68, v100, v109,
v113): only new 2026 information has moved the gate the right way (v101, v102, v104).

| member | stop-month RMSE | on the v105 base at 0.35 / 0.4, MSE | state |
|---|---|---|---|
| `tabm_anchor`, target `y - clip(MVT - AOBT_3, 0)` | 163.99 s | -105 at 0.2 | closed; with `tabm_e24` +112 |
| `tabm_e32`, 32 epochs, best epoch 27 | 162.19 s | -382 / -386 | 24 MSE better than `tabm_e24` |

v112 = `tabm_e24` at 0.35 plus `src_v3/step_a_date.py` (one row). Live 244.29 s, -0.94 s. The two parts cannot be split: a file that differs from the best on one row is not allowed (rule 12).

### 18.6 More ADS-B ground coverage at LIRF, LFPG and LTFM

Search of 2026-10-06 for open trajectory data beyond ADSB.lol and OPDI:

| source | data for 2025 and January/July 2026 | licence | state |
|---|---|---|---|
| ADSB.lol `globe_history` | `prod-0` (used), `staging-0` and `mlatonly-0` (K1: +0.6 points at most); no `prod-1` dump in 2025 or 2026 | ODbL 1.0 | exhausted; it already holds FlyItalyADSB and TheAirTraffic |
| ADS-B Exchange samples | the 1st of each month only (2 of 62 ranking days) | non-commercial research and evaluation, no derivative works without written permission | not open data |
| adsb.fi | live API only; history replay answers 403 | personal non-commercial use, citation | no archive |
| Airplanes.Live | live API only; history replay answers 403 | educational, non-commercial | no archive |
| Wingbits | one rotating sample day | evaluation only | no |

Evaluation probe on 2026-07-01 (ADS-B Exchange sample, aggregate numbers only): share of departures with an ADS-B
ground segment and with a clock at the assigned stand.

| airport | ADSB.lol segment | ADSB.lol stand clock | ADS-B Exchange segment | ADS-B Exchange stand clock |
|---|---|---|---|---|
| LIRF | 0.0 % | 0.0 % | 83.8 % | 8.8 % |
| LTFM | 0.0 % | 0.0 % | 28.5 % | 3.5 % |
| LFPG | 23.7 % | 0.1 % | 0.7 % | 0.0 % |
| EGLL | 98.5 % | 72.3 % | 98.3 % | 85.2 % |

Reading:

1. Receivers near LIRF and LTFM exist, but their data is not open. No open archive covers them.
2. At LFPG no source beats ADSB.lol.
3. An ADS-B clock does not show how LIRF reported the off-block. It can sharpen the genuine term only;
   the ADS-B columns in `R_norm` already lost live (K2, v100, +0.12 s).
4. Arithmetic, if 28.5 % of LTFM departures carried a segment on all 62 days: about 0.285 x 47,581 x
   (240^2 - 175^2) / 344,841 = 1,060 live MSE, about 2 s. The two free sample days give 2/62 of that.

### 18.7 X-Plane stands and ATFM slots (v115)

OSM has no position for the assigned stand on 42 % of EDDM and 32 % of EHAM departures. The X-Plane Scenery
Gateway packs (GPL v2+) name every stand; two naming rules match them (sub-positions `215A`/`215B` for `215`;
leading zeros, `C06` for `C6`). On stands both sources have, the positions agree to a median of 28 m (EDDM)
and 41 m (EHAM), inside the 100 m radius of a stand clock.

| airport | stand clock 2026, OSM | with X-Plane | stand clock 2025, OSM | with X-Plane |
|---|---|---|---|---|
| EDDM | 0.396 | 0.726 | 0.315 | 0.557 |
| EHAM | 0.519 | 0.763 | 0.454 | 0.639 |

The tree members keep the tables they trained on. The stack, the A5 bound and TabM read the new clocks; TabM
also reads the ATFM slot adherence per airport and UTC day (take-off times only). Hold-out against the v114
path: -33 MSE (stack -27, TabM -12). Live: 243.69 s, -0.33 s, about 5 times the price.

### 18.5 Reading

1. Neural members transferred at 2 to 4 times their hold-out price three
   times (v94, v105, v111). Feature, parameter and weight changes of the
   tree members did not (v96, v98, v108, v110).
2. The new row-level observables of this pass add nothing on top of the
   base: the base already reads the stand link, the airline shares and
   the ADS-B clocks.
3. The next measures are more TabM capacity and seeds, and a second
   neural target (the `AOBT_3` anchor). 220 s needs about 11,700 MSE more;
   no measured lever comes near that.

## Seventeenth pass, 2026-10-05: strong and weak points of v105, and the CatBoost gate (N7)

### 17.1 Gap

1. v105 scores 246.48 s = 60,752 MSE. 220 s = 48,400 MSE. The gap is
   12,352 MSE, 20 % of the current error (arithmetic).
2. Leaderboard API, 2026-10-05 16:50 UTC: top 213.92 s (strong-crane),
   then 216.88, 217.28 and 217.47 s; rank 5 is 220.08 s. kind-mango is
   rank 34 of 107 teams.
3. v108 (N2 LightGBM members on the v105 recipe) scored 246.49 s, +0.02 s.
   It is rejected. v105 stays the best file.

### 17.2 Strong points

| point | evidence |
|---|---|
| pushback clocks of the scored flight | rows with a stand clock serve 93.8 s against a 92.4 s floor (F16.2) |
| six base members of three classes | tree members correlate 0.975 to 0.991 with the v97 blend; the MLP 0.950 (hold-out residuals, this pass) |
| arrival information in the LIRF gate | v101, v102 and v104: -0.94 s live in three uploads |
| exact rebuild | v105 and the v104 base rebuild at max difference 1e-13 s and 0.0 s from the recorded recipe (this pass) |

### 17.3 Weak points

Live projection of the v97 budget (section 2.3 of the third edition),
grade C:

| block | live MSE | share | lever class that can act |
|---|---|---|---|
| LIRF, all classes | about 20,460 | 33 % | gate (fallback probability), `R_norm` |
| outside LIRF, no clock | 18,332 | 30 % | base members only; LTFM and LFPG have no clock |
| outside LIRF, segment only | 10,419 | 17 % | base members and stack |
| tail and 24-h outside LIRF | about 9,410 | 15 % | none: no observable (F16.4) |
| outside LIRF, stand clock | 2,604 | 4 % | none: at the floor (F16.2) |

Second reading of the weak points:

1. The LIRF gate is the largest block that a model change can reach. A
   perfect gate cuts the LIRF hold-out MSE from 21,714 to 11,126
   (`lirf_gap.holdout.json`). `R_norm` changes did not transfer (v49,
   v68, v100). Gate changes did (v101, v102, v104).
2. The served gate is one LightGBM classifier. Its main signal is the
   fallback rate of the airline and the operator: a rate per category.
   CatBoost computes ordered target statistics per category and per
   combination of categories. No CatBoost gate was tried before this pass.
3. Base-member work outside LIRF has a small ceiling. A second plain
   LightGBM member at share 0.2 costs +50 MSE on the hold-out (residual
   correlation 0.986). An XGBoost study on the v97 columns (two trials,
   stopped on the user's decision) reached a stop-month RMSE of 170.8 s
   against 166.8 s for the N2 LightGBM study.
4. Blend weights failed twice (v77 cross-fitted +63 MSE; v98 +0.36 s live).
   Per-airport experts gained -0.22 s live (v89).

Choice: N7, a CatBoost member in the LIRF gate. It acts on the largest
block, and in the only component where recent changes transferred.

### 17.4 N7 price

`python -m src_v3.catboost_gate price` (`models/v3/catboost_gate.holdout.json`).
CatBoost Logloss, depth 6, learning rate 0.05, on the 176 g3 columns; 991
rounds from an early stop on months 11 and 12 (fit months 2 to 6 and 8 to 10).
Both classes are scored out of fold on the six month-pair folds of the
v64 protocol, and each class gets a nested isotonic map. The served gate is
the mean of the calibrated members. The bar, set before the run: -150 MSE on
the hold-out scale at the CatBoost share 0.5.

| CatBoost share | log loss, `sd` <= 3,600 s | 3,600 to 14,400 s | above 14,400 s | mixture MSE, Jan + Jul | Jan | Jul |
|---|---|---|---|---|---|---|
| 0 (g3, v105) | 0.3599 | 0.1213 | 0.1200 | 0 | 0 | 0 |
| 0.25 | 0.3569 | 0.1187 | 0.1195 | -347 | +20 | -368 |
| **0.5** | **0.3555** | **0.1179** | **0.1209** | **-484** | **+71** | **-555** |
| 0.75 | 0.3557 | 0.1184 | 0.1244 | -409 | +152 | -561 |
| 1.0 | 0.3578 | 0.1210 | 0.1310 | -124 | +263 | -387 |

Facts:

1. The pre-registered share 0.5 passes the bar: -484 MSE.
2. The calibrated scores of the two classes correlate 0.964.
3. The gain sits in July. January loses 71 MSE. A share above 0.5 loses
   more in January; 0.5 stays the served share, and no share is chosen
   on the hold-out.
4. Above 14,400 s the log loss rises slightly (0.1200 to 0.1209). Step A
   serves most of these rows, so the mixture does not use the gate there.

### 17.5 N7 live result

v109 = the v104 base recipe with `--gate-catboost catboost_p_fb_lirf_g3c.cbm
lirf_regime_g3c.isotonic.pkl`, then the v105 stack. Only LIRF rows move:
26,482 rows, mean +3.2 s, max 1,870 s. LIRF rows above 7,200 s rise from
92 to 100. Each new row is a late flight (`sd` 7,437 to 12,657 s) that the
higher gate moves toward `sd`. Live: **247.37 s, +0.89 s against v105**
(about +440 MSE). Rejected; N7 closes (stop rule 6.2).

Reading, grade D: the hold-out gain sat in July, and January lost. The
CatBoost target statistics fix the 2025 fallback rate of each stand and
operator. The 2026 rates moved (RYR arrival artefacts 0.130 in 2025, 0.267
in July 2026, F16.6). The g3 LightGBM gate reads that shift from the 2026
arrival-share columns. A new model class in the LIRF head without new
information has now failed four times (v49, v68, v100, v109).

### 17.6 N3: NNLS member weights per coverage class (v110)

`python -m src_v3.blend_nnls price` (`models/v3/blend_nnls.holdout.json`). Non-negative least squares
of the six hold-out members per coverage class, on rows with `0 < y <= 7,200` s; each month takes the
weights fitted on the other month; the price includes the stack and the A5 bound. The v105 shares are
0.1, 0.1, 0.2, 0.2, 0.2 and 0.2 (MLP).

| arm | base only | base, stack and bound | January | July |
|---|---|---|---|---|
| per coverage class | +97 | +139 | +1 | +138 |
| one global weight set | +44 | +118 | +37 | +81 |

The weights of the two months disagree: on rows without a clock the plain LightGBM member gets 0.014
from January and 0.195 from July. v110 served the per-class weights fitted on both months, on the
user's request: **246.57 s, +0.10 s against v105**. Rejected; N3 closes. The fixed v105 shares stay,
as v77 and v98 found.



The brief is `docs/AUDIT_PROMPT.md`, written for v67. This pass runs its
task A (budget), task C (new signal) and task D (floor) on v97. The plan
that acts on it is `docs/WINNING_PLAN.md`, third edition; its section 2
holds the full tables.

### 16.0 How this pass was made

1. Two modules rebuild the v97 hold-out prediction and the v41 LIRF head
   from saved members. They fit no model and run in about 2 min:
   - `python -m src_v3.measure_budget_v97` writes `models/v3/budget_v97.holdout.json`;
   - `python -m src_v3.measure_lirf_gap` writes `models/v3/lirf_gap.holdout.json`.
2. The leaderboard snapshot comes from the public API
   (`https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`,
   paged with `cursor=`), read on 2026-10-03 at 09:25 UTC.
3. The ADSB.lol feed list comes from the git tags of
   `adsblol/globe_history_2025` and `_2026`; the archive sizes come from
   the GitHub release API. Both were read on 2026-10-03.
4. Grades as in 0.2. A number names its JSON key or is arithmetic.

### 16.1 The served stack (v97)

| component | rows | recipe | source |
|---|---|---|---|
| LightGBM term | outside LIRF | plain members (3 seeds, MS3) mixed 50/50 with per-airport members; blended 0.5 with the members on `y - clip(MVT - AOBT_3, 0, 7,200)` | `release_v97.py`, `predict_v67.py` |
| CatBoost term | outside LIRF | plain member blended 0.5 with the anchored member | same |
| base | outside LIRF | 0.5 LightGBM term + 0.5 CatBoost term; every tree member reads the NM clocks and the OPDI and ADS-B columns of 350 days of 2025; the LightGBM members also read the v85 forward windows | same |
| MLP member | outside LIRF | 0.15 of the base; NM clocks and v85 forward windows, no OPDI or ADS-B column | `mlp_member.py` |
| stack | rows with an OPDI or ADS-B clock outside LIRF | LightGBM on `y - base`, trained on the out-of-sample v97 hold-out blend of 62 days, 600 rounds | `stack_pushback_v90.py`, `submission/kind-mango_v97.recipe.json` |
| post-processing | every row | MS1, MS2, then the A5 stand bound | `predict_v57.write_ms` |
| LIRF head | LIRF | v55 `R_norm` (5 seeds), v56 gate with the v64 out-of-fold map, Step A reads `R_norm` (v63), ITY340 hedge | `predict_v57.serve_v30` |

### 16.2 Budget (task A)

Hold-out outside LIRF, hold-out scale (`budget_v97.holdout.json`):

| coverage class | rows | MSE | MSE without the stack | RMSE, s |
|---|---|---|---|---|
| clock at the assigned stand | 50,300 | 1,495 | 1,536 | 101.2 |
| ADS-B ground segment only | 98,683 | 8,740 | 8,565 | 174.6 |
| OPDI only | 2,460 | 185 | 185 | 160.9 |
| no clock | 166,346 | 54,825 | 54,825 | 336.9 |
| total | 317,789 | 65,245 | 65,111 | |

Live scale, projected onto the 2026 coverage mix (`projection_2026`;
grade C) and with the LIRF term from 16.3 F16.5:

| block | live MSE | share of 61,448 |
|---|---|---|
| outside LIRF, no clock | 18,332 | 29.8 % |
| outside LIRF, segment only | 10,419 | 17.0 % |
| outside LIRF, stand clock | 2,604 | 4.2 % |
| outside LIRF, OPDI only | 224 | 0.4 % |
| LIRF | about 20,460 | 33.3 % |
| tail and 24-h outside LIRF | about 9,410 | 15.3 % |

### 16.3 Findings

**F16.1. The v95 stack adds nothing on the v97 base.** V, medium.
- Evidence: hold-out MSE 65,245 with the stack, 65,111 without (`mse`,
  `mse_no_stack`). The stack gains 41 on stand-clock rows and loses 175 on
  segment rows (`by_coverage`); EGLL segment rows lose 173
  (`by_airport_coverage`).
- Mechanism: since v97 the base members read the same clocks, learned on
  350 days. The stack relearns them from 62 days.
- Cost: 134 hold-out MSE. Serving the stack on stand-clock rows only gains
  175 against v97 (third-edition N1).

**F16.2. Rows with a clock at the assigned stand sit at their floor.** V, high.
- Evidence: on clean stand-clock rows the served RMSE is 93.8 s; the best
  single clock per row, chosen with the label, gives 92.4 s
  (`clocks.own_rows_served_rmse`, `clocks.own_rows_best_clock_rmse`).
  `adsb_own_move` lies within 60 s of the label on 74.4 % of clean rows.
- Mechanism: the rest is the gap between the reported off-block and any
  physical event in the track.
- Cost: none left to take. These rows hold 2,604 live MSE.

**F16.3. Segment-only rows do not show the pushback.** V, high.
- Evidence: 89.8 % of clean segment-only rows are first seen already
  moving; 24.4 % have no OSM position for the stand (`segment_rows`). RMSE
  by distance of the first point from the assigned stand: 107 to 109 s up
  to 300 m, 139 s at 300 to 600 m, 168 s at 0.6 to 1 km, 194 s at 1 to
  2 km, 255 s at 2 to 4 km.
- Mechanism: the track starts after the aircraft left the stand, so no
  clock marks the off-block. Why the apron is not seen is not measured.
- Cost: 10,419 live MSE in the block. A route distance to the stand (K3)
  and more receivers (K1) are the two ways in.

**F16.4. The January and July 2025 hold-out cannot price ADS-B levers at
EGLL and LEMD.** V, medium.
- Evidence: stand-clock share 2025 Jan+Jul against 2026: EGLL 0.037 and
  0.465, LEMD 0.003 and 0.138; LEMD segment share 0.032 and 0.567
  (`coverage_share`). EGLL had a stand clock on at most 10 % of
  departures in any 2025 month; LEMD carries segments on 86 to 95 % of
  departures from September to December 2025 (`by_month_2025`). EDDF lost
  stand clocks in 2026 (0.154 to 0.061).
- Mechanism: ADS-B ground coverage at EGLL and LEMD rose after the
  hold-out months. The cause is not measured.
- Cost: decision risk. The v97 gain at these airports is unpriced, and a
  2025 price misses them. Report prices per airport; use a September to
  December fold at LEMD.

**F16.5. The fallback gate holds half of the LIRF error.** V, high.
- Evidence: v41 head on the LIRF hold-out rows: 21,714 MSE; with a
  perfect gate 11,126 (`bands.mse`, `bands.mse_perfect_gate`). The 330
  null-record rows with a delay of 1 to 4 h hold 6,152 of it, and 672
  with a perfect gate. `R_norm` RMSE on genuine rows: 323.4 s. 2026 row
  counts per band match the hold-out within 15 % in every cell above
  100 rows (`by_band.rows_2026`).
- Mechanism: the mixture `p sd + (1 - p) R_norm` pays `p (1 - p) (sd -
  R_norm)^2` per row; delayed rows with an uncertain gate pay most.
- Cost: ceiling 10,588 hold-out MSE. Live LIRF about 20,460 (21,714
  minus the live LIRF gains of v55, v56, v63 and v64, 1,224 MSE,
  arithmetic). The hold-out band table and rate maps are in-sample, so
  the live LIRF error can be larger.

**F16.6. At LIRF the departure fallback follows the airline's arrival
artefact.** V, high.
- Evidence: per flight prefix and month of 2025, the departure fallback
  rate against the arrival rate of in-block within 60 s of the schedule:
  correlation 0.77 across 714 cells, 0.58 within an airline over months
  (`proxy.LIRF`). At the other nine airports: -0.09 to 0.12. Late LIRF
  departures (`sd` > 1,800 s): fallback rate 3.5 % when the same
  airline's same-day arrival share is 0.05 or less, 25.0 % at 0.2 to 0.4,
  88.8 % above 0.4; AUC 0.674 for the share alone, 0.788 for the static
  airline rate (`lirf_day_buckets`). RYR arrival share: 0.130 in January
  and July 2025, 0.267 in July 2026 (`lirf_shift_2026`).
- Mechanism: the correlation is measured, the cause is not. One reading:
  the same reporting chain sets the block time of both phases. The
  ranking file keeps arrival block times, so the 2026 share is known. The
  served gate reads static 2025 rates.
- Cost: unmeasured. Lever G1 prices it with the v64 protocol.

**F16.7. LIRF has partial ADS-B coverage.** V, high.
- Evidence: 7,900 of 26,899 LIRF ranking departures carry an ADS-B
  segment; 56,991 of 160,704 in 2025; 84 % in July 2025, 0 % from January
  to April 2025. The first point lies within 300 m of the assigned stand
  on 14.7 % (2026) and 12.2 % (2025) of those rows (`lirf_adsb`,
  `by_month_2025`).
- Mechanism: a departure seen moving at `t` before take-off has a taxi of
  at least `t`. `R_norm` reads no such column.
- Cost: unmeasured. Lever K2.

**F16.8. ADSB.lol publishes two feeds that the pipeline does not read.**
V, medium.
- Evidence: release tags per day in 2025: `prod-0` 353, `staging-0` 352,
  `mlatonly-0` 199; in 2026: 288, 262 and 293. On 2026-07-15 the
  `staging-0` archive is 4.11 GB in 3 parts, `mlatonly-0` 0.18 GB.
- Mechanism: a feed with other receivers adds ground points.
- Cost: unmeasured. Lever K1 probes one day first.

**F16.9. A second LIRF artefact: block equals `EOBT_1`.** V, low.
- Evidence: 12.1 % of genuine LIRF departures have the block within 60 s
  of `EOBT_1`; the other airports 2.8 to 7.0 %. With `mvt_eobt1` of
  1,800 to 3,600 s the LIRF share is 3.5 % (`eobt_copies`).
- Cost: small; test after G1 and K2.

**F16.10. The tail outside LIRF is mostly unobservable, and a long ADS-B
segment is not a hold.** V, medium.
- Evidence: tail and 24-h rows hold 32,176 of 65,245 hold-out MSE; LFPG
  rows with no clock hold 30,177 (`by_airport_class`). 450 hold-out rows
  have a segment above 21,000 s, 447 of them at EGLL; median label
  1,434 s, 7 above 7,200 s (`segment_rows.long_segments`).
- Cost: the LFPG rows are a floor. A rule on segment length would lift
  normal rows.

**F16.11. The leaderboard moved.** V, decision.
- Evidence: API, 2026-10-03, 09:25 UTC: 216.57 s (strong-crane), 218.85 s,
  219.07 s; tenth 229.90 s; kind-mango team rank 31.
- Consequence: 210 s is 6.6 s below the current top. Rank 3 needs 13,457
  live MSE (arithmetic).

### 16.4 Floor and reachable gap (task D)

1. Floor: the LFPG rows with no clock and the null-record 24-h rows have
   no observable. On the hold-out they hold 30,177 MSE; live, the tail
   remainder is about 9,410 (grade D).
2. Reachable gap: the third-edition levers sum to -1,100 to -15,100 live
   MSE (245.7 s to 215.3 s). 210 s needs -17,348. It needs a new
   observable beyond the list, most likely for the LIRF gate or for LTFM
   and LFPG, which have no clock on 82,576 2026 rows.

### 16.5 Measures for the next model

| finding | third-edition lever | test before upload |
|---|---|---|
| F16.1 | N1 stack on stand-clock rows only | measured: -175 hold-out MSE |
| F16.6 | G1 arrival artefact shares in the gate | `measure_gate_clocks --extra`, bar -150 |
| F16.7 | K2 ADS-B columns in `R_norm` | equal-round refit control on LIRF rows, bar -150 |
| F16.8 | K1 extra ADSB.lol feeds | one-day coverage probe, rule in WINNING_PLAN 4.3 |
| F16.3 | K3 route-distance clock | member -600, then blend -150 |
| F16.4 | every ADS-B lever | per-airport prices; a September to December fold at LEMD |
| F16.2 | none: stop clock work on stand-clock rows | |
| F16.10 | K4 long-hold rule; no rule on segment length | 2025 price, bar -150, row review |

### 16.6 Corrections to earlier statements

| earlier statement | this pass |
|---|---|
| WINNING_PLAN second edition 10, A6: "0 % ADS-B ground coverage at LIRF in 2026" | true for 2026-07-15 only; over 62 days 29 % of LIRF departures carry a segment (F16.7) |
| Final status table, MH3: closed after v83 | reopened at the airline level (F16.6) |
| WINNING_PLAN second edition 2.1: clean outside LIRF about 35,800 live MSE (v89) | normal classes outside LIRF about 31,578 for v97 (`projection_2026`) |
| RECAP v95 to v97: the stack carries the clock gain | on the v97 base the stack costs 134 hold-out MSE (F16.1) |

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
