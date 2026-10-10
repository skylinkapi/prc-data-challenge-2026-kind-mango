# Plan to the deadline, third edition: from 247.89 s toward 210 s

Status: 2026-10-03, 12:00 local time. Base: v97 (247.89 s), the working
tree on top of commit `2a4ec1b`. The v90 to v98 work is not in a commit
yet (step Q1).

Update, 2026-10-07: best 243.65 s (v116), rank 36; the top is 213.52 s.
`docs/MODEL_ANALYSIS.md` nineteenth pass audits v115 and v116 and adds
one id family, J (NM join rules). Section 4.12 holds J1 and J2; section 10
holds their state and the revised order of work to 11 October. No
measured lever reaches 220 s (MODEL_ANALYSIS 19.6).

This edition replaces the second edition of 2026-09-29. Appendix B keeps
the second edition word for word until it is in a commit; its lever ids
A1 to C1 stay valid. This edition uses four new id families: G (LIRF
head), K (clocks and coverage), N (models and frameworks), Q (process).

Scope: analysis and plan. Section 2 numbers come from two measurement
modules of 2026-10-03 and their records:

- `python -m src_v3.measure_budget_v97` writes `models/v3/budget_v97.holdout.json`.
- `python -m src_v3.measure_lirf_gap` writes `models/v3/lirf_gap.holdout.json`.

Every other number carries a source or the word "arithmetic". Grades as in
`docs/MODEL_ANALYSIS.md` 0.2: A measured live, B measured on the hold-out,
C arithmetic on measured parts, D reasoning only.

## 0. Conclusions

1. 247.89 s is 61,448 MSE. 210 s is 44,100 MSE. The gap is
   **17,348 MSE**, 28 % of the current error.
2. The leaderboard API (2026-10-03, 09:25 UTC) puts the top at 216.57 s
   (strong-crane), then 218.85 s and 219.07 s; tenth place is 229.90 s.
   kind-mango is team rank 31. 210 s is 6.6 s below the current top.
   Rank 3 needs 13,457 MSE (arithmetic).
3. Live budget of v97, grade C (section 2.3): LIRF about 20,460 MSE
   (33 %); rows outside LIRF without an ADS-B or OPDI clock 18,332 (30 %);
   rows with an ADS-B ground segment but no clock at the assigned stand
   10,419 (17 %); tail and 24-h rows outside LIRF about 9,410 (15 %);
   rows with a stand clock 2,604 (4 %).
4. Rows with a clock at the assigned stand sit at their floor. Their
   served RMSE is 93.8 s. The best single clock per row, chosen with the
   label, gives 92.4 s. Clock work on these rows cannot move the score.
5. The v95 stack adds nothing on the v97 base: hold-out 65,245 MSE with
   the stack, 65,111 without. It gains 41 MSE on stand-clock rows and
   loses 175 on segment rows.
6. A perfect LIRF fallback gate would cut the LIRF hold-out MSE from
   21,714 to 11,126. At LIRF only, an airline's departure fallback rate
   follows its arrival rate of in-block on the schedule: correlation 0.77
   across airline-months, 0.58 within an airline over months. The ranking
   file keeps arrival block times, so this signal exists for 2026
   (lever G1). v83 tried a weaker form, keyed on the stand and the day.
7. LIRF has an ADS-B ground segment on 7,900 of 26,899 ranking departures
   (29 %) and on 84 % of July 2025 departures. The second edition closed
   A6 on a one-day probe that showed 0 %. A6 reopens as K2.
8. ADSB.lol publishes two daily feeds that the pipeline does not read:
   `planes-readsb-staging-0` (262 days of 2026; 4.11 GB on 2026-07-15) and
   `planes-readsb-mlatonly-0` (293 days of 2026; 0.18 GB). Their ground
   coverage is unmeasured (lever K1).
9. The ranked levers of section 3 sum to -1,100 to -15,100 live MSE:
   245.7 s to 215.3 s, central 231.0 s (arithmetic). Prices do not add,
   so the sum is an upper bound. **210 s needs about 2,250 MSE more than
   the high end of every lever.** Rank 3 (219.07 s) needs 89 % of the high
   end. The order of work puts the new observables first.
10. Deadline: 2026-10-11, 23:59:59 CET. Upload before 21:00 UTC on that
    day. The limit is 5 uploads per UTC day; 3 remain on 2026-10-03. A
    top-3 file without a reproduction path wins no prize (Q2).
11. Added 2026-10-07: the NM join window `|BLOCK - LOBT| <= 3,606` s holds
    on every 2025 record departure. Two rules from it (J1, J2; section
    4.12) price -564 hold-out MSE with no fit. With every open lever at
    its high end the score reaches about 240.8 s; 220 s needs 10,965 MSE
    and no measured lever or open data source reaches it
    (`docs/MODEL_ANALYSIS.md` 19.6).

## 1. What changed since the second edition

Live results from the official result files.

| upload | live, s | delta, s | change | state |
|---|---|---|---|---|
| v90 | 263.17 | -1.17 | A1: OPDI pushback clock in a stack on the v89 blend | accepted |
| v91 | 259.39 | -3.78 | A2: ADSB.lol pushback clocks, 24 ranking days | accepted |
| v92 | 253.82 | -5.57 | A2 on all 62 ranking days; A5 stand bound | accepted |
| v93 | 251.44 | -2.38 | stack trained on 21 days of 2025 | accepted |
| v94 | 250.51 | -0.93 | MLP member at 0.15 in the base outside LIRF | accepted |
| v95 | 249.39 | -1.12 | stack trained on all 62 January and July 2025 days | accepted |
| v96 | 249.92 | +0.53 | ADS-B specialist as a stack column | rejected |
| v97 | 247.89 | -1.50 | every base member refit with the OPDI and ADS-B columns of 350 days of 2025 | accepted, best |
| v98 | 248.25 | +0.36 | serve-time shares chosen on the hold-out | rejected |

Lessons:

1. New row-level observables transferred at 1 to 2 times their hold-out
   price (v90 to v95, v97). A derived column and serve-time parameters did
   not: v96 priced -302 MSE and landed +0.53 s; v98 priced about -120 MSE
   and landed +0.36 s.
2. More training days for a component that reads a new observable paid
   twice: the stack on 7, 21 and 62 days (-2.38 s, -1.12 s), then the base
   on 350 days (-1.50 s).
3. From v89 to v97, new observables and more training days for them gave
   -15.52 s; one new model class (the MLP member) gave -0.93 s
   (arithmetic on the table). The top fell from 224.88 s (2026-09-26) to
   216.57 s (2026-10-03) in the same period.

## 2. Where the error sits (v97)

### 2.1 Hold-out outside LIRF

Source: `budget_v97.holdout.json`, keys `by_class` and `by_coverage`.
Hold-out scale (344,336 rows). The prediction follows the v97 serve path:
the v97 member blend, the MLP member at 0.15, the v95 stack cross-fitted
January against July, and the A5 stand bound.

| label class | rows | MSE | MSE without stack | RMSE, s |
|---|---|---|---|---|
| clean | 292,319 | 30,530 | 30,408 | 189.6 |
| fallback (`abs(y - sd) <= 60`) | 25,339 | 2,450 | 2,447 | 182.5 |
| tail (`7,200 < y <= 80,000`) | 34 | 11,931 | 11,921 | 10,992 |
| 24-h (`y > 80,000`) | 1 | 20,245 | 20,245 | |
| low (`y < 30`) | 96 | 89 | 90 | |
| total | 317,789 | 65,245 | 65,111 | |

| coverage class | rows | MSE | MSE without stack | RMSE, s |
|---|---|---|---|---|
| ADS-B clock at the assigned stand (`adsb_own`) | 50,300 | 1,495 | 1,536 | 101.2 |
| ADS-B ground segment only (`adsb_seg`) | 98,683 | 8,740 | 8,565 | 174.6 |
| OPDI clock only | 2,460 | 185 | 185 | 160.9 |
| no clock (`none`) | 166,346 | 54,825 | 54,825 | 336.9 |

Facts:

1. Clean rows with an error above 300 s are 7.5 % of the clean rows and
   hold 69 % of the clean MSE. Above 600 s: 1.3 % of the rows, 41 % of the
   MSE (`clean_concentration`).
2. `adsb_own_move` lies within 60 s of the label on 74.4 % of clean rows
   after the airport median bias. On clean stand-clock rows the served
   RMSE is 93.8 s against 92.4 s for the best clock per row (`clocks`).
3. Of the clean segment-only rows, 89.8 % are first seen already moving,
   and 24.4 % have no OSM position for the assigned stand
   (`segment_rows`). The pushback itself is not in the track.
4. On those rows the RMSE grows with the distance of the first point from
   the assigned stand: 109 s up to 150 m, 107 s at 150 to 300 m, 139 s at
   300 to 600 m, 168 s at 0.6 to 1 km, 194 s at 1 to 2 km, 255 s at 2 to
   4 km (`segment_rows`).
5. Tail and 24-h rows hold 32,176 of the 65,245 hold-out MSE; LFPG rows
   with no clock hold 30,177 of it (`by_airport_class`). They have no
   observable: that is a floor.

### 2.2 Coverage moved between 2025 and 2026

Share of departures per coverage class (`coverage_share`). OPDI-only rows
are left out of the table (5.5 % of 2026 LSZH rows and 4.8 % of EDDF).

| airport | 2026 stand clock | 2026 segment | 2026 none | 2025 Jan+Jul stand clock | segment | none |
|---|---|---|---|---|---|---|
| EDDF | 0.061 | 0.619 | 0.272 | 0.154 | 0.715 | 0.130 |
| EDDM | 0.396 | 0.567 | 0.038 | 0.254 | 0.359 | 0.383 |
| EGLL | 0.465 | 0.391 | 0.144 | 0.037 | 0.286 | 0.678 |
| EHAM | 0.519 | 0.432 | 0.049 | 0.502 | 0.464 | 0.034 |
| LEBL | 0.346 | 0.568 | 0.086 | 0.283 | 0.612 | 0.104 |
| LEMD | 0.138 | 0.567 | 0.294 | 0.003 | 0.032 | 0.965 |
| LFPG | 0.001 | 0.121 | 0.878 | 0.000 | 0.046 | 0.953 |
| LSZH | 0.364 | 0.521 | 0.060 | 0.333 | 0.513 | 0.052 |
| LTFM | 0 | 0 | 1.000 | 0 | 0 | 1.000 |

Consequences:

1. The January and July 2025 hold-out cannot price an ADS-B lever at EGLL
   or LEMD. EGLL had a stand clock on at most 10 % of departures in any
   2025 month; LEMD had no segment in January and February and 7 to 9 % in
   July and August, against 86 to 95 % from September to December
   (`coverage_share.by_month_2025`). Price such levers per airport, and at
   LEMD also on a September to December fold.
2. EDDF lost stand clocks in 2026 (0.154 to 0.061) and gained
   uncovered rows (0.130 to 0.272). The members learned a 2025 coverage mix.

### 2.3 Projection onto 2026, live scale

Per-row hold-out MSE of the normal classes (clean, fallback, low) per
airport and coverage cell, times the 2026 rows of the cell
(`projection_2026`). A cell with fewer than 1,000 hold-out rows takes the
pooled value of its coverage class. Grade C: the method assumes no drift
inside a cell.

| block | 2026 rows | live MSE | share of 61,448 |
|---|---|---|---|
| outside LIRF, no clock | 115,869 | 18,332 | 29.8 % |
| outside LIRF, segment only | 124,230 | 10,419 | 17.0 % |
| outside LIRF, stand clock | 74,804 | 2,604 | 4.2 % |
| outside LIRF, OPDI only | 3,039 | 224 | 0.4 % |
| LIRF, all classes | 26,899 | about 20,460 | 33.3 % |
| tail and 24-h outside LIRF | | about 9,410 | 15.3 % |

- LIRF: hold-out 21,714 (v41 head, `lirf_gap.holdout.json` key
  `bands.mse`) minus the live LIRF gains of v55, v56, v63 and v64
  (1,224 MSE, arithmetic on the result files). The hold-out LIRF number
  is optimistic: the band table and the rate maps are in-sample.
- Tail: the remainder 29,870 (`remainder_lirf_and_tail`) minus LIRF.
- Normal classes outside LIRF by airport: LTFM 7,619; LFPG 6,569; EGLL
  4,529; LEBL 3,153; EDDF 3,146; LEMD 2,674; EHAM 1,852; LSZH 1,044; EDDM
  992 (`projection_2026.by_airport`).

### 2.4 LIRF

Source: `lirf_gap.holdout.json`. The v41 head on the LIRF hold-out rows:
`R_norm` and gate out of sample; band table and rate maps in-sample. The
perfect gate puts `p` = 1 on fallback rows and 0 elsewhere; Step A and
ITY340 rows keep their value.

| delay `sd`, flight record | rows Jan+Jul 2025 | rows 2026 | fallback rate | MSE | MSE, perfect gate |
|---|---|---|---|---|---|
| up to 1,800 s, record | 14,897 | 15,349 | 0.275 | 1,694 | 1,258 |
| 1,800 to 3,600 s, record | 7,443 | 7,330 | 0.085 | 3,304 | 1,800 |
| 3,600 to 7,200 s, record | 3,047 | 3,077 | 0.065 | 4,653 | 1,762 |
| 7,200 to 14,400 s, record | 632 | 665 | 0.017 | 649 | 507 |
| 3,600 to 7,200 s, null record | 194 | 167 | 0.227 | 1,737 | 362 |
| 7,200 to 14,400 s, null record | 136 | 138 | 0.353 | 4,415 | 310 |
| above 14,400 s (Step A cells and others) | 157 | 138 | | 4,837 | 4,786 |
| other null-record rows | 22 | 35 | | 425 | 340 |
| total | 26,528 | 26,899 | | 21,714 | 11,126 |

Facts:

1. The gate costs up to 10,588 hold-out MSE. The 330 null-record rows
   with a delay of 1 to 4 h hold 6,152 MSE; a perfect gate keeps 672 of it.
   Record rows with a delay of 0.5 to 2 h cost 4,395 above a perfect gate.
2. In every cell above 100 rows, the 2026 count lies within 15 % of the
   hold-out count (table). The LIRF hold-out structure transfers to the
   ranking rows.
3. `R_norm` RMSE on genuine rows: 323.4 s (`bands.r_norm_rmse_genuine`).
   Outside LIRF, clean rows without a clock serve 240 s at LTFM and LFPG.
4. Airline-month proxy (`proxy.LIRF`): the departure fallback rate against
   the arrival rate of in-block within 60 s of the schedule, per flight
   prefix and month: correlation 0.77 across 714 cells, 0.58 within an
   airline over months. Per operator: 0.72 and 0.57. At the other nine
   airports both correlations lie between -0.09 and 0.12.
5. Airline-day proxy on late departures (`sd` > 1,800 s, 60,773 rows of
   2025, `lirf_day_buckets`): the fallback rate is 3.5 % where the same
   airline's same-day arrival artefact share is 0.05 or less, 25.0 % at
   0.2 to 0.4 and 88.8 % above 0.4 (89 rows). AUC of the share alone
   0.674; of the static airline rate 0.788.
6. 2026 shift (`lirf_shift_2026`): the RYR arrival artefact rate is 0.130
   in January and July 2025, 0.136 in January 2026 and 0.267 in July 2026.
   RYR has 3,188 LIRF departures in 2026. The served gate reads static
   2025 rates and cannot see this.
7. A second artefact (`eobt_copies`): on 12.1 % of genuine LIRF departures
   the block equals `EOBT_1` within 60 s; elsewhere 2.8 to 7.0 %. Where
   it costs (`mvt_eobt1` 1,800 to 3,600 s) the share is 3.5 %. Small.

## 3. Ranked levers

Live MSE on the 344,841-row scale; negative is a gain. The hold-out bar is
on the 344,336-row scale and is set before the run. Hours are working
hours on the data machine, fits included.

| rank | id | lever | evidence | live MSE (low, high) | grade | hours | uploads | hold-out bar |
|---|---|---|---|---|---|---|---|---|
| 1 | G1 | airline-day and airline-month arrival artefact shares in the LIRF gate | 2.4 facts 4 to 6 | -300, -2,500 | C | 6 | 1 | -150 on LIRF rows |
| 2 | K2 | ADS-B columns in the LIRF `R_norm` (A6 reopened) | section 0 item 7; 84 % of July 2025 LIRF rows | -200, -1,500 | C | 5 | 1 | -150 on LIRF rows |
| 3 | K1 | ADSB.lol `staging-0` and `mlatonly-0` feeds, unioned with `prod-0` | section 0 item 8; probe first | 0, -3,000 | D | 2 probe; about 17 download; 6 build | 1 to 2 | probe rule 4.3 |
| 4 | K3 | route-distance clock for segment rows | 2.1 fact 4 | -300, -2,000 | D | 10 | 1 | member -600, blend -150 |
| 5 | N5 | 2026 stand-clock rows as pseudo-labelled training rows | 74,804 such 2026 rows; 74.4 % within 60 s | 0, -2,000 | D | 8 | 1 | blend -150 in a 2025 simulation |
| 6 | N2 | Optuna study of the LightGBM members on the v97 columns | v76 study: member -965, live -465 | -200, -800 | C | 4 study, 6 refit | 1 | blend -150 |
| 7 | N4 | neural member on the v97 columns: MLP first, TabM if the MLP gains | v94 MLP without ADS-B columns: -213 hold-out, -470 live | -100, -1,000 | D | 6 | 1 | blend -150 |
| 8 | K4 | ADS-B long-hold rule for the tail | tail about 9,410 live | 0, -1,500 | D | 4 | 1 | -150, plus a row review |
| 9 | N1 | stack on stand-clock rows only | 2.1 coverage table | 0, -300 | B | 1 | 1 | met: -175 |
| 10 | N3 | blend shares per coverage class | the MLP reads no clock | 0, -500 | D | 2 | 0 to 1 | -150, cross-fitted |
| - | Q1 | commit the v90 to v98 work and this edition | 0 status line | 0 | required | 1 | 0 | |
| - | Q2 | reproduction path for the final file | eligibility page | 0 | required | 8 | 0 | parity 0.0 s |

Sum: -1,100 to -15,100 live MSE (arithmetic). The rank orders the levers
by expected gain per hour; new observables come first because they
transferred and derived columns did not (section 1, lesson 1).

## 4. Lever details

### 4.1 G1: arrival artefact shares in the LIRF gate

Mechanism. On about 20 % of LIRF departures the airport reports the
scheduled time as off-block (`lirf_regime_v64.meta.json`, rate per `sd`
bin). The arrivals of the same airline show the same artefact (2.4 facts
4 and 5). The served gate reads 14 static 2025 fallback-rate columns; it
cannot see a 2026 change such as RYR in July (2.4 fact 6). v83 tried a
stand-and-day form of this link: hold-out -277
MSE, live +0.14 s. G1 keys on the airline, where the 2025 evidence is
stronger (0.77 across cells against 0.43 per airport-day, RECAP v68 row).

Build, following the v83 chain:

1. Write `src_v3/build_arr_artefact.py` from `build_arr_link_v83.py`. For
   each departure at LIRF, from the LIRF arrivals of the same file:
   - `art_pre_day`: share of arrivals of the same flight prefix
     (`FLIGHT_mvt[:3]`, the gate's key) on the same UTC day with
     `abs(BLOCK - SCHED) <= 60` s, smoothed toward the airport-day share
     with `ARTEFACT_SMOOTH_K` = 10;
   - `art_pre_day_n`: the arrivals behind it;
   - `art_pre_month`: the same share per prefix and month;
   - `art_day`: the airport-day share.
   Use every arrival of the day; the organiser allows "what is coming"
   (RECAP, Discord 2026-09-09). Run one code path for the 2025 training
   files and the ranking file. Key on `MVT_ID_mvt` and assert one row per
   id.
2. Price: `python -m src_v3.measure_gate_clocks --tag arr_art_g1 --extra
   models/arr_art_train_g1.parquet`. It runs the v64 protocol: six
   month-pair fold boosters per arm, a nested isotonic map, and the served
   mixture with the out-of-sample v41 `R_norm` members on January and July.
   Bar: -150 MSE on the hold-out scale. Report log loss per `sd` bin.
3. Refit: copy `train_p_fb_v83.py` with the new columns (v56 recipe,
   207 rounds, 12 months, out-of-fold isotonic map).
4. Serve: the v97 `predict_v67` command (`release_v97.serve`) with
   `--gate-booster`, `--gate-iso`, `--gate-features` and the rank columns
   in `--extra`; then the v97 stack command. Only LIRF rows may move.
5. Label-free check before upload: the mean gate per prefix and month in
   2026 against 2025. The RYR July mean must rise. LIRF rows above
   7,200 s: count against v97.

Stop: one upload. If the hold-out misses the bar, close G1. Do not try
other keys; v83 already spent the stand and day keys.

### 4.2 K2: ADS-B columns in the LIRF `R_norm`

Mechanism. A LIRF departure seen moving on the ground at time `t` before
take-off has a genuine taxi of at least `t`. `R_norm` (RMSE 323.4 s on
genuine rows) reads no such column. In 2025, 56,991 LIRF departures carry
a segment; July 2025, a hold-out month, has 84 % (`lirf_adsb`,
`coverage_share.by_month_2025`).

1. Fit `R_norm` on LIRF genuine rows (`abs(y - sd) > 60`, `y < 80,000`)
   with the columns of `adsb_clock_train` (without `adsb_day_loaded`) and
   of `opdi_own_train_v90` added; v55 recipe; train months 2 to 6 and 8 to
   10, early stop on 11 and 12 (`measure_catboost_r_norm.py` shows the
   split).
2. Price as a replacement of the five v41 members inside the served
   mixture on the LIRF hold-out rows; equal rounds against a refit control
   without the columns. Bar: -150 MSE.
3. If it passes, refit five seeds on 12 months (`train_r_norm_v55.py`
   pattern). The LIRF head reads its own frame in `predict_v30.main`; add
   an argument that joins the ADS-B columns by `MVT_ID_mvt` for `R_norm`.
4. Then, separately, test the same columns in the gate (G1 protocol).

### 4.3 K1: two more ADSB.lol feeds

Probe, about 2 h, no upload:

1. Add a `--feed` argument to `fetch_adsb_days.py` (`prod-0` default,
   `staging-0`, `mlatonly-0`); `C.ADSB_URL` holds the tag pattern. Keep
   each feed's points in its own folder.
2. Download 2026-07-15 for both feeds (4.11 GB and 0.18 GB).
3. Run `build_adsb_points` per feed. For the departures of that day,
   compare per airport the share with a ground segment and with a stand
   clock: `prod-0` alone against the union of all feeds (deduplicate on
   `icao` and `ts`).
4. GO if the union adds 5 points or more of segment-or-clock share at one
   of EDDF, LEMD, LFPG, LIRF or LTFM, or 5 points of stand-clock share at
   any airport. Otherwise close K1.

If GO:

5. Download the 62 ranking days first, then January and July 2025, then
   the other 2025 days while time remains. A day takes about 8 min at
   8.4 MB/s (RECAP v91 row). Delete the raw archives after extraction.
6. Rebuild both clock tables from the union. A new train table changes
   every member: price with `measure_adsb_base_v97 fit` per member and
   `price`, as for v97, then refit and serve as `release_v97.py` does.
7. Attribution: same ODbL 1.0 licence and line as the `prod-0` feed.

Value, arithmetic: if 30 % of the uncovered LTFM and LFPG rows (82,576)
reach segment quality (per-row MSE about 29,000 against 55,500), the gain
is about 1,900 MSE.

### 4.4 K3: route-distance clock for segment rows

Mechanism. The aircraft is first seen while taxiing on 89.8 % of segment
rows. The trees read straight-line distances (`adsb_own_d0`, `adsb_dist0`).
The taxi route from the stand to the first point is longer, and it differs
by stand and runway.

1. In `build_adsb_clock.py`, snap the assigned stand and the first segment
   point to the nearest node of the OSM taxi graph
   (`src/features_osm_path.py`). Add `adsb_route_d0`, the shortest-path
   distance; `adsb_route_frac`, that distance over the stand-to-runway
   path; `adsb_t_est`, `adsb_seg + adsb_route_d0 / v`, with `v` the
   airport median taxi speed between the stand and the first point on
   2025 stand-clock rows.
2. Price on one LightGBM member (v97 recipe, MB5 parameters, equal
   rounds) with `measure_adsb_base_v97 fit lgb`, with and without the
   three columns. Bar: member -600 MSE, which maps to about -150 in the
   blend (v76: -965 to -240; v85: -604 to -208).
3. If it passes, refit all members as v97 did.

### 4.5 N5: 2026 stand clocks as pseudo-labels

Policy first. The rows are ranking-file features with an ADS-B clock.
No label and no score enters. Record the decision in the README stance
section before the first upload that uses it.

1. Pseudo-label: on stand-clock rows with `adsb_own_dmin` <= 50 m and
   `adsb_gap_max` <= 120 s, `y_pseudo = adsb_own_move - b`, where `b` is
   the airport median of `adsb_own_move - y` on 2025 clean rows.
2. Simulation on 2025: add the pseudo-labelled stand-clock rows of
   January and July to the fit months of one LightGBM member; score
   January and July against the true labels on the rows without a stand
   clock. Equal rounds against the member without them. Bar: -150 in the
   blend.
3. If it passes, add the 2026 pseudo rows (up to 74,804 before the
   filter) to every full-year member.

Limit: LTFM and LFPG have no 2026 stand clocks and gain nothing.

### 4.6 N2: Optuna study on the v97 columns

1. Run `tune_lgbm_v76.py` on the v97 column set (forward windows, OPDI
   and ADS-B tables added to the study frame), 24 trials, seeded sampler,
   trial 0 = the v76 parameters (`--control`), MB3 rows, fit months 2 to 6
   and 8 to 10, early stop on 11 and 12.
2. Price the best trial against the v97 hold-out member with
   `measure_tune_v76.py`, then inside the v97 blend. Bar: -150 in the
   blend.
3. If it passes, refit plain and anchored (3 seeds each) and the
   per-airport members. About 6 h; run detached overnight.

### 4.7 N4: neural member on the v97 columns

1. `mlp_member.py` reads the v85 columns only. Point it at the v97 set
   (NM clocks, forward windows, OPDI, ADS-B) and run `measure` against
   `blend_v97_holdout.parquet`. Price weights 0.1, 0.15, 0.2 and 0.3.
2. If the MLP gains 150 or more in the blend, try TabM (`tabm` package
   on PyTorch). Check its licence for GPLv3 compatibility before it goes
   into `requirements.txt`, and state the reason there (AGENTS.md 2).
3. Serve at the optimum of the quadratic through the hold-out prices, as
   v94 did.

### 4.8 K4: long holds in the tail

1. List the 2026 rows where the track shows a move away from the stand
   more than 7,200 s before take-off, and the served value. A5 already
   floors each row at `min(adsb_own_last, adsb_push)`.
2. Price on all 350 days of 2025 a floor `max(pred, adsb_move - k)` on
   rows whose first point is stationary and whose segment has no gap above
   180 s, for `k` in 0, 120 and 300 s. Bar: -150.
3. Do not read a long segment as a hold: 450 hold-out rows have a segment
   above 21,000 s, 447 of them at EGLL. Their median label is 1,434 s and
   7 of them lie above 7,200 s (`segment_rows.long_segments`). These
   aircraft sat on the ground with the transponder on; the segment does
   not measure the taxi.

### 4.9 N1: stack on stand-clock rows only

Add a `--stack-rows own` option to `stack_pushback_v90 serve`: the stack
moves a row only where `adsb_own_move` or `adsb_own_last` exists. Segment
and OPDI rows keep the v97 base. A5 stays on every row. Hold-out: -175 MSE
against v97 (2.1, coverage table). It needs no fit; upload it first, alone.

### 4.10 N3: blend shares per coverage class

Fit the CatBoost, anchored, per-airport and MLP shares per coverage class
on the v97 hold-out members, cross-fitted January against July (the v98
grid of `release_v98.py`, per class). A global grid failed live (v98).
Accept a cross-fitted gain of 150 or more only, and upload it alone.

### 4.11 Considered and deferred

| option | reason it ranks below the table |
|---|---|
| sequence model (1D CNN or GRU) over the raw ADS-B ground track | stand-clock rows are at their floor (2.1 fact 2); segment rows lack the pushback in the track (2.1 fact 3); 12 to 16 h to build; reopen only if K3 shows route geometry carries signal |
| CatBoost study (depth, learning rate) | one full-year fit takes 2.6 to 5 h (RECAP rows v67 and v87); the `l2_leaf_reg` 10 variant lost (v80) |
| XGBoost or a constant-leaf LightGBM member | closed: residual correlation 0.975 and 0.983 with the blend (v84, v77) |
| TabPFN | built for small tables; the base trains on 1.9 M rows |
| per-airport CatBoost members | the per-airport LightGBM members gained only -0.22 s live (v89) |

### 4.12 J1 and J2: NM join rules (added 2026-10-07)

Mechanism. Every 2025 departure with an NM record has
`|BLOCK - LOBT| <= 3,606` s (2,062,577 rows, no exception): the organiser
joined a movement to its NM flight only inside that window. The served
path ignores it (MODEL_ANALYSIS F19.1 to F19.3).

J1, one upload, no fit:

1. At LIRF, set the gate to 0 on record rows with
   `|SCHED - LOBT| > 3,606` s, before the mixture. A schedule copy would
   break the window there. Add a gate hook to `predict_v30.main`, as
   `r_norm_post` changes `R_norm`.
2. After the stack, the A5 bound and `step_a_date`, project every value of
   a record row onto `MVT - LOBT` plus or minus 3,606 s.
3. Label-free gate: no row leaves the window; no row without `LOBT`
   moves; outside LIRF only the rows outside the window move.

Price: hold-out -371 MSE (outside LIRF -256, LIRF -116;
`python -m src_v3.measure_nm_join`, `models/v3/nm_join.holdout.json`).
On v116, 54 rows lie outside the window; their projection guarantees a
live gain of at least 80.5 MSE.

J2, one upload after J1: at LIRF, on rows without a record, take the
`LOBT` from the arrival row of the same flight number at the destination
(landing 10 min to 14 h after the take-off). Where
`|SCHED - LOBT| <= 3,606` s, set the gate to 0: a schedule copy would have
joined. Price: hold-out -193 MSE (20 rows); 20 ranking rows move.

Stop: AGENTS.md rule 11.5 applies. Conflict for the team to decide: the
window part guarantees only 80.5 MSE, about 0.17 s, under the 0.3 s bar,
yet it cannot raise any row's error. If J1 lands between the best minus
0.3 s and the best, the team decides whether later uploads build on it.
Result: v118 landed at 242.81 s, -0.85 s against v116, so the bar holds and
the conflict did not arise. Every later upload builds on v118.

J3, added 2026-10-07: `mvt_lobt` is clipped at 7,200 s, so the trees do
not see rows whose `LOBT` lies 2 to 4 h before the take-off. There the
label is `MVT - LOBT` minus a median `BLOCK - LOBT` of 634 to 1,040 s per
band (2025), and the served path under-predicts by 860 s on average.
`lobt_window --anchor` serves that value: hold-out -255 MSE on 49 rows
(medians fit outside January and July; `nm_join.holdout.json`, key `j3`);
93 ranking rows. Below 7,200 s the anchor loses (+41 to +138 MSE).

J4, closed 2026-10-07: the arrival-side `AOBT_3` as a clock for departures
without a record serves worse than the model (RMSE 646 against 473 s on
246 hold-out rows; +38 to +139 MSE).

Rule 12 applies to every J rule: J2 moves 20 rows and J3 93, so each rides
with a model change (v119: J2 with `tabm_plr`; v120: J3 with `tabm_j`).

## 5. Order of work and uploads to 11 October

UTC days. The next free version is v99.

| day | work | uploads |
|---|---|---|
| Sat 3 Oct | Q1 (team approval). N1: serve and upload as v99. K1 probe. G1 builder and price. | 1 to 2 |
| Sun 4 Oct | G1 upload if it passed. K2 price. K1 decision; if GO, start the ranking-day downloads. N2 study overnight, detached. | 1 to 2 |
| Mon 5 Oct | K2 upload if it passed. N2 refits overnight. N4 MLP price. | 1 to 2 |
| Tue 6 Oct | N2 and N4 uploads. K3 builder and price. **Checkpoint 1.** | 1 to 2 |
| Wed 7 Oct | K3 upload. N5 simulation. K1 rebuild and price if GO. | 1 to 2 |
| Thu 8 Oct | N5 and K1 uploads. K4. **Hard stop for new feature families at 23:59 UTC.** | 1 to 2 |
| Fri 9 Oct | Q2: parity test and manifest for the best file; `REPRODUCE.md`; README. | at most 1 |
| Sat 10 Oct | Q2: Tier-1 run on a clean clone; fix what fails. N3 if time remains. | at most 1 |
| Sun 11 Oct | No new model. Last upload before 21:00 UTC. | 0 to 1 |

Planned total: 9 to 15 uploads of 43.

## 6. Stop rules

- Per upload: accept only if live is at or below the best minus 0.3 s.
  Retrain noise is about 0.15 s; v98 moved +0.36 s on a -120 MSE price.
- Per lever: close after one rejected upload, unless a written defect
  fix exists. The deadline leaves no second try.
- One change per upload. The live delta of a bundle cannot be split.
- Checkpoint 1, Tue 6 Oct, 23:59 UTC: if the best is above 240 s, drop
  N5 and K4, and spend the time on the levers that passed and on Q2.
- Hard stop: no new feature family after Thu 8 Oct, 23:59 UTC.
- Machine: one full-frame fit at a time (32 GB RAM). Run fits longer than
  30 min as detached processes with a log in `logs/`.

## 7. Closed levers: do not repeat

Everything closed in Appendix B (second edition 3.1 and 10) stays closed:
A3, A4, B1, B2, the ADS-B specialist as a stack column (v96), a global
serve-time share search (v98), XGBoost (v84), CatBoost variants (v69, v70,
v80, v87), fitted blend weights (v77), LIRF-head model changes without new
information (v49, v68, v78, v83), tail rules that pull outliers to the
2025 tail, MS3 on the MB3 base, operator encoder refits (v58, v60), MB1 on
`mvt_eobt1` (v54), the calendar block (v52), weather at `EOBT_1` (v66),
arrival drift (v59), neighbour NM clocks (L2).

Closed by the measurements of 2026-10-03:

- More clock work on stand-clock rows: 93.8 s served against 92.4 s for
  the best clock per row.
- A long ADS-B segment as a tail signal (4.8 step 3).

Not allowed on policy grounds: seed lotteries selected by live score,
per-row probing, files that differ from the best on a few rows, OSN state
vectors, Trino.

## 8. Risks

| risk | effect | control |
|---|---|---|
| the hold-out has few clocks at EGLL and LEMD | a lever's price misses those airports | report prices per airport; use a September to December fold at LEMD |
| G1 repeats v83 (a LIRF gate change that did not transfer) | +0.1 to +0.4 s | one upload; reject on a miss; no other key |
| the extra feeds copy `prod-0` | 2 h lost | probe rule set before the download |
| pseudo-labels read as use of the ranking process | eligibility | team decision in the README first; features only, no labels, no scores |
| CPU contention between N2 refits and K1 parsing | late levers | one full-frame fit at a time; parse at night |
| the v90 to v98 work is not in a commit | loss of nine versions | Q1 first |
| no reproduction path for the final file | no prize | Q2 on Fri 9 and Sat 10 Oct |

## 9. Decisions for the team

1. Approve Q1: commit v90 to v98 and this edition on a branch, then merge.
2. Approve G1, K2 and K1 as the first three levers.
3. Decide on N5 (pseudo-labels from 2026 ADS-B clocks) and record the
   stance in `README.md`.
4. Keep Fri 9 to Sun 11 Oct for Q2 and the final upload.
5. Set the season target at Checkpoint 1. The levers give 231 s central
   and 215 s at the high end of every range; 210 s needs a new observable
   beyond this list, most likely for the LIRF gate or for LTFM and LFPG.

## 10. Status of every third-edition lever

Fill one row per lever as it closes.

| id | lever | upload | live, s | state |
|---|---|---|---|---|
| N1 | stack on stand-clock rows only | v99 | 248.21 (+0.32) | rejected: the stack helps on segment rows live; the 2025 hold-out misprices segment rows (F16.4) |
| G1 | arrival artefact shares in the LIRF gate | v101 | 247.69 (-0.20) | accepted, best: hold-out -52 MSE, live about -98 |
| K2 | ADS-B columns in the LIRF `R_norm` | v100 | 248.01 (+0.12) | rejected: hold-out -267 MSE on genuine rows did not transfer, like every LIRF-head model change |
| K1 | `staging-0` and `mlatonly-0` feeds | | | closed: the union adds at most 0.6 points of coverage (`adsb_feeds_probe.json`) |
| K3 | route-distance clock | | | closed: member +164, blend plus stack -28 MSE (`swap_k3.holdout.json`) |
| N5 | 2026 pseudo-labels | | | closed: day-split simulation member +34, blend +7 MSE (`pseudo_n5.holdout.json`) |
| N2 | Optuna study on the v97 columns | v108 | 246.49 (+0.02) | rejected: hold-out member -330 MSE did not transfer |
| N4 | neural member on the v97 columns | v105 | 246.48 (-0.48) | accepted, best: hold-out -64 MSE, live about -235 |
| K4 | long-hold rule | | | closed: 4 hold-out candidates, none above 7,200 s |
| N3 | blend shares per coverage class | v110 | 246.57 (+0.10) | rejected, closed: NNLS weights per class, cross-fitted hold-out +139 MSE |
| N6 | XGBoost member on the v97 columns | | | stopped after two study trials on the user's decision (stop RMSE 170.8 s against 166.8 s for N2) |
| N7 | CatBoost member in the LIRF gate, mean with g3 | v109 | 247.37 (+0.89) | rejected, closed: hold-out -484 MSE (January +71, July -555) did not transfer |
| N8 | TabM member on the v97 columns at 0.3 on top of the v105 base | v111 | 245.23 (-1.25) | accepted, best: hold-out full path -231 MSE (MODEL_ANALYSIS eighteenth pass) |
| N8b | TabM 24-epoch member at 0.35, plus the Step A date-crossing fix | v112 | 244.29 (-0.94) | accepted, best |
| N8c | TabM 32-epoch member, three seeds, at 0.4 | v114 | 244.02 (-0.27) | under the 0.3 s bar |
| K5 | X-Plane stands at EDDM and EHAM; ATFM slot columns in TabM | v115 | 243.69 (-0.33) | accepted, best |
| K5b | X-Plane stands also at LEMD, LEBL and LFPG | v116 | 243.65 (-0.035) | inside noise |
| K5c | tree members refit on the v116 clock tables | v117 | 243.74 (+0.09) | rejected |
| N9 | TabM member in the LIRF gate | v113 | 247.40 (+3.12) | rejected, closed: LIRF-head model changes keep failing live |
| Q1 | commit | | | done: `8114b81` on branch `plan-v99-third-edition` |
| Q2 | reproduction path for the final file | | | open |
| J1 | LOBT window on every record row; LIRF gate 0 where a schedule copy breaks the window | v118 | 242.81 (-0.85) | accepted, best: hold-out -371 MSE, live -411 |
| J2 | LIRF gate 0 on rows without a record whose arrival row gives `LOBT` within 1 h of `SCHED` | v119 | 242.51 (-0.295) | leaderboard best, bundled with N10 for rule 12; 0.005 s under the bar |
| N10 | second network family: TabM with periodic embeddings (`tabm_plr`), averaged with `tabm_v116` | v119 | 242.51 (-0.295) | bundled with J2; full path -83 MSE on the hold-out |
| J3 | rows with `LOBT` 2 to 4 h before take-off at `MVT - LOBT` minus the band median of `BLOCK - LOBT` | v120 | 242.91 (+0.40) | rejected, closed, with `tabm_j` at the TabM share 0.5 (bundled for rule 12); hold-out -255 and -17 MSE, all of J3 on rows above 3,600 s |
| J4 | arrival-side `AOBT_3` as a clock for departures without a record | | | closed: +38 to +139 MSE |
| K6 | OPDI `entry-deicing_pad` events as stack columns | | | open: probe one January day; OSN sees the ground only at LSZH, LEBL, EDDF, EHAM, LEMD |
| Q3 | price levers by the gain on rows with `y <= 3,600` s and a 2026 importance weight | | | open: backtest on the uploads v105 to v117 first |
| P1 | LFPG departures with `MVT` off the minute grid raised to `MVT - LOBT` minus 242 s (MODEL_ANALYSIS F20.2) | v121 | 242.03 (-0.49) | accepted, best: bundled with N10 seeds for rule 12; hold-out -250 and +34 MSE, live -235 |
| N10b | six TabM seeds per family (`tabm_v116_b`, `tabm_plr_b`) | v121 | 242.03 (-0.49) | bundled with P1; hold-out +34 MSE (stack-refit noise, grade D) |
| N10c | `tabm_j` as a third TabM family at the share 0.5, without J3 (reopened by F20.3) | v122 | 241.84 (-0.18) | leaderboard best, 0.11 s under the bar: hold-out -42 MSE, live -88 |
| N11 | TabM share 0.65 (neural weight 0.72) | v123 | 241.79 (-0.06) | leaderboard best, under the bar; no hold-out price (user directive) |
| N12 | fourth TabM family `tabm_plr_j` (periodic embeddings, J column set) | v124 | 241.67 (-0.12) | leaderboard best, under the bar |
| L15 | NM clocks in the LIRF gate (g5) | v125 | 242.35 (+0.67) | rejected, closed: hold-out +108, live +326 MSE |
| N13 | fifth TabM family `tabm_s` (structure terms of the record) | v126 | 241.66 (-0.01) | leaderboard best, inside noise |
| K7 | no-record departures linked to their ADS-B and OPDI tracks by flight number (F22.1) | v127 | 240.70 (-0.95) | accepted, best |
| N14 | stack with coverage weights and tuned leaves (F22.3) | v128 | 240.60 (-0.11) | leaderboard best, under the bar |
| K7b | the five TabM families refit on the K7 tables | v129 | 239.72 (-0.87) | accepted, best |
| E1 | EUROCONTROL label-derived series | | | decision: organiser ruling needed; the daily series makes the LIRF gate worse (+523 MSE) |

Revised order of work, 2026-10-07 (MODEL_ANALYSIS 19.7): Wed 7 Oct J1;
Thu 8 Oct J2, N10 prices on the GPU, K6 probe (hard stop for new feature
families at 23:59 UTC); Fri 9 Oct N10 full-year members and Q2; Sat 10 Oct
Q2 on a clean clone; Sun 11 Oct last upload before 21:00 UTC.

Added during execution (2026-10-03 and 04): the LIRF gate family with 2026 arrival information. v101 (G1 airline
shares) -0.20 s, v102 (plus the v83 stand link) -0.55 s, v104 (plus operator shares, gate g3) -0.19 s; v107 (g4, plus exact-copy
and stand-day shares) +0.21 s, rejected. A 63-leaf stack (v103) +0.20 s, rejected. The hold-out underrated every accepted change:
the user rule of 2026-10-03 is to finish and upload a lever that misses its bar and judge it live.

## Appendix A. Rerun the measurements

Both modules read saved hold-out members and need no fit. Together they
run in about 2 min.

```bash
python -m src_v3.measure_budget_v97   # models/v3/budget_v97.holdout.json
python -m src_v3.measure_lirf_gap     # models/v3/lirf_gap.holdout.json
```

`measure_budget_v97` reads `models/v3/blend_v97_holdout.parquet`, the MLP
hold-out file and `external/adsblol/v97/adsb_clock_train.parquet`. After a
lever changes the members, point `BLEND` at the new hold-out blend and
rerun it: the coverage table shows where the lever acts.

## Appendix B. Second edition, 2026-09-29 (superseded)

This appendix keeps the second edition word for word until it is in a commit.
After that commit, replace the appendix with one line:
`git show <hash>:docs/WINNING_PLAN.md`. Section numbers below are those of the
second edition; other files cite them as "second edition".

**Plan to the deadline, second edition: from 264.34 s toward 220 s**

Status: 2026-09-29. Base commit: `2a4ec1b` (`origin/main`, v89).
This edition replaces the edition of 2026-09-23. The lever ids L1 to L15
in `README.md`, `RECAP.md` and `docs/MODEL_ANALYSIS.md` refer to that
edition: `git show 2a4ec1b:docs/WINNING_PLAN.md`.
Scope: analysis and plan only. No number in this file comes from a new
model run. Every number carries a source or the word "arithmetic".
Grades as in `docs/MODEL_ANALYSIS.md` 0.2: A measured live, B measured on
a partial frame, C arithmetic or scenario, D reasoning only.

### 0. Conclusions

1. The metric is RMSE in **seconds**. v89 scores 264.34 s = 69,874 MSE.
   220 s = 48,400 MSE. The gap is **21,474 MSE**, 31 % of the current
   error. The leaderboard top was 224.88 s on 2026-09-26
   (`docs/MODEL_ANALYSIS.md:9`): 19,303 MSE below v89.
2. The current signal set is saturated. After v74, every accepted upload
   gained 0.9 s or less (v75 to v89: -2.50 s in total). Model class, blend
   weights, a second tuning study, XGBoost, CatBoost variants and every
   LIRF-head model change are closed (`RECAP.md` rows v77 to v87).
   `MODEL_ANALYSIS.md:179`: "No measured signal is left above the 150 MSE
   bar." The remaining small levers sum to less than 1,000 MSE. They
   cannot reach 220 s.
3. One new information source can close a large part of the gap: the
   **scored flight's own pushback time, taken from open ADS-B data**.
   The organiser allowed exactly this on 2026-09-18: "if you can find
   open trajectory data with good ground coverage and extract off-block
   times, we are ok" (`RECAP.md:415-424`). ADSB.lol (ODbL 1.0) counts as
   open data (`RECAP.md:410-414`). OPDI counts as open data
   (`RECAP.md:392`). Trino access stays forbidden (`RECAP.md:406`).
4. The repo already holds OPDI ground events for 2025-01 to 2026-07
   (`src/fetch_opdi_events.py`) and a per-flight pushback table
   (`src/build_opdi_taxi_v2.py`, `external/opdi/taxi_out_v2.parquet`).
   The pipeline uses it only for **other** flights (`opdi_live_*`,
   removed in v26). No served column links a row to its own pushback.
5. OPDI pushback records in 2026 (`models/coverage_monitor.json`, key
   `opdi_taxi_out_counts`): EDDF 5,582 (Jan) and 11,388 (Jul), LSZH
   5,630 and 12,588, EDDM 1 and 995. Every other airport has 65 or fewer. Upper
   bound of the covered share: EDDF 47 %, LSZH 79 %, EDDM 4 % of the
   ranking rows (arithmetic on the row counts in
   `submission/kind-mango_v89.check.json`).
6. ADSB.lol publishes one archive per day for all 31 days of January and
   July 2026 and for at least 350 days of 2025 (git tags of
   `adsblol/globe_history_2025` and `_2026`, read 2026-09-29; May 2025
   has 27 days, June 2025 has 20). A participant measured a first
   position near the gate for about 40 % of EDDM flights
   (`RECAP.md:417-419`). Coverage at the other airports is unmeasured.
7. Arithmetic (section 2): if an ADS-B clock with a residual RMSE of
   90 s covers a share `c` of the clean non-LIRF rows, the gain is about
   `28,900 * c` MSE. **220 s needs `c` of about 0.74** (0.66 at a 60 s
   residual). Lower bounds, neighbour columns and tail catches reduce the
   need to about 0.55 to 0.65 (grade C).
8. Target after the probe, not before. OPDI only (`c` about 0.10): about
   259 s. `c` = 0.4: about 242 s. `c` = 0.6: about 230 s. The day-1
   probe (section 4.1) measures `c`.
9. `REPRODUCE.md` rebuilds v67 only. No parity test exists for v73 to
   v89. A top-3 file without a reproduction path is not prize-eligible
   (eligibility page, `docs/PRC_Data_Challenge_2026_BRIEF.md:250-255`).
   The plan reserves 2 days for it.
10. Deadline: 2026-10-11, 23:59:59 CET. Upload before 21:00 UTC on that
    day. 13 UTC days remain, today included. The limit is 5 uploads per
    UTC day.

### 1. What changed since the edition of 2026-09-23

Live results from `README.md` and `RECAP.md`.

| upload | live, s | delta, s | change | state |
|---|---|---|---|---|
| v61 | 288.32 | +0.99 | v45 parameters on the v57 recipe (L4) | rejected |
| v63 | 287.09 | -0.24 | Step A normal term reads `R_norm` (L12) | accepted |
| v64 | 285.95 | -1.14 | out-of-fold isotonic map on the gate (L10) | accepted |
| v65 | 284.74 | -1.22 | `plan_nm_taxi` (L9) | accepted |
| v66 | 285.07 | +0.33 | METAR at `EOBT_1` (L11) | rejected |
| v67 | 281.87 | -2.87 | CatBoost blended 0.5 (L3) | accepted |
| v68, v71 | 281.90, 281.85 | +0.03, -0.015 | CatBoost in `R_norm`; runway-queue columns | rejected |
| v73 | 270.58 | -11.29 | NM clocks of the scored flight (L1, track B) | accepted |
| v74 | 266.84 | -3.73 | clocks in CatBoost; anchored LightGBM (L5) | accepted |
| v75 | 266.15 | -0.70 | anchored CatBoost | accepted |
| v76 | 265.27 | -0.88 | MB5 study on the clock-era base | accepted |
| v83 | 265.41 | +0.14 | arrival artefact link in the LIRF gate | rejected |
| v85 | 264.55 | -0.72 | forward windows of NM clocks (L8) | accepted |
| v87 | 264.86 | +0.31 | forward columns in both CatBoost members | rejected |
| v89 | 264.34 | -0.22 | per-airport LightGBM members, 50/50 | accepted, best |

Closed on the hold-out since 2026-09-23 (no upload): converged and
second base CatBoost (v69, v70), fallback heads outside LIRF (MH2),
blend weights and a constant-leaf member (v77), `R_norm` study (v78,
held), anchored study (v79), CatBoost `l2` and Lossguide (v80), stand
sequence (v81), second MB5 study and XGBoost (v84), neighbour clocks
(L2), clocks in `R_norm` (L6), clock bias per stand, `ARVT_3`, clocks in
the gate.

Reading: one new observable (the NM clock) gave -15.0 s in two uploads.
Model and feature work around it gave -2.5 s in six uploads. The next
large step needs a new observable, not a new model.

### 2. Error budget and the arithmetic of 220 s

#### 2.1 Live budget, estimated

The live file has no per-class score. This table transfers hold-out
shares to the live scale. Grade D.

| class | rows | estimated live MSE | source |
|---|---|---|---|
| clean, outside LIRF | about 294,000 | about 35,800 (RMSE about 205 s) | `RECAP.md:73`, v84 error budget; row share 0.925 of 317,942 |
| LIRF, all classes | 26,899 | about 17,000 to 20,500 | v41 hold-out LIRF 21,715 (`MODEL_ANALYSIS.md:263`) minus live LIRF gains of v55, v56, v63, v64 (-1,222) |
| tail outside LIRF (`7,200 < y <= 80,000`) | about 35 | about 8,000 to 12,000 | hold-out 11,962 in 35 rows (`RECAP.md:73`) |
| fallback outside LIRF | about 23,800 | about 2,500 | hold-out 2,559 (`RECAP.md:73`) |
| sum | | 63,300 to 70,800 | live 69,874 |

The sum matches the live score only without a 24-h row outside LIRF.
The clean class holds about half of the error. LIRF holds a quarter from
8 % of the rows.

#### 2.2 Gain from an ADS-B clock on clean rows

`gain = N * c * (sigma_now^2 - sigma_adsb^2) / 344,841`, with
`N` = 294,000 clean non-LIRF rows and `sigma_now` = 205 s (arithmetic).

| `sigma_adsb` | gain per unit of `c`, MSE |
|---|---|
| 60 s | 32,760 |
| 90 s | 28,923 |
| 120 s | 23,552 |

Share `c` of clean non-LIRF rows that a pushback clock must cover:

| target | MSE to gain | `c` at 60 s | `c` at 90 s | `c` at 120 s |
|---|---|---|---|---|
| 250 s | 7,374 | 0.23 | 0.26 | 0.31 |
| 240 s | 12,274 | 0.37 | 0.42 | 0.52 |
| 230 s | 16,974 | 0.52 | 0.59 | 0.72 |
| 225 s | 19,249 | 0.59 | 0.67 | 0.82 |
| 220 s | 21,474 | 0.66 | 0.74 | 0.91 |

Limits of this table:

- `sigma_now` differs per airport. EGLL, LFPG and LTFM sit above 205 s;
  LEMD and LSZH sit below (grade C estimate, section 3). A clock at EGLL gives more than a clock at
  LSZH. Replace the flat value with the per-airport hold-out RMSE of the
  v89-style blend before the decision in 4.1.
- Tail rows, LIRF rows and lower bounds add gain outside this table.
- `sigma_adsb` is unmeasured. The label is the airport-reported
  off-block. Its capture method differs by airport (automatic docking
  systems, surface radar, manual input). The probe measures it.

#### 2.3 Why the ADS-B clock is different from the NM clock

- `AOBT_3_flt` behaves like a planned taxi time. Its median error is
  175 s. On rows with `y > 7,200` s its median is 1,082 s, so it does not
  see long holds (`README.md:62-66`). At LIRF it is about 900 s on every
  row (`RECAP.md:81`).
- An ADS-B surface track observes the pushback itself. Its error comes
  from the label definition and the detection, not from a plan.
- Row-level observables transferred to live at 0.8 to 1.6 times the
  hold-out price (v67 -1,627 for -2,079; v73 -6,237 for about -3,865).
  Feature additions and LIRF-head changes did not (v66, v68, v71, v83).
- The top fell from 235.31 s (2026-09-23, section 5.1 of the first
  edition) to 224.88 s (2026-09-26), after the ruling of 2026-09-18.
  This fits an ADS-B clock. It proves nothing. Grade D.

### 3. Ranked levers

MSE is on the 344,841-row live scale; negative is a gain. Hours are
collaborator hours on the data machine. The hold-out bar is set before
the run and uses the hold-out scale.

| rank | id | lever | data | live MSE | grade | hours | uploads | hold-out bar |
|---|---|---|---|---|---|---|---|---|
| 1 | A2 | own pushback from ADSB.lol, all airports | ADSB.lol, 124 days | -4,000 to -18,000 | C | 16 to 24 | 2 to 3 | -1,000 |
| 2 | A1 | own pushback from OPDI at EDDF, LSZH, EDDM | on disk | -1,500 to -3,500 | C | 6 | 1 | -300 |
| 3 | A5 | long-hold and return-to-stand detector for the tail; MS1 exemption | A1/A2 tracks | 0 to -3,000 | D | 4 | 1 | -150 |
| 4 | A3 | lower bound for rows first seen already moving | A1/A2 tracks | -500 to -3,000 | D | 4 | 1 | -150 |
| 5 | A6 | own pushback in the LIRF genuine term | A2, if FCO coverage >= 20 % | 0 to -2,000 | D | 3 | 1 | -150 on LIRF genuine rows |
| 6 | A4 | neighbour ADS-B taxi times, backward and forward windows | A1/A2 tracks | -100 to -600 | D | 3 | 0 to 1 | -150 |
| 7 | B1 | Step A analogue outside LIRF: null record, large `sd` | files | +300 to -800 | D | 2 | 0 to 1 | -150 |
| 8 | B2 | 3 seeds for the per-airport members | files | 0 to -150 | C | 2 | 0 to 1 | -150 |
| - | C1 | reproduction path and parity test for the final file | | 0 | required | 8 | 0 | parity 0.0 s |

A1 arithmetic: EDDF 15,800 rows at 198 s, LSZH 16,700 rows at 182 s,
residual 90 s: -2,700 MSE, about 259 s. The per-airport values of
`sigma_now` are grade C: v41 per-airport clean MSE (`MODEL_ANALYSIS.md:
254-266`) times 0.777, the ratio of the v84 clean budget to the v41
clean budget.

B1 note: outside LIRF, only 22 of 36,432 rows with more than 2 h of
delay have `y` near `sd` (`RECAP.md:73`). The null-record subset is
unmeasured. Measure first. Ship only if the rule moves at most 50 rows
in 2026. On rows with an A1 or A2 clock, the clock replaces B1.

#### 3.1 Closed levers: do not repeat

Everything in section 10 of the first edition stays closed. Add: CatBoost variants
and refits (v69, v70, v80, v87), XGBoost (v84), blend-weight fits (v77),
LIRF-head model changes of any kind without new information (v49, v68,
v78, v83), clocks in the gate and in `R_norm` (L6), neighbour NM clocks
(L2), clock bias per stand, `ARVT_3`, queue lengths on top of forward
windows (v85 note). Do not start a CatBoost fit after 2026-10-02: one fit
takes 2 to 5 h, and the CPU goes to A1 and A2.

Not allowed on policy grounds, unchanged: seed lotteries selected by
live score, per-row probing, and files that differ from the best on a
handful of rows.

### 4. ADS-B build

#### 4.1 Probe, day 1: go or no-go

No upload. About 4 h of work plus the download time.

**P1. OPDI own match (data on disk).**

1. Link each departure to its OPDI record. Keys: same airport, the
   **last** OPDI `entry-runway` before `MVT` in `[MVT - 600 s, MVT]`, and
   the stand of the first `exit-parking_position` equal to `STAND_mvt`
   where both exist. `build_opdi_taxi_v2.py` keeps the **first** runway
   entry, which can be a runway crossing; do not reuse its
   `actual_taxi_sec`. If two records compete, keep the one with the
   smaller `MVT - runway entry`.
   Download the OPDI flight lists only if this match is ambiguous
   (`.../OPDI/v002/flight_list/flight_list_YYYYMM.parquet`).
2. Assert one record per `MVT_ID_mvt`.
3. Report per airport and month, 2025 and 2026: matched share, share by
   `pushback_source` (`exit_park`, `entry_tax`).
4. Report on 2025 rows: bias `median(y - (MVT - pushback_ts))` per
   airport and source; RMSE after the bias; share within 60 s after the
   bias; the same numbers for `MVT - AOBT_3` on the same rows.

**P2. ADSB.lol, 4 days.**

1. Download `v2025.01.15`, `v2025.07.15`, `v2026.01.15`, `v2026.07.15`
   (`planes-readsb-prod-0`). Record the size and the download time of
   each day. Multiply by 124 days for the full plan. Stop and replan if
   the full download takes more than 2 days. The disk holds at most 2 raw
   days at a time.
2. Stream the split tar. Keep traces with a ground point within 6 km of
   one of the 10 airports. Delete the raw archive after the pass.
3. Match and detect as in 4.2 and 4.3.
4. Report per airport, 2026 days: share of departures in Q1, Q2 and Q3.
   Report per airport, 2025 days: the P1 accuracy numbers for Q1 rows.

**Decision rule, set now.**

| probe result | plan | expected live |
|---|---|---|
| Q1 >= 0.25 at 4 or more non-LIRF airports, and Q1 RMSE after bias <= 100 s | GO: full A2 build | from the table in 2.2 |
| only EDDF, LSZH, EDDM pass | A1 plus A2 at those airports | about 255 to 259 s |
| no airport passes | A1 only; B1; then C1 and the paper | about 259 to 262 s |

Compute the expected `c` from the probe days. Write the target of the
season into `README.md` on the same day.

#### 4.2 Match a departure to a track

1. For each trace, find each take-off near the airport: the first
   airborne point after a ground point within 6 km of the airport
   reference point (`external/airports/airports.csv`).
2. Candidates: same airport, take-off within 120 s of `MVT_TIME_UTC_mvt`.
3. Rank candidates: ADS-B type `t` equal to `AIRCRAFT_TYPE_mvt`; then a
   first ground point near stand `STAND_mvt` (OSM stands from
   `src/build_osm_stands.py`); then the smaller time difference.
   Callsigns are a check, not a key: `FLIGHT_mvt` holds the commercial
   flight number, and ATC callsigns are often alphanumeric.
4. Assign one track per departure and one departure per track per
   airport-day.
5. Keep `adsb_dt_to` = ADS-B take-off minus `MVT` as a quality column.

#### 4.3 Detect the pushback

Use the ground points of the matched track in `[MVT - 6 h, MVT]` within
the airport area.

1. A point is stationary if the ground speed is at most 1 kt and the
   position moves at most 15 m in 60 s.
2. Walk back from the runway entry. Find the last stationary interval of
   120 s or more within 80 m of an OSM stand, or inside an apron polygon
   and more than 300 m from a runway.
3. `adsb_aobt` is the first point after that interval that moves.
4. Classes:
   - **Q1**: step 2 found an interval, and no gap between `adsb_aobt`
     and the take-off exceeds 180 s.
   - **Q2**: the first ground point already moves and lies within 150 m
     of a stand. The true off-block is earlier than this point.
   - **Q3**: the first ground point already moves, far from any stand.
   - **Q0**: no ground point.
5. Return to stand: two stationary intervals at a stand, with movement
   between them. Keep the first movement time as well.

#### 4.4 Columns, keyed on `MVT_ID_mvt`

One builder serves the training days and the ranking file through the
same code path. Assert one row per id.

| column | definition |
|---|---|
| `adsb_src` | 0 none, 1 OPDI, 2 ADSB.lol, 3 both within 60 s |
| `adsb_q` | 0 to 3, as in 4.3 |
| `adsb_taxi` | `MVT - adsb_aobt`, Q1 only, clip 0 to 7,200 s |
| `adsb_taxi_raw` | the same value without the clip, for the tail |
| `adsb_taxi_lb` | `MVT - first ground point`, Q2 and Q3 |
| `adsb_stop_max` | longest stop between `adsb_aobt` and take-off |
| `adsb_rts` | 1 if the track shows a return to stand |
| `adsb_first_push` | `MVT - first movement from a stand`, for `adsb_rts` rows |
| `adsb_gap_max` | largest gap between `adsb_aobt` and take-off |
| `adsb_dt_to` | ADS-B take-off minus `MVT` |
| `adsb_day_loaded` | 1 if the builder read ADS-B data for that day |

`adsb_day_loaded` separates "no coverage" from "day not downloaded" on
2025 rows. On 2026 rows it is 1 everywhere.

#### 4.5 Integration, smallest diff first

1. **Specialist on Q1 rows (A1, then A2).** A LightGBM member with the
   v85 columns plus the ADS-B columns. Target `y - adsb_taxi`, clipped at
   +/-7,200 s, as the anchored member of v74. Rows: Q1 rows of the loaded
   2025 days; for A1 that is every 2025 month with OPDI records. Serve on
   Q1 rows as `(1 - w) * v89 + w * (adsb_taxi + offset)`, before MS1 and
   MS2, as `predict_v67.py` does for every member. Price `w` in
   {0.5, 0.7, 0.85, 1.0} on the hold-out. Rows outside Q1 keep v89.
2. **Lower bound (A3).** On Q2 and Q3 rows, a stack layer on the
   out-of-sample hold-out predictions of the v89-style blend: inputs
   blend prediction, `adsb_taxi_lb`, `adsb_q`, airport. Cross-fit January
   against July 2025. Compare with the plain rule
   `max(blend, adsb_taxi_lb + airport median excess)`. Ship the better
   one only if it passes the bar.
3. **Tail (A5).** Exempt a row from the MS1 caps when Q1 shows a long
   hold (`adsb_taxi_raw > 7,200` s, `adsb_gap_max <= 180` s). For
   `adsb_rts` rows, give the specialist `adsb_first_push` as a column.
4. **LIRF (A6).** Only if the probe shows Q1 >= 0.2 at LIRF. Replace the
   `R_norm` term with the specialist on Q1 rows. Keep the gate: the label
   of a fallback row is `sd` whatever the track shows.
5. **Neighbours (A4).** Mean and count of the Q1 `adsb_taxi` of other
   departures at the airport and on the runway, windows as in v85.
   Last in the order: small gain, and only at covered airports.

A full retrain of both LightGBM bases with the ADS-B columns comes last,
and only if the specialist passes. One LightGBM refit takes about 45 min
per seed (`RECAP.md` v85 row).

#### 4.6 Price on the hold-out

- Build the out-of-sample hold-out prediction of the v89-style blend once
  (v46 months) and save it, for example as
  `models/v3/blend_v89_holdout.parquet`. Every ADS-B price reads it.
- A1: fit on the other 10 months. The hold-out months January and July
  2025 hold 19,312 OPDI records at LSZH and 622 at EDDF
  (`coverage_monitor.json`). LSZH carries the price. Add fold months
  where EDDF has 1,000 or more records.
- A2: download January and July 2025 in full. Cross-fit the specialist:
  fit on January, score July; fit on July, score January. Compare with
  the saved out-of-sample predictions of the v89-style blend on the same
  rows. The final specialist fits on both months.
- Equal-round arms. One full-frame fit at a time (32 GB RAM).
- Report per airport and per class: clean, fallback, tail, 24-h.

#### 4.7 Label-free gate for every ADS-B upload

Run `src/check_submission_2026.py` against v89, plus these checks:

1. Q1 share per airport and month in 2026 against the probe.
2. Quantiles 10, 50 and 90 of `adsb_taxi` per airport, 2026 against 2025.
3. Every row that moves by more than 3,000 s is Q1 with
   `adsb_gap_max <= 180` s and `abs(adsb_dt_to) <= 60` s.
4. Non-LIRF rows above 7,200 s rise only on rows with such Q1 support.
5. LIRF rows move only if A6 runs.

#### 4.8 Data volume and machine

- The ADSB.lol day size is unmeasured here. Measure it in P2 before the
  full download.
- Order of downloads: January and July 2026 first (62 days, needed to
  serve), then January and July 2025 (62 days, needed to fit and price).
- Downloads run next to model fits. Trace parsing runs at night: it
  competes for the CPU.
- Start with `gzip` and `json` from the standard library. Add `orjson`
  only if one day takes more than 10 min to parse. State the reason in
  `requirements.txt` (`AGENTS.md` 2).

#### 4.9 Policy, licence and stance

- Post one written question on Discord `#prc-data-competition` today:
  "We extract the pushback time of the scored departure from OPDI v002
  flight events and ADSB.lol globe_history (ODbL 1.0), with attribution.
  Is this allowed for a prize-eligible entry?" Record the reply in
  `README.md`. The ruling of 2026-09-18 already allows it; the question
  makes the record complete.
- Add ADSB.lol and the OPDI flight events to the data-source table of
  `README.md`, with the licence and the attribution line.
- The ruling of 2026-09-18 excludes raw OSN state vectors
  (`RECAP.md:413-414`). OPDI is a published derived dataset and stays
  allowed (`RECAP.md:392`). Do not read OSN state vectors or Trino.
- Do not commit derived ADS-B tables. ODbL makes a derived database
  share-alike. Commit the builder, not its output.
- Extend the stance section of `README.md` with one paragraph for
  ADS-B pushback times and the quote of 2026-09-18.

### 5. Plan to 11 October

UTC days. Version numbers are the next free ones; v90 is the first.

| day | work | uploads |
|---|---|---|
| Tue 29 Sep | Discord question. P1 on the OPDI data on disk. Start the P2 downloads. | 0 |
| Wed 30 Sep | P2 extraction and table. Decision rule of 4.1. A1 specialist, priced. If GO: start the full ADSB.lol download, 2026 first. | 0 to 1 (v90 = A1) |
| Thu 1 Oct | Builder for A2 on the loaded days. B1 measurement while the CPU is free. | 0 to 1 |
| Fri 2 Oct | A2 specialist, cross-fitted on January and July 2025, priced. **Checkpoint 1**: apply the stop rule of 6. | 1 to 2 (A2) |
| Sat 3 Oct | A3 lower bound. A5 tail detector and MS1 exemption. | 1 to 2 |
| Sun 4 Oct | A6 if the LIRF probe passed. A4 if time remains. | 1 to 2 |
| Mon 5 Oct | Full retrain of both LightGBM bases with the ADS-B columns, if the specialist passed. **Checkpoint 2**. | 1 to 2 |
| Tue 6 Oct | Stack the accepted levers. Final specialist fit on all loaded 2025 days. | at most 2 |
| Wed 7 Oct | C1: `REPRODUCE.md` for the final stack; `test_vNN_parity.py`; METAR fetch script (`REPRODUCE.md:159`). | at most 2 |
| Thu 8 Oct | C1: README refresh. **Hard stop for new feature families at 23:59 UTC.** | at most 2 |
| Fri 9 Oct | Tier-1 run on a clean clone. Fix what fails. | at most 1 |
| Sat 10 Oct | Rerun the final recipe with fixed seeds for the record. | at most 1 |
| Sun 11 Oct | No new model. Last upload before 21:00 UTC. | 0 to 1 |

Planned total: 12 to 20 uploads of about 60. Keep the rest unused.

### 6. Stop rules

- **Per upload.** Accept only if live is at or below best minus 0.3 s.
  Retrain noise is about 0.15 s (v52, v57).
- **Per lever.** Close a lever after two rejected uploads. Retry once
  only after a written defect fix.
- **Checkpoint 1, Fri 2 Oct.** If the A2 specialist misses its hold-out
  bar of -1,000 MSE, stop A2 work and keep A1. Move to B1, then C1 and
  the JOAS paper.
- **Checkpoint 2, Mon 5 Oct.** If the best is above 250 s, drop A4 and
  A6. Spend the time on C1.
- **Hard stop.** No new feature family after Thu 8 Oct 23:59 UTC.

### 7. Risks

| risk | effect | control |
|---|---|---|
| ADS-B ground coverage is low at LFPG, LIRF, LEMD, LTFM (organiser, `RECAP.md:422-424`) | `c` stays low; 220 s out of reach | probe first; target set from the measured `c` |
| coverage differs between 2025 and 2026 (OPDI EDDF: 7 records in January 2025, 5,582 in January 2026) | a model that reads presence as signal shifts in 2026 | specialist on Q1 rows only; `adsb_day_loaded`; per-airport gate 4.7 |
| wrong track matched | a 3 h error costs 338 MSE per row; 30 such rows cost about 10,000 MSE | type and stand check; `abs(adsb_dt_to) <= 120` s; one-to-one assignment; gate 4.7.3 |
| label off-block defined differently per airport | bias per airport and stand | the specialist learns the bias; report it in P1 |
| fallback rows: the label is `sd`, not the pushback | a clock can overrule a correct hedge | serve weight `w` below 1 if the price says so; LIRF keeps its gate |
| download or disk limit | A2 does not finish | 2026 days first; A1 does not depend on A2 |
| the top keeps falling | 220 s is not first place on 11 Oct | the prize needs top 3 and a reproduction path, not first place |

### 8. Decisions for the team

1. Adopt A1 and A2. The organiser allows them; post the question of 4.9
   anyway.
2. Accept ODbL for derived ADS-B tables: attribution in `README.md`, no
   derived table in git.
3. Set the season target after the probe with the table in 2.2. Write it
   in `README.md`.
4. Reserve Wed 7 Oct to Sat 10 Oct for C1. Without C1, a top-3 rank does
   not win a prize.

### 9. Repo corrections to make with the next commit

1. `README.md` "What the model is", "Quick start" and "Reproduction"
   describe v41 or v67. The best file is v89.
2. `REPRODUCE.md` rebuilds v67 only. No parity test exists for v73 to v89.
3. `README.md` layout: add `src_v3/` files of v73 to v89 and the new
   ADS-B builders.
4. `README.md` data sources: add ADSB.lol and the OPDI flight events for
   the scored flight.
5. `docs/PRC_Data_Challenge_2026_BRIEF.md` lists 11 airports with LTAI.
   The ranking file holds 10 (`submission/kind-mango_v89.check.json`).
   Add one line to the brief.

### 10. Status of every lever, 2026-10-01

Live best: **249.39 s (v95)**, 14.95 s below v89. Live scores from the
official result files; hold-out prices on the hold-out scale.

| id | lever | upload | live, s | state |
|---|---|---|---|---|
| A1 | OPDI pushback clock of the scored flight | v90 | 263.17 (-1.17) | served |
| A2 | ADSB.lol pushback clocks: 24 ranking days | v91 | 259.39 (-3.78) | served |
| A2 | all 62 ranking days, plus A5 | v92 | 253.82 (-5.57) | served |
| A2 | stack trained on 21 days of 2025 | v93 | 251.44 (-2.38) | served |
| A2 | stack trained on all 62 January and July 2025 days | v95 | 249.39 (-1.12) | served, best |
| A2 | specialist on 291 days outside months 1 and 7 (4.5 step 1) | v96 | 249.92 (+0.53) | rejected: hold-out -302 MSE did not transfer |
| A3 | floor at the first moving ADS-B point | | | closed: hold-out +4,875 MSE or worse |
| A4 | neighbour pushback clocks | | | closed: +2 (7 days), within 30 (21 days), +37 MSE (62 days) |
| A5 | stand bound `min(adsb_own_last, adsb_push)` after MS1 | v92 | | served; offsets of +60 to +180 s cost +29 to +523 MSE |
| A6 | pushback clock in the LIRF term | | | blocked: 0 % ADS-B ground coverage at LIRF in 2026 |
| B1 | Step A analogue outside LIRF | | | closed: 0 of 412 null-record rows with `sd > 14,400` sit on `sd` |
| B2 | 3 seeds for the per-airport members | | | closed: -5 MSE |
| C1 | reproduction path | | | served: `test_parity.py` (v67, v92, v94, v95), SHA-256 manifests, `fetch_metar.py`, `REPRODUCE.md` for v95; a Tier-1 run on a clean clone is still to do |
| - | MLP member at 0.15 | v94 | 250.51 (-0.93) | served |
| - | full LightGBM retrain with ADS-B columns (4.5, last) | | | waits for the 2025 downloads (February to December in progress) |

Lessons:

1. Row-level observables transferred at 1 to 2 times their hold-out
   price (v92 to v95). Derived columns did not (v96, as v66, v71, v83).
2. More training days for the stack was the largest lever after the
   ranking coverage itself: 7 to 21 days -2.38 s, 21 to 62 days -1.12 s.
3. The ADS-B ground coverage, not the linking, limits EDDF, LEMD, LFPG,
   LIRF and LTFM: 97 % of callsigns appear, mostly airborne.
