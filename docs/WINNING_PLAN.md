# Plan to the deadline, second edition: from 264.34 s toward 220 s

Status: 2026-09-29. Base commit: `2a4ec1b` (`origin/main`, v89).
This edition replaces the edition of 2026-09-23. The lever ids L1 to L15
in `README.md`, `RECAP.md` and `docs/MODEL_ANALYSIS.md` refer to that
edition: `git show 2a4ec1b:docs/WINNING_PLAN.md`.
Scope: analysis and plan only. No number in this file comes from a new
model run. Every number carries a source or the word "arithmetic".
Grades as in `docs/MODEL_ANALYSIS.md` 0.2: A measured live, B measured on
a partial frame, C arithmetic or scenario, D reasoning only.

## 0. Conclusions

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

## 1. What changed since the edition of 2026-09-23

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

## 2. Error budget and the arithmetic of 220 s

### 2.1 Live budget, estimated

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

### 2.2 Gain from an ADS-B clock on clean rows

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

### 2.3 Why the ADS-B clock is different from the NM clock

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

## 3. Ranked levers

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

### 3.1 Closed levers: do not repeat

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

## 4. ADS-B build

### 4.1 Probe, day 1: go or no-go

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

### 4.2 Match a departure to a track

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

### 4.3 Detect the pushback

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

### 4.4 Columns, keyed on `MVT_ID_mvt`

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

### 4.5 Integration, smallest diff first

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

### 4.6 Price on the hold-out

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

### 4.7 Label-free gate for every ADS-B upload

Run `src/check_submission_2026.py` against v89, plus these checks:

1. Q1 share per airport and month in 2026 against the probe.
2. Quantiles 10, 50 and 90 of `adsb_taxi` per airport, 2026 against 2025.
3. Every row that moves by more than 3,000 s is Q1 with
   `adsb_gap_max <= 180` s and `abs(adsb_dt_to) <= 60` s.
4. Non-LIRF rows above 7,200 s rise only on rows with such Q1 support.
5. LIRF rows move only if A6 runs.

### 4.8 Data volume and machine

- The ADSB.lol day size is unmeasured here. Measure it in P2 before the
  full download.
- Order of downloads: January and July 2026 first (62 days, needed to
  serve), then January and July 2025 (62 days, needed to fit and price).
- Downloads run next to model fits. Trace parsing runs at night: it
  competes for the CPU.
- Start with `gzip` and `json` from the standard library. Add `orjson`
  only if one day takes more than 10 min to parse. State the reason in
  `requirements.txt` (`AGENTS.md` 2).

### 4.9 Policy, licence and stance

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

## 5. Plan to 11 October

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

## 6. Stop rules

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

## 7. Risks

| risk | effect | control |
|---|---|---|
| ADS-B ground coverage is low at LFPG, LIRF, LEMD, LTFM (organiser, `RECAP.md:422-424`) | `c` stays low; 220 s out of reach | probe first; target set from the measured `c` |
| coverage differs between 2025 and 2026 (OPDI EDDF: 7 records in January 2025, 5,582 in January 2026) | a model that reads presence as signal shifts in 2026 | specialist on Q1 rows only; `adsb_day_loaded`; per-airport gate 4.7 |
| wrong track matched | a 3 h error costs 338 MSE per row; 30 such rows cost about 10,000 MSE | type and stand check; `abs(adsb_dt_to) <= 120` s; one-to-one assignment; gate 4.7.3 |
| label off-block defined differently per airport | bias per airport and stand | the specialist learns the bias; report it in P1 |
| fallback rows: the label is `sd`, not the pushback | a clock can overrule a correct hedge | serve weight `w` below 1 if the price says so; LIRF keeps its gate |
| download or disk limit | A2 does not finish | 2026 days first; A1 does not depend on A2 |
| the top keeps falling | 220 s is not first place on 11 Oct | the prize needs top 3 and a reproduction path, not first place |

## 8. Decisions for the team

1. Adopt A1 and A2. The organiser allows them; post the question of 4.9
   anyway.
2. Accept ODbL for derived ADS-B tables: attribution in `README.md`, no
   derived table in git.
3. Set the season target after the probe with the table in 2.2. Write it
   in `README.md`.
4. Reserve Wed 7 Oct to Sat 10 Oct for C1. Without C1, a top-3 rank does
   not win a prize.

## 9. Repo corrections to make with the next commit

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
