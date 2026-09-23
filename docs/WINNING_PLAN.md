# Plan to the deadline: gap analysis and ranked levers

Status: 2026-09-23, 07:30 UTC. Base commit: `bbd0d62` (`origin/main`).
Scope: analysis only. This file changes no code and uploads nothing. No
number in this file comes from a new model run.

## 0. Conclusions

1. kind-mango is 57th of 178 teams at 287.334 s (v57). Third place is
   238.783 s (jolly-lobster). The gap is 25,543 MSE. At 287 s, 1 s of RMSE
   is 575 MSE.
2. The challenge closes on 11 October 2026 at 23:59:59 CET. `README.md`
   and `RECAP.md` state 31 October. That date is wrong. 19 UTC days and at
   most 95 uploads are left.
3. The official API holds live scores for v44 (299.846), v45 (296.125) and
   v50 (293.816). The repo lists all three as "pending". With these scores,
   the L1 retune (v45 to v46) cost +3.01 s (+1,793 MSE). v57 still trains
   with the retuned parameters.
4. Five teams between rank 23 and rank 38 publish GPLv3 code that reads
   `AOBT_3_flt`. They score 268.5 to 278.0 s, 9 to 19 s ahead of us. On the
   2025 hold-out, two of them report 316 and 372 s against 320 s for our v41
   stack.
5. Inside the current stance (no `AOBT_3_flt`, no `LOBT_flt`), the ranked
   levers give a planning range of 279 to 284 s. Top 3 is not reachable on
   this path.
6. With the NM clocks of the scored flight, the planning range is 254 to
   271 s (grade C/D). Third place at the deadline is likely 230 to 235 s.
   No public evidence covers the last 20 to 40 s.
7. Four decisions belong to the team only. Section 11.1 lists them.

## 1. Sources and limits

| source | status | how read |
|---|---|---|
| repo `origin/main` `bbd0d62` | read | docs, `src_v3/`, `src/predict_v30.py`, `models/*.json`, `submission/*.json` |
| official API `datacomp.opensky-network.org/api/competitions/<id>/leaderboard` | read at 07:26 UTC | 2,412 scored submissions, 178 teams, 49 pages |
| mirror `prc-leaderboard.fly.dev/data.json` | read at 07:22 UTC | best per team, agrees with the API |
| challenge site `prc-data-challenge-2026.netlify.app` | read live | index, data, ranking, rationale, eligibility, teams |
| site source `github.com/euctrl-pru/prc_data_challenge_website_2026` | cloned at `54c67d06` | page history with dates |
| OpenSky Discord `#prc-data-competition` | **not reachable** | Claude in Chrome was not connected. I did not sign in anywhere. Discord content below comes from `RECAP.md` and is not re-verified. |
| competitor repos | 14 cloned | 9 hold code; section 6 |

Limits:

- The API history starts at 2026-09-04 09:58 UTC. Uploads before that time
  are absent. Some teams start at v11 or v541 in the API.
- The API exposes two endpoints only: leaderboard and teams
  (`/api/openapi`). It holds no per-airport or per-row data.
- This machine holds no training data, no ranking data and no booster.

## 2. The deployed v57 stack

`src_v3/predict_v57.py` calls `src/predict_v30.py:main`, then applies MS1
and MS2.

| component | rows served in 2026 | files | training rows and months | features |
|---|---|---|---|---|
| base | 317,942 non-LIRF rows | `lgbm_r_all_v57_s{42,43,44}`, 767 / 2,265 / 1,278 rounds | 1,923,953 rows: not LIRF, `0 < y <= 80,000`, all 12 months (`src_v3/train_v57_base.py:60-64`) | 117 columns (`models/lgbm_r_all_v51.meta.json`); retuned parameters: 436 leaves, `min_data_in_leaf` 291, learning rate 0.0183, `feature_fraction` 0.563, `linear_lambda` 0.0056; encoders copied from v26 |
| `plan_taxi_res` | every row | `plan_taxi_res_{train,rank}_v57.parquet` | route medians on 12 months, 5,883 routes | `MVT - ARVT_1` minus the route median, clip +/-3,600 s |
| LIRF regressor `R_norm` | genuine term of 26,899 LIRF rows | `lgbm_r_norm_lirf_v55_s{42..46}`, 266 / 422 / 489 / 326 / 268 rounds | 128,053 LIRF rows with `abs(y - sd) >= 60` and `y < 80,000`, 12 months | 166 columns, 38 `ec_*` columns included (finding D1) |
| LIRF gate `p_fb` | every LIRF row | `lgbm_p_fb_lirf_v56` (207 rounds) and `lirf_regime_v56.isotonic.pkl` | 160,704 LIRF rows, 12 months | 167 columns, 14 fallback-rate columns from the v23 maps |
| mixture | LIRF rows | `src/predict_v30.py:198-203` | | `p_fb * sd + (1 - p_fb) * R_norm` |
| Step A | LIRF, null `FLIGHT_ID_mvt`, `sd > 14,400`: 43 rows | `lirf_band_table_v30.json`, 11 bands | 12 months | its normal term reads the mixture (finding T1) |
| ITY340 | LIRF, `sd > 70,000`, outside Step A: 1 row | `src/predict_v30.py:209-215` | | `(5/6)(86,400 + 1,150) + (1/6) sd` = 88,718 s on `sd` = 94,560 |
| post-processing | every row | `src_v3/postprocess.py`, `predict_v57.py:69-70` | 2025 support tables | MS1 caps and MS2 floors; no MS3 |

Label-free state of v57 (`submission/kind-mango_v57.check.json`): 125 rows
above 7,200 s, 3 rows above 80,000 s, 0 rows at 0 s.

Two defects remain in the served path:

- The v56 gate docstring states a leave-one-month-out isotonic fit
  (`src_v3/train_p_fb_v56.py:9-10`). The code fits the isotonic map on the
  in-sample scores of the full-year booster (`:86-91`). In-sample AUC is
  0.926 (`models/lirf_regime_v56.meta.json`). The last honest stop AUC was
  0.858 (`RECAP.md:237`). The calibrated probabilities are too extreme.
- Step A reads the mixture as its normal term (MODEL_ANALYSIS T1). This is
  unchanged since v30.

## 3. Corrected ledger, v41 to v60

Scores come from the official API. MSE deltas use the 344,841-row scale.

| upload | live, s | parent | change | delta, s | delta, MSE | repo record |
|---|---|---|---|---|---|---|
| v41 | 299.314 | | | | | correct |
| v44 | 299.846 | v41 | L3.a EOBT quartiles | +0.531 | +318 | "pending" |
| v45 | **296.125** | v44 | L9 `plan_taxi_res` | **-3.721** | **-2,217** | "pending" |
| v46 | 299.138 | v45 | L1 retune only | **+3.012** | **+1,793** | "-0.17 s vs v41" |
| v47 | 296.570 | v46 | L2 12-month refit, retuned parameters | -2.568 | -1,530 | correct; +263 MSE vs v45 |
| v48 | 294.239 | v47 | MS1 and MS2 | -2.331 | -1,377 | correct |
| v49 | 294.654 | v48 | LIRF head without 56 drifted columns | +0.416 | +245 | correct |
| v50 | **293.816** | v48 | MS3 member median | -0.422 | -248 | "pending" |
| v51 | 288.248 | v50 | MB3 base on served rows (built on v48, no MS3) | -5.568 | -3,241 | "-5.99 vs v48" |
| v52 | 288.359 | v51 | MB4 calendar | +0.111 | +64 | correct |
| v54 | 324.105 | v51 | MB1 anchored target | +35.857 | +21,957 | correct |
| v55 | 287.827 | v51 | MB8 `R_norm` | -0.421 | -242 | correct |
| v56 | 287.499 | v55 | MB8 gate | -0.328 | -189 | correct |
| v57 | 287.334 | v56 | MB8 route medians, base retrained | -0.165 | -95 | correct |
| v58 | 288.625 | v57 | MB8 operator encoders | +1.291 | +743 | correct |
| v59 | 287.700 | v57 | MF4 arrival drift | +0.366 | +211 | correct |
| v60 | 289.579 | v57 | MD4 leave-one-month-out encoders | +2.245 | +1,295 | correct |

v41 to v57 in total: -11.98 s, -7,028 MSE.

### 3.1 Readings in the repo that the ledger does not support

1. **The L1 retune lost 3 s.** v45 and v46 differ only in the base
   parameters and round counts: same features, months, stop mask and LIRF
   head (`src/train_r_all_v46.py:1-5, 74-91`; `src/predict_v46.py`). The
   paired price was -662 MSE (`models/lgbm_r_all_v46.holdout.json`). Live
   was +1,793 MSE. v47 to v57 still use the retuned parameters.
2. **The 12-month refit is not shown to be the main driver.** v47 is
   +263 MSE worse than v45. No upload tests the old parameters on 12 months.
   The v46 to v47 gain can partly be a repair of the retune damage.
3. **The harness failed on single levers, not only on stacks.** L3.a:
   paired -565, live +318. L9: paired -650, live -2,217. L1: paired -662,
   live +1,793. MODEL_ANALYSIS 2.3 and section 9 say that stacked prices do
   not transfer. The ledger shows that single prices did not transfer
   either. MODEL_ANALYSIS P6 is now resolved: the failing levers were L1 and
   L3.a.
4. **MS3 left the stack.** v50 (MS3) was the best file when v51 shipped.
   v51 was built on v48. From v51 to v57, `predict_v5x.py` applies MS1 and
   MS2 only.
5. **"MB8 held at five levels" overstates v57.** v57 retrained the base.
   Every non-LIRF row moved, with a mean absolute move of 7.1 to 9.7 s per
   airport (`kind-mango_v57.check.json`). A near-null base retrain (v52)
   landed +64 MSE. v57 landed -95 MSE. This is inside the retrain noise,
   about 0.15 s. v55 and v56 moved LIRF rows only; those two results carry
   more weight.
6. **The quoted warning is the team's own text.** `README.md:38-40` cites
   "the brief". The brief is `docs/PRC_Data_Challenge_2026_BRIEF.md:178`, a
   team file. No organiser page holds that sentence.
7. **v39 read `LOBT_flt`.** `README.md:44-45` says that no submission reads
   `AOBT_3_flt` or `LOBT_flt`. v39 served `mvt_lobt = MVT - LOBT_flt`
   (`src_v2/frame.py:76`, `src_v2/features.py:10`). v39 scored 352.19 s and
   is not the best file. The sentence is false for v39.
8. **`AOBT_3_flt` does not reconstruct the label.** On 2025 departures,
   `BLOCK - AOBT_3` is exact on 0.65 % of rows and within 60 s on 21.0 %.
   The median absolute difference is 175 s (section 8.1).
9. **Coverage figure.** `AOBT_3_flt` is non-null on 98.9 % of all ranking
   rows (DEP and ARR). On the 344,841 scored DEP rows it is 98.47 %.
10. **Deadline.** The site has stated 11 October since 2026-08-13 (site
    commit `24ff5a8a`). `README.md` and `RECAP.md:414` state 31 October.
11. **Rank.** `RECAP.md:129` gives rank 44 of 111. The rank is 57 of 178.

## 4. Error budget and the gap

### 4.1 Cost of one row

| absolute error on one row | MSE on 344,841 rows |
|---|---|
| 1,000 s | 2.9 |
| 5,000 s | 72 |
| 10,000 s | 290 |
| 30,000 s | 2,610 |
| 86,400 s (one 24-h row) | 21,648 |

One 24-h row is 85 % of the gap to third place. A clean-class gain from
250 s to 220 s on 300,000 rows is `300,000 * (250^2 - 220^2) / 344,841` =
12,267 MSE.

### 4.2 What the 2025 hold-out says

The last end-to-end hold-out is the v41 stack (`models/v41.holdout.json`):
clean 60,365, fallback 7,688, tail 12,681, 24-h 21,517 MSE (hold-out
scale). One LFPG row holds 20,245 of the 24-h class. The same stack scored
89,589 MSE live.

In 2025, all 15 rows with `y > 86,400` and 33 of 34 rows with
`y > 43,200` have no NM record (likable-eagle repo,
`analysis/output/05_extremes.txt:60-88`). 516 of the 584 rows with
`y > 7,200` carry `AOBT_3_flt`. NM clocks cannot repair 24-h rows. They can
reduce the clean class and the matched tail.

### 4.3 What 2026 says without labels

- The arrival mirror (`models/v3/arrival_mirror.json`) scores a 2025-trained
  taxi-in model on 2026 arrivals. EHAM: 405.5 s in January, 129.4 s in
  July. LFPG: 250.7 s and 189.1 s. A January 2026 regime at EHAM is not in
  the 2025 training months. Its cause is unmeasured.
- A model that reads the NM clocks of the scored flight follows such a
  regime row by row. A model on plans and counts does not. This is one
  mechanism for the live advantage of `AOBT_3_flt` users (grade D).

### 4.4 Where 25,543 MSE can come from

| source | evidence | MSE | grade |
|---|---|---|---|
| NM clocks of the scored flight | five public users at 268.5 to 278.0 s: 5,269 to 10,457 MSE ahead of v57 | 5,300 to 10,500 | C |
| neighbour NM taxi proxies | zestful-fountain v8 to v9: -0.88 s, -491 MSE, on top of own clocks. Our OPDI-live family measured the same quantity until v26 dropped it for 0 % 2026 coverage. | 500 to 5,000 | D |
| stronger learner, second model class | zestful-fountain v7 to v8: deeper CatBoost residual, -7.11 s, -4,015 MSE on their stack | 300 to 4,000 | C/D |
| revert of the L1 retune | v45 to v46: +1,793 MSE | 0 to 1,800 | B |
| LIRF and matched tail with NM clocks | 516 of 584 2025 rows above 7,200 s carry `AOBT_3_flt` | 300 to 2,000 | D |
| extreme rows without an observable | 21,648 MSE per 24-h row; all 2025 cases lack an NM record | not modelable | none |
| sum | | 6,400 to 23,300 | |

The sum gives 276.0 to 243.4 s. The ranges overlap: the public users'
scores already hold part of rows 2, 3 and 5. The top of the range misses
third place today (238.8 s). It also misses the likely deadline bar of 230
to 235 s (27,336 to 29,661 MSE from v57).

## 5. Leaderboard analysis

### 5.1 Thresholds by day (API, best per team at the end of each UTC day)

| day | teams | 1st | 3rd | 10th | 20th | 50th | kind-mango | rank |
|---|---|---|---|---|---|---|---|---|
| 09-06 | 58 | 265.76 | 273.21 | 294.94 | 318.60 | 562.44 | 561.06 | 48 |
| 09-09 | 92 | 246.36 | 264.90 | 278.21 | 289.67 | 330.73 | 317.50 | 42 |
| 09-13 | 132 | 245.02 | 261.92 | 271.04 | 278.90 | 296.86 | 299.31 | 52 |
| 09-16 | 152 | 241.75 | 245.02 | 263.38 | 271.78 | 292.00 | 296.13 | 61 |
| 09-19 | 165 | 239.58 | 242.70 | 258.60 | 268.52 | 286.31 | 287.33 | 53 |
| 09-21 | 171 | 239.13 | 239.55 | 255.33 | 267.87 | 285.68 | 287.33 | 55 |
| 09-23 | 178 | 235.31 | 238.78 | 254.52 | 266.66 | 284.77 | 287.33 | 57 |

Third place fell 6.2 s in the last 7 days and 3.9 s in the last 4 days.
At 0.3 to 0.6 s per day, third place reaches 230 to 235 s on 11 October.
The 50th place moves about 0.5 s per day. Without a change, kind-mango
falls to rank 60 to 70.

### 5.2 Top 20 trajectories

"Largest drop" is the largest fall of the running best while the previous
best was under 345 s.

| # | team | best | uploads | first in API | first score | first day under 280 | largest drop |
|---|---|---|---|---|---|---|---|
| 1 | vigorous-whistle | 235.307 | 23 | 09-13 | 436.9 | 09-14 | -45.5 s on 09-14 (322.9 to 277.5) |
| 2 | gentle-tractor | 238.651 | 24 | 09-15 | 306.9 | 09-17 | -11.5 s on 09-20 (258.6 to 247.1) |
| 3 | jolly-lobster | 238.783 | 47 | 09-04 | 423.1 | 09-15 | -29.9 s on 09-10 (330.2 to 300.4) |
| 4 | jovial-uniform | 239.945 | 18 | 09-05 | 345.8 | 09-08 | -41.3 s on 09-08 (319.1 to 277.9) |
| 5 | zesty-puzzle | 241.954 | 49 | 09-04 | 546.5 | 09-07 | -13.4 s on 09-20 (263.2 to 249.8) |
| 6 | gentle-igloo | 242.656 | 37 | 09-13 | 501.2 | 09-14 | -59.6 s on 09-14 (339.2 to 279.6) |
| 7 | enthusiastic-daisy | 243.291 | 81 | 09-04 | 314.4 | 09-04 | -34.6 s on 09-04 (314.4 to 279.8) |
| 8 | youthful-giraffe | 245.021 | 21 | 09-06 | 265.8 | 09-06 | -14.3 s on 09-08 (262.7 to 248.5) |
| 9 | bubbly-telephone | 253.337 | 53 | 09-11 | 528.6 | 09-15 | -31.3 s on 09-13 (323.7 to 292.4) |
| 10 | quick-boat | 254.523 | 61 | 09-04 | 283.2 | 09-04 | -6.1 s on 09-07 (273.2 to 267.1) |
| 11 | gentle-lemon | 254.737 | 38 | 09-07 | 349.0 | 09-10 | -3.7 s on 09-16 (262.7 to 259.0) |
| 12 | merry-quicksand | 256.970 | 26 | 09-08 | 689.7 | 09-11 | -14.1 s on 09-11 (281.5 to 267.5) |
| 13 | gentle-ladder | 259.226 | 41 | 09-09 | 518.3 | 09-12 | -19.2 s on 09-10 (312.7 to 293.4) |
| 14 | strong-crane | 260.121 | 38 | 09-04 | 547.3 | 09-12 | -7.0 s on 09-12 (284.8 to 277.7) |
| 15 | strong-turtle | 260.561 | 32 | 09-11 | 360.4 | 09-13 | -11.6 s on 09-13 (290.0 to 278.4) |
| 16 | humble-giraffe | 265.057 | 21 | 09-11 | 538.9 | 09-17 | -16.4 s on 09-16 (296.9 to 280.4) |
| 17 | upbeat-goblin | 265.284 | 32 | 09-04 | 281.4 | 09-04 | -3.6 s on 09-11 (275.9 to 272.3) |
| 18 | organized-marshmallow | 266.111 | 3 | 09-13 | 285.7 | 09-15 | -10.4 s on 09-19 (276.5 to 266.1) |
| 19 | upstanding-firefly | 266.405 | 9 | 09-06 | 411.6 | 09-07 | -56.3 s on 09-06 (337.0 to 280.7) |
| 20 | dependable-eagle | 266.662 | 45 | 09-05 | 407.3 | 09-17 | -14.3 s on 09-17 (284.5 to 270.2) |
| 57 | kind-mango | 287.334 | 55 | 09-04 | 562.4 | none | -13.1 s on 09-10 (316.8 to 303.7) |

### 5.3 Single-upload jumps of 30 s or more (previous best under 345 s)

| team (rank) | day | from, to | MSE |
|---|---|---|---|
| gentle-igloo (6) | 2026-09-14 | 339.2 to 279.6 | -36,875 |
| upstanding-firefly (19) | 2026-09-06 | 337.0 to 280.7 | -34,754 |
| vigorous-whistle (1) | 2026-09-14 | 322.9 to 277.5 | -27,303 |
| jovial-uniform (4) | 2026-09-08 | 319.1 to 277.9 | -24,653 |
| versatile-watermelon (25) | 2026-09-15 | 313.8 to 276.8 | -21,836 |
| enthusiastic-daisy (7) | 2026-09-04 | 314.4 to 279.8 | -20,561 |
| dynamic-lobster (21) | 2026-09-16 | 340.3 to 310.6 | -19,376 |
| bubbly-telephone (9) | 2026-09-13 | 323.7 to 292.4 | -19,299 |
| jolly-lobster (3) | 2026-09-10 | 330.2 to 300.4 | -18,840 |

### 5.4 Reading

- Seven teams land between 276.8 and 280.7 s after one jump from 314 to
  340 s. A jump of that size does not prove a new data source by itself.
  Our own v16 (Step A, a rule on a few LIRF rows) gained 46,603 MSE. One
  24-h row is worth 21,648 MSE.
- The stronger signal is the first-day level. quick-boat scored 283.2 and
  upbeat-goblin 281.4 on 09-04. youthful-giraffe scored 265.8 on its first
  API upload. elegant-alligator scored 285.1 on its first upload with
  `MVT - AOBT_3` as a feature. kind-mango needed 15 days to reach 287.3.
- Since 09-16, the top 3 gain 0.3 to 1 s per day with small steps. This
  pattern fits model and feature work on top of a strong base signal.

## 6. Competitor code

"Clocks" means NM timestamp deltas of the scored flight. "Neighbour"
means the same quantity of other flights.

| repo | team (rank, best) | target and model | `AOBT_3_flt` | `LOBT_flt` | `ARVT_3_flt` | licence |
|---|---|---|---|---|---|---|
| `javidmardanov/PRC-Data-Challenge-2026` | elegant-alligator (24, 268.558) | TimesFM-3 airport-hour forecasts, CatBoost and LightGBM, mixture of experts split on `AOBT_3` null, LIRF identity correction | clock `MVT - AOBT_3` and all `dt_*` deltas (`scripts/proxy_baseline.py`) | residual expert on `MVT - LOBT` (`scripts/autoresearch_known_lobt.py`) | listed, not a feature | GPLv3; TimesFM weights restricted |
| `Phoenix-Ops-LTD/prc2026-taxiout` | zestful-fountain (30, 273.556) | CatBoost residual on the baseline `MVT - AOBT_3` (fallback `LOBT`, then plans), LightGBM blend, LIRF specialist on null `IOBT` | own clock; neighbour proxies over past windows (v9) and future windows (v12) | baseline fallback; v11 overlay `max(0, MVT - LOBT)` on 21 rows | not used | GPLv3 |
| `ChandanHegde07/OpenAir` | likable-eagle (36, 276.986) | per-airport OLS on clocks, then LightGBM residual; LIRF unmatched rows served `MVT - SCHED` | primary clock, `AOBT - EOBT`, neighbour state over 30 min | clock spread deferred | not used | GPLv3 |
| `col3name/air-data` | kind-earthquake (23, 268.522) | CatBoost with traffic, runway and historical-target features | `AOBT_3 - EOBT_1` | listed | listed | no licence file; one committed configuration file not read |
| `satam2/knowledgeable-helicopter` | team name from repo name (38, 278.015) | CatBoost residual, long-proxy specialist | `takeoff - AOBT_3` and flags | `LOBT - IOBT` | not used | GPLv3 |
| `sergiuv11/prc-data-challenge-2026` | jubilant-vase (84, 299.286) | LightGBM, LIRF overlay, three airport specialists | `implied_taxi_aobt` and deltas | deltas | not used | GPLv3 |
| `ahmetabdullahgultekin/prc-taxiout-2026` | vibrant-lollipop (91, 310.550) | P10 reference plus residual | optional flag | excluded | excluded | GPLv3 |
| `fenil264/prc-2026-taxi-out` | joyful-lemon (83, 298.860) | early notebooks | not used | not used | not used | no licence file |
| `victoralcadi/prc-data-challenge-2026` | eager-jungle (175, 689.690) | early | excluded as leaky | not found | not found | GPLv3 |

Empty or preparation only: `RavindraTarunokusumo`, `ruchir321`,
`georgejhanlon`, `Enzoferrariz`, `Necklacesaura`. No top-20 team has a
public repo that a name search or a code search finds.

jubilant-vase: the README claims an official 286.656 s "on all 215,876
ranking pairs". The API has no such score. All 2,412 API scores use
344,841 pairs. The best jubilant-vase score in the API is 299.286 s. The
claim refers to a file with 62.6 % of the template rows and is not
comparable.

Useful measurements from these repos (2025 departures):

- `BLOCK - AOBT_3`: mean absolute 238 s, median 175 s, exact 0.65 %,
  within 60 s 21.0 %, within 300 s 74.6 %. `BLOCK - LOBT`: within 60 s
  11.3 %, within 300 s 44.1 %, bounded at +/-3,606 s (OpenAir,
  `analysis/output/01_structure_leakage.txt:508-523`).
- The naive predictor `MVT - AOBT_3` has RMSE 384.9 s, from 255 s at EDDF
  to 557 s at LIRF (vibrant-lollipop, `docs/facts.md:145-146`).
- `AOBT_3_flt` has minute resolution. `BLOCK_TIME_UTC_mvt` has second
  resolution.
- zestful-fountain holds 2025 January+July hold-out RMSE 315.8 s (v8) and
  scores 278.6 s live. likable-eagle holds 372.4 s and scores 277.0 s.
  Our v41 stack holds 319.9 s and scored 299.3 s.

### 6.1 Generic ideas and copying

Generic, free to reimplement with a citation in `README.md`:

- a residual target on a clock anchor, with a bounded offset;
- CatBoost with ordered target statistics on high-cardinality categories;
- neighbour statistics of a taxi proxy over time windows;
- features from arrivals completed before the scored take-off;
- `MVT - SCHED` for unmatched LIRF rows (we hold this rule since v16).

Copying, not allowed without rights and "significant modifications"
(eligibility page): their scripts (for example
`nm_following_features.py`, `nm_clock_overlay.py`, the TimesFM pipeline),
their fixed thresholds, tree counts, depths and blend weights. GPLv3
permits reuse with attribution. The prize rule on originality is stricter.
Reimplement the idea in our pipeline; do not import their files.

## 7. Organiser statements

### 7.1 Site pages (read live on 2026-09-23; dates from the site git history)

| page | statement | since |
|---|---|---|
| index | Timeline: 1 September to 11 October 2026, 23:59:59 CET | 2026-08-13 |
| index | The organisers can change the rules or stop the challenge | 2026-08 |
| data | Ranking file: only `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` of DEP rows are blank. `LOBT_flt`, `AOBT_3_flt` and `ARVT_3_flt` are listed flight columns with no restriction. | 2026-08-11 |
| ranking | The organisers watch for attempts "to learn from or exploit the ranking process" | 2026-08-27 |
| ranking | 5 uploads per day (3 before 2026-09-07), 1 GB per bucket | 2026-09-07 |
| eligibility | Open datasets, GPLv3 source forked by the challenge account, documentation to reproduce, original solution | 2026-07-24 |

There is no FAQ or news page. No site text names `AOBT_3_flt`, `LOBT_flt`,
`ARVT_3_flt`, forward windows, a second phase or a hidden test set.

### 7.2 Discord (from `RECAP.md:351-410`; not re-verified)

| date | ruling by the organiser (espinielli) |
|---|---|
| 2026-09-08 | `_flt` values come from NM. "Any provided value is fair to use as a proxy (or as itself) if it helps." |
| 2026-09-09 | Fallback rows are in the ranking set. Open data is allowed if documented. Features from earlier movement rows are allowed: the model can use "what is occupied, what is coming". |
| 2026-09-11 | No phase 2 is planned. The organiser keeps the right to add one if reverse engineering of the ranking is too easy, a practice "we despise". Trino access is not allowed. |
| 2026-09-17/18 | AIU daily files and ADSB.lol count as open data with attribution. Reply to a question on `AOBT_3_flt` and `MVT_TIME_UTC_mvt`: the model is for post-ops, not tactical use; for off-block times from open trajectory data, "there are no such restrictions". |

Not found: any ruling after 2026-09-18, any ruling on `ARVT_3_flt`, a
hidden final test set, and an explicit eligibility audit procedure. The
final ranking is the public leaderboard unless the organiser adds a phase.

## 8. `AOBT_3_flt`, `LOBT_flt` and `ARVT_3_flt`

### 8.1 Facts

- `AOBT_3_flt`, `LOBT_flt`, `ARVT_3_flt`, `EOBT_1_flt`, `IOBT_flt` and
  `ARVT_1_flt` share one null mask. They are non-null on 339,551 of the
  344,841 ranking DEP rows (98.47 %).
- `AOBT_3_flt` is NM's actual off-block time of the flown (M3) trajectory.
  It is a strong, noisy clock: 21.0 % of 2025 rows lie within 60 s of the
  label's off-block, and the median error is 175 s.
- `LOBT_flt` is the last known off-block time. It is a plan value on most
  rows. It equals `AOBT_3_flt` on 4.3 % of rows (`README.md:41-42`).
- `ARVT_3_flt` is the arrival time of the flown trajectory. It is later
  than the take-off. Alone, it holds no off-block information.
  `ARVT_3 - AOBT_3` is NM's flown block time. It adds nothing beyond
  `MVT - AOBT_3` for the off-block. Grade D: this rests on the column
  definitions, not on a measurement. No repo in section 6 uses it as a
  feature.

### 8.2 Two readings of the rules

| | permissive reading | restrictive reading |
|---|---|---|
| basis | Data page blanks two columns only. 2026-09-08: any provided value is fair to use. 2026-09-18: post-ops model, "no such restrictions". The ranking warning dates from 2026-08-27 and concerns the ranking process, not dataset columns. | The organiser despises reverse engineering of the ranking (2026-09-11). The 2026-09-18 answer on trajectories rests on a practical limit, not on a permission. A field close to the label defeats the purpose of a taxi-out model. |
| risk | A later ruling or a phase 2 without NM actuals voids the gain. | The team gives up 5,300 to 10,500 MSE that five public teams take openly. |
| what competitors do | All five public repos of teams above rank 40 read `AOBT_3_flt`. Four of them publish under GPLv3. | One public repo (rank 175) excludes it as leaky. |

A middle reading exists: use other flights' `MVT - AOBT_3` only, over
windows that end before the scored take-off. This reveals no label of the
scored row. It matches the 2026-09-09 permission for features from earlier
movement rows. It still reads `AOBT_3_flt`, which the README excludes.

This file does not decide. Section 11.1 lists the decision. Whatever the
choice, ask the organiser one written yes/no question today and record the
answer in `README.md`.

## 9. Ranked levers

Tracks: **A** is inside the current stance. **M** reads other flights'
`AOBT_3_flt` only. **B** reads the NM clocks of the scored flight.
Hours are collaborator hours on the data machine. MSE is on the
344,841-row scale; negative is a gain.

| rank | id | lever | track | MSE | grade | hours | uploads |
|---|---|---|---|---|---|---|---|
| 1 | L1 | NM clocks of the scored flight in the base | B | -5,300 to -10,500 | C | 4 | 2 |
| 2 | L2 | Neighbour NM taxi proxies, backward windows | M | -500 to -5,000 | D | 4 | 1 to 2 |
| 3 | L3 | Second model class (CatBoost) and a fixed blend | A | -300 to -4,000 | C/D | 6 to 8 | 2 to 3 |
| 4 | L4 | Revert the L1 retune on the v57 recipe | A | 0 to -1,800 | B | 1.5 | 1 |
| 5 | L5 | Residual target on the `AOBT_3` anchor, blended | B | -500 to -3,000 | D | 4 | 2 |
| 6 | L6 | NM clocks in the LIRF head | B | -300 to -2,000 | D | 3 | 1 |
| 7 | L7 | Day-level artefact share from ranking arrivals (MH3) | A | 0 to -1,500 | D | 5 | 1 to 2 |
| 8 | L8 | Forward windows of the neighbour proxy | M, policy | -500 to -1,000 | C | 2 | 1 |
| 9 | L9 | Planned-taxi proxy, `ARVT_1 - EOBT_1` residual (MF5) | A | 0 to -900 | D | 2 | 1 |
| 10 | L10 | Out-of-fold isotonic map for the LIRF gate | A | 0 to -800 | D | 1.5 | 1 |
| 11 | L11 | Weather at `EOBT_1` (L7, MF1) | A | -100 to -400 | B | 2 | 1 |
| 12 | L12 | Step A normal term reads `R_norm` (T1) | A | +300 to -800 | C | 1 | 1 |
| 13 | L13 | MS3 member median on the current base | A | 0 to -250 | A | 0.5 | 1 |
| 14 | L14 | Fixed 50/50 blend of two base recipes | A | 0 to -400 | D | 0.3 | 1 |
| 15 | L15 | Final 12-month refit of every component | all | 0 to -500 | D | 3 | 1 |

Planning totals. Stacked levers lost part of their value in section 3. The
plan keeps 35 to 80 % of the summed midpoints.

| track | summed midpoints, MSE | planned gain, MSE | planned live |
|---|---|---|---|
| A | 5,500 | 2,000 to 4,500 | 279.4 to 283.8 s |
| A and M | 9,000 | 3,500 to 8,000 | 273.1 to 281.2 s |
| A, M and B | 20,000 | 9,000 to 18,000 | 254.1 to 271.2 s |

The top-3 bar at the deadline needs 27,300 to 29,700 MSE.

### 9.1 Lever details

**L1. NM clocks of the scored flight (track B).**
- Mechanism: add `mvt_aobt3 = MVT - AOBT_3`, `aobt3_eobt1`, `aobt3_sched`,
  `aobt3_null`, `mvt_lobt`, `lobt_aobt3` to the base. The trees learn the
  per-airport slope and bias of the NM clock (OLS slope 0.59, airport bias
  -236 s at LTFM to +214 s at LIRF, OpenAir `status.md:127-142`).
- Evidence: five public users at 268.5 to 278.0 s; elegant-alligator
  scored 285.1 s on its first upload with this clock.
- Arithmetic: `82,561 - 278.0^2 = 5,269`; `82,561 - 268.5^2 = 10,457`.
- Risk: a later organiser ruling; eligibility review; loss of the paper's
  "no actual off-block" claim.
- Test: live at or below 284.3 s (-3 s). Rows above 7,200 s outside LIRF
  stay at or below 30.

**L2. Neighbour NM taxi proxies (track M).**
- Mechanism: for each departure, mean, median and count of
  `MVT - AOBT_3` of other departures at the same airport, and at the same
  runway, with `MVT` in the 30 and 60 min before the scored take-off. Keep
  proxies in 0 to 7,200 s. Exclude the scored row. This is a live measure
  of the airport's taxi times with 98.5 % coverage in 2026.
- Evidence: our OPDI-live family (other flights' observed taxi,
  `README.md:180-181`) left the base in v26 for 0 % 2026 coverage. zestful-fountain's
  neighbour block gained 491 MSE on top of own clocks.
- Risk: without own clocks the value is unmeasured.
- Test: live at or below best minus 0.5 s.

**L3. Second model class (track A).**
- Mechanism: CatBoost (depth 8, RMSE) on the same 117 columns and the MB3
  rows; fixed blend 0.3 and 0.5 with the LightGBM mean. Ordered target
  statistics replace our in-sample encoders on the categories.
- Evidence: zestful-fountain v7 to v8, -4,015 MSE live with a deeper
  CatBoost residual. Our v46const blend failed, but it used the same
  library with constant leaves (`RECAP.md:88`).
- Cost: new dependency `catboost` (reason: a second learner with a
  different treatment of categories). About 2 h CPU per fit.
- Test: live at or below best minus 0.5 s at the better weight.

**L4. Revert the L1 retune (track A, run first).**
- Mechanism: train the v57 recipe with the v45 parameters: `BEST_PARAMS`,
  220 leaves, `linear_lambda` 1.0 (`src/train_r_all_v45plan.py:49-50`).
  Rounds: v45 best iterations 1,661 / 1,426 / 783
  (`models/lgbm_r_all_v45plan.holdout.json`, key `v45plan.best_iters`)
  times 1.2585 = 2,091 / 1,795 / 986.
- Evidence: v45 to v46, +1,793 MSE live, parameters only.
- Risk: MB3 removed the LIRF and 24-h rows that the tuner followed (P3).
  If that removed the harm, the result is noise (about +/-0.15 s).
- Test: live at or below 287.03 s.

**L5. Residual target on the `AOBT_3` anchor (track B).**
- Mechanism: target `y - clip(MVT - AOBT_3, 0, 7,200)` on non-LIRF rows
  with `AOBT_3`, clipped at +/-7,200 s. Serve anchor plus offset; serve L1
  where `AOBT_3` is null or the proxy is outside 0 to 7,200 s. Blend 50/50
  with L1.
- Evidence: zestful-fountain and likable-eagle use this form. Our v54
  failed with `mvt_eobt1` as anchor, because a plan anchor does not move
  with a gate hold. `AOBT_3` moves with the real pushback on most rows.
- Risk: the v54 failure mode on rows where `AOBT_3` is wrong by hours.
- Test: rows above 7,200 s outside LIRF at or below 30; live below L1.

**L6. NM clocks in the LIRF head (track B).**
- Mechanism: add the L1 columns to `R_norm` and to the gate.
- Evidence: LIRF clean is 13,984 of the v41 hold-out MSE; LIRF matched
  rows carry `AOBT_3`.
- Test: LIRF-only change; live at or below best minus 0.3 s.

**L7. Day-level artefact share (track A).**
- Mechanism: per airport and hour, the share of ranking arrivals with
  in-block within 5 s of the scheduled arrival, and the share of
  departures with a null record or `EOBT_1 = SCHED`. Feed the LIRF gate
  and the base. The ranking file holds arrival block times.
- Evidence: 8.98 % of LIRF arrivals in 2026 sit on the schedule grid
  (MODEL_ANALYSIS T9). Clustering by day in 2025 is unmeasured.
- Gate: first measure the 2025 correlation between the departure and the
  arrival artefact shares per airport-day. Build only if it exceeds 0.3.

**L8. Forward windows (track M or B, policy).**
- Mechanism: L2 over `(t, t + 15 min]` and `(t, t + 60 min]`.
- Evidence: zestful-fountain v11 to v12, -778 MSE live.
- Policy: reads movements after the scored take-off. Two served columns
  already do this (`features_congestion_v2.py`, finding L4).

**L9. Planned-taxi proxy (track A).**
- Mechanism: `plan_nm_taxi = (ARVT_1 - EOBT_1) - median` per
  (`ADEP`, `ADES`, `AIRCRAFT_TYPE`) on 12 months, clip +/-3,600 s. This
  isolates the NM planned taxi time. The served `plan_taxi_res` mixes
  take-off delay with it (finding C2).
- Evidence: `plan_taxi_res` gained 2,217 MSE live (v45), 3.4 times its
  paired price. v38 served the raw form next to two raw columns and was
  unstable (seed 43 stopped at round 6).
- Test: live at or below best minus 0.3 s.

**L10. Out-of-fold isotonic map (track A).**
- Mechanism: fit six gate boosters on the month-pair folds with v56
  parameters and rounds. Fit the isotonic map on the out-of-fold scores.
  Serve the full-year booster with this map.
- Arithmetic: a 0.05 bias in `p_fb` on a row with `sd - R_norm` of 1,500 s
  is 75 s. On 26,899 rows that is 440 MSE.
- Test: out-of-fold calibration error per `sd` bin below that of the v56
  map; live at or below best minus 0.3 s.

**L11. Weather at `EOBT_1` (track A).**
- Evidence: paired -367 MSE on all served classes on the v45 recipe
  (`models/lgbm_r_all_v48wx.holdout.json`). The build script exists
  (`src/build_weather_eobt.py`).

**L12. Step A normal term (track A).**
- Mechanism: pass `R_norm` instead of the mixture as the normal term of
  Step A.
- Arithmetic: MODEL_ANALYSIS T1, 32 MSE per cell row at a gate value of
  0.8; up to 800 MSE on 43 rows. The gate values on these rows are
  unrecorded, so the sign is uncertain.
- Test: isolated upload on v57; the 43 rows are the only rows that move.

**L13. MS3 member median (track A).**
- Evidence: v50, -248 MSE live on v48 (15 rows). On v51, MS3 moved 1 row
  (`models/v3/predict_v53.json`).
- Test: any row moved; live at or below best.

**L14. Fixed blend of two base recipes (track A).**
- Mechanism: 50/50 mean of the L4 base and the v57 base before MS1. The
  two parameter sets give diverse trees.
- Test: upload only if L4 lands within 0.5 s of v57.

**L15. Final refit (all tracks).**
- Mechanism: refit every accepted component on 12 months with the final
  recipe: base, `R_norm`, gate, route medians. Do not refit the operator
  encoders (v58, v60).

## 10. Closed levers: do not repeat

| lever | live result | source |
|---|---|---|
| anchor on `mvt_eobt1` (MB1) | +21,957 MSE (v54) | `RECAP.md:72` |
| 12-month or leave-one-month-out operator encoders | +743 (v58), +1,295 (v60) | `RECAP.md:66-68` |
| arrival drift features (MF4) | +211 (v59) | `RECAP.md:67` |
| LIRF head without drifted columns | +245 (v49) | `RECAP.md:76` |
| calendar block, numeric month out (MB4) | +64 (v52) | `RECAP.md:73` |
| Optuna retune of the base (L1) | +1,793 (v46) | this file, section 3 |
| median imputation of dead columns | +58 s (v25) | `RECAP.md:106` |
| hard regime swaps, old mixture | 625 s (v15) | `RECAP.md:118` |
| removal of the ITY340 hedge, cold-start rewrite | +50 s (v39) | `RECAP.md:92` |
| 7-seed base, 5-seed gate | +0.13 s (v32), +0.65 s (v37) | `RECAP.md:94, 100` |
| honest refits of LIRF statistics | +0.21, +0.22 s (v34, v35) | `RECAP.md:96-97` |
| zero-clip repair | +0.18 s (v36) | `RECAP.md:95` |
| tempo variants L3.b to L3.e, constant-leaf blend L4 | paired, inside noise | `RECAP.md:84-88` |
| EOBT quartiles (L3.a) | +318 (v44) | this file, section 3 |

Not recommended on policy grounds: seed lotteries selected by live score,
per-row probing, and any file that differs from the best only on a handful
of rows. These learn from the ranking process.

## 11. Plan to 11 October

### 11.1 Decisions for the team

1. **Stance on NM clocks.** Choose track A (current), track M (other
   flights' `AOBT_3_flt` only) or track B (own `AOBT_3_flt` and
   `LOBT_flt`). This decides the planning range: 279 to 284 s, 273 to
   281 s, or 254 to 271 s.
2. **Written question to the organiser, today.** Suggested text: "May a
   prize-eligible solution use `AOBT_3_flt` and `LOBT_flt` of the scored
   departure as model inputs?" A team member posts it. Record the reply.
3. **New dependency.** Accept `catboost` for L3, or not.
4. **Time split.** On track A, top 3 is not reachable. Decide how many
   collaborator hours go to the leaderboard and how many to
   reproducibility and the JOAS paper.

### 11.2 Upload plan (UTC days; 5 uploads per day; version numbers are next free)

| day | work | uploads |
|---|---|---|
| Wed 23 Sep | Decisions 1 to 4. R0 audit. R1 (L4). R2 (L13 on v57). R3 (L12 on v57). | v61 (L4), v62 (L13), v63 (L12) |
| Thu 24 Sep | Read v61 to v63; keep the accepted set. R4 (L10). R5 (L9). Track M or B: build R7 columns. | v64 (L10), v65 (L9) |
| Fri 25 Sep | Track B: R8 (L1). Track M: R7 (L2). Track A: R6 (L11) and a bundle of accepted A levers. | 2 |
| Sat 26 Sep | Track B: L1 plus L2; L6. Track A: start R9 (CatBoost fits). | 1 to 2 |
| Sun 27 Sep | R9 blends at 0.3 and 0.5 on the best base. | 2 |
| Mon 28 Sep | Track B: R10 (L5). Track A: L7 measurement in 2025, no upload. | 0 to 1 |
| Tue 29 Sep to Thu 1 Oct | L7 if its gate passes. L8 if decision 1 allows it. L14. | at most 2 per day |
| Fri 2 Oct | Checkpoint: apply the season stop rule (11.4). | 0 |
| Sat 3 Oct to Mon 5 Oct | Stack the accepted levers. L15 refit. | 1 to 2 per day |
| Tue 6 Oct to Thu 8 Oct | Two pre-registered final bundles per day at most. Lock file, final `REPRODUCE.md`, parity test, README and RECAP corrections (section 12). | at most 2 per day |
| Fri 9 Oct to Sat 10 Oct | No new feature family. Rerun the final recipe with fixed seeds for the record. | at most 1 per day |
| Sun 11 Oct | No new model. Last upload before 21:00 UTC: the site says CET, and local time in Europe is CEST until 25 October. | 0 to 1 |

Planned total: 30 to 40 uploads of 95. Keep the rest unused.

### 11.3 Runs on the data machine

Run every command from the repo root. Check each file against v57 before
upload:

```bash
python src/check_submission_2026.py submission/kind-mango_vNN.parquet --ref submission/kind-mango_v57.parquet
```

Label-free gate for every upload: 3 rows above 80,000 s unless the lever
targets LIRF; at most 30 non-LIRF rows above 7,200 s; every row that moves
by more than 3,000 s has a written reason.

**R0. Audit of the NM clock (30 min, no upload).** Measure, per airport and
month, the 2025 share of `abs(BLOCK - AOBT_3) <= 60` and `<= 300`. Measure
the 2026 quantiles (10, 50, 90 %) of `MVT - AOBT_3` against the same 2025
month. A rise of the share through 2025 supports the 2026 advantage of
track B.

```python
import glob, pandas as pd
c = ["ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"]
t = pd.concat(pd.read_parquet(f, columns=c) for f in sorted(glob.glob("training/*.parquet")))
t = t[t.PHASE_mvt == "DEP"]
t["m"] = t.MVT_TIME_UTC_mvt.dt.month
t["d"] = (t.BLOCK_TIME_UTC_mvt - t.AOBT_3_flt).dt.total_seconds().abs()
print((t.d <= 60).groupby([t.ADEP_mvt, t.m]).mean().unstack().round(3))
r = pd.read_parquet("submission/ranking.parquet", columns=c)
r = r[r.PHASE_mvt == "DEP"]
p = (r.MVT_TIME_UTC_mvt - r.AOBT_3_flt).dt.total_seconds()
print(p.groupby([r.ADEP_mvt, r.MVT_TIME_UTC_mvt.dt.month]).quantile([.1, .5, .9]).unstack().round(0))
```

**R1. L4, v61 (1.5 h).**
1. Copy `src_v3/train_v57_base.py` to `src_v3/train_v61_base.py`.
2. Replace the tuned parameters with `{**BEST_PARAMS, "objective":
   "regression", "metric": "rmse", "linear_tree": True, "linear_lambda":
   1.0, "num_leaves": 220}`. Import `BEST_PARAMS` from `train_lgbm_v21`.
3. Read the iterations from `models/lgbm_r_all_v45plan.holdout.json`, key
   `v45plan.best_iters`. Keep the served-row scale. Expect 2,091 / 1,795 /
   986 rounds.
4. Write `lgbm_r_all_v61_*`. Copy `src_v3/predict_v57.py` to
   `src_v3/predict_v61.py` with `base_model="lgbm_r_all_v61"`.
5. Run `python -m src_v3.train_v61_base`, then `python -m src_v3.predict_v61
   --pre-ms kind-mango_v61_pre_ms.parquet --out kind-mango_v61.parquet`.

**R2. L13, v62 (0.5 h).** Copy `src_v3/predict_v53.py` to
`src_v3/predict_v62.py`. Point it to the v57 base, the v55 `R_norm` members,
the v56 gate and the v57 plan columns (`predict_v57.py:43-54`).

**R3. L12, v63 (1 h).** Add a keyword to `predict_v30.main` that is off by
default. When on, fill LIRF rows of the Step A input with `r_norm_lirf`
instead of the mixture, then copy only the cell rows back. Confirm v57
parity at 0.0 s with the keyword off.

**R4. L10, v64 (1.5 h).** Copy `src_v3/train_p_fb_v56.py`. Train six fold
boosters with `config.FOLDS` and the same rounds. Fit the isotonic map on
the concatenated out-of-fold scores. Save it as
`lirf_regime_v64.isotonic.pkl`. Keep the v56 booster.

**R5. L9, v65 (2 h).** Copy `src_v3/build_plan_taxi_res_v57.py`. Compute
`(ARVT_1 - EOBT_1)` minus its 12-month median per (`ADEP_mvt`, `ADES_mvt`,
`AIRCRAFT_TYPE_mvt`), clip +/-3,600 s. Add one column to the extra columns.
Retrain the base with the best parameters from R1.

**R6. L11 (2 h).** Rebuild `src/build_weather_eobt.py` output. Add its
columns to the extra columns. Retrain on the best recipe.

**R7. L2 (4 h, track M or B).** New builder `src_v3/build_nm_neighbours.py`.
For each DEP row, other DEP rows at the same `ADEP_mvt` with `MVT` in
`[t - W, t)` and proxy in 0 to 7,200 s; W = 30 and 60 min; groups airport
and runway. Output mean, median and count keyed on `MVT_ID_mvt`. Build on
training and ranking rows with one code path. Assert one row per id.

**R8. L1 (4 h, track B).** New builder `src_v3/build_nm_clocks.py` with the
six L1 columns keyed on `MVT_ID_mvt`. Add to the extra columns. Retrain
the base with the best recipe.

**R9. L3 (6 to 8 h).** Train CatBoost on the MB3 rows and the served
columns. Blend with the LightGBM mean at 0.3 and 0.5 before MS1.

**R10. L5 (4 h, track B).** Train on the anchored target, serve as in L5,
blend 50/50 with R8. Apply MS1 and MS2 after the blend.

### 11.4 Stop rule

- **Per upload.** Accept only if live is at or below best minus 0.3 s.
  The retrain noise is about 0.15 s (v52 +0.11 s, v57 -0.17 s).
- **Per lever.** Close a lever after two rejected uploads. Retry once only
  after a written defect fix.
- **Season.** On Fri 2 Oct, if the best is above 284.0 s on track A or
  above 275.0 s on track B, stop feature work. Spend the rest on stacks of
  accepted levers, reproducibility and the paper.
- **Hard stop.** No new feature family after Thu 8 Oct 23:59 UTC.

## 12. Repo corrections (applied on 2026-09-23)

1. `README.md`, `RECAP.md`, `docs/PRC_Data_Challenge_2026_BRIEF.md` and
   `docs/MODEL_ANALYSIS.md`: deadline 11 October 2026, 23:59:59 CET.
2. `README.md` and `RECAP.md`: v44 299.846, v45 296.125, v50 293.816. v45
   and v50 are new bests at upload; v47 is not. `docs/MODEL_ANALYSIS.md`
   holds a dated correction note for its v47 status line.
3. `README.md` and `RECAP.md`: the L1 retune cost +3.01 s; the 12-month
   refit is not isolated.
4. `README.md` ethics section: the quoted warning is the team's brief; v39
   read `LOBT_flt`; `AOBT_3_flt` is a noisy clock on 98.47 % of DEP rows.
5. `src_v3/train_p_fb_v56.py`: the docstring now matches the in-sample
   isotonic fit. The code is unchanged.
6. `RECAP.md`: the rank snapshot is from 2026-09-23.
7. `README.md`: the challenge home points to the new site, and the header
   points to this file.
