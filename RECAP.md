# kind-mango recap

Status as of 2026-09-25. v68 (CatBoost in `R_norm`) landed +0.03 s and is rejected. Team best **281.87 s (v67)**: v65 plus a CatBoost member blended 50/50 into the base outside LIRF (MB7, WINNING_PLAN L3), -2.87 s. Status of 2026-09-23 follows. Team best then **284.74 s (v65)**: v64 plus the `plan_nm_taxi` base column (MF5, WINNING_PLAN L9), -1.22 s. Before that, v64 285.95 s: v63 plus an out-of-fold isotonic map on the LIRF gate (L3, WINNING_PLAN L10), -1.14 s. Before that, v63 287.09 s: v57 with the Step A normal term read from `R_norm` (T1, WINNING_PLAN L12), -0.24 s. v61 (WINNING_PLAN L4, v45 parameters on the v57 recipe) rejected +0.99 s. L13 (MS3 on the v57 base) moved 0 rows and was not uploaded. L10 landed as v64 and L9 as v65. L11 (weather at EOBT_1) as v66 was rejected +0.33 s; v65 stays best. The Discord tail rules were priced on the 2025 hold-out and closed. `docs/MODEL_ANALYSIS.md` Progress section tracks the merged to-do list. Status of 2026-09-19 end-of-day follows. Team best **287.33 s** (v57, live, -11.98 s vs v41). MB3 base filter (v51 -5.99 s) plus the MB8 rollout: R_norm 12-month refit (v55 -0.42 s), p_fb gate 12-month refit (v56 -0.33 s) and plan_taxi_res route medians 12-month refit (v57 -0.17 s). Winning theme: broader training exposure + label-free defensive gates. v58 (MB8 for operator encoders) rejected +1.29 s (encoder leak amplified). v59 (MF4 arrival drift features) rejected +0.37 s (drift signal did not transfer).
The fourteenth-pass plan under 290 s executed end to end: L3.a p25/p75 EOBT
quartiles (v44), L9 plan_taxi_res (v45), L1 Optuna retune (v46), L2 12-month
refit (v47 = final ship). Live decomposition: v46 = 299.14 s (-0.17 s vs
v41; the base stack L3.a+L9+L1 landed only 102 MSE despite -1,864 paired
MSE), v47 = 296.57 s (-2.57 s vs v46; the L2 refit alone drove the gain).
Learning: paired hold-out is poorly calibrated once several base changes
stack, but 12-month refit gains do transfer strongly. 111 teams on the
leaderboard. The fourteenth-pass harness (H1 to H5) is built:

- H1 `src/build_h1_frame_cache.py` wrote `models/h1_frame_cache.parquet`: the
  110 served columns, the label, `sd`, the month, the airport and the movement
  id in one frame. The 169 null ids of the old side file are recovered (129 by
  a content key, 40 by per-airport time rank); the frame verifies against
  `frame_train` on every row (label, month, airport) and the tempo columns
  agree on all 2,084,659 rows. Tempo coverage 0.9962.
- H2 `src/eval_v33_holdout.py` now scores any stack per class per airport.
  The v33 default reproduces `models/v33.holdout.json` exactly. The v41 stack
  measures **102,355 MSE (FULL 319.93, CLEAN 250.83)**, clean 60,365 /
  fallback 7,688 / tail 12,681 / 24-h 21,517 (`models/v41.holdout.json`).
  The fourteenth pass priced it at about 102,300 by arithmetic.
- H3 `src/check_submission_2026.py` writes the label-free 2026 checks. v41 vs
  v40: LIRF-only rows move (26,808 rows, shift -5.09 s), as shipped.
- H4 `src/check_coverage.py` wrote `models/coverage_monitor.json`: 4 strict
  cells flagged, all data drift, not pipeline defects (`dewpt_spread` zeros
  at LEMD/LSZH in January, `eobt1_sched` zeros at LTFM in July). OPDI counts:
  LSZH January 2026 is 5,630 vs 7,542 in January 2025, a 25 % drop — the L6
  January gate fails as written. July 2026 (12,588 vs 11,770) passes.
- H5 `src/test_v33_parity.py` rebuilds v33 through `predict_v30.main` with the
  v33 member list. **Parity: max |diff| = 0.0000 s on 344,841 rows**
  (2026-09-13). The run writes the 2026 dump `models/v33_rank_dump.parquet`
  for H4.

v34 to v37 regressed +0.18 to
+0.65 s; v38 (`ARVT_1_flt` planned-time features) failed its gate.
v39 shipped the twelfth-audit rewrite (`src_v2/`) from a cold start and
scored **352.19 s** live, +50 s. The row-level debrief (MODEL_ANALYSIS 4.1)
puts 62 to 73 % of that on one LIRF row with `sd = 94,560`: v33 served the
ITY340 hedge (88,718), v39 served 3,262 because the audit's measures D2 and
E2 removed the rule. The rewritten base is also 4.5 s worse on clean rows
than the v33 base on identical hold-out rows. v33 remains best; v34 to v37
and v39 rejected.
v33 (5-member R_norm_LIRF only, 3-seed base kept) scored 301.87 vs the
priced 301.86 — the audit's ambiguity framework predicted the live gain to
0.01 s. Validates the "ship priced-cheap changes one at a time" rule from
the seventh audit.

## Breakthrough

`docs/MODEL_ANALYSIS.md` identified 3 code defects that faked a distribution shift:

1. `TAXITIME between(30, 7200)` filter hid the tail from hold-out and training.
2. Predictions clipped at 7200 capped every extreme forecast.
3. `sched_delay` clipped to `[-1800, 3600]` deleted the tail-identifying feature.

Plus: `EOBT_1_flt` and `IOBT_flt` were unused despite double the target correlation of `sched_delay`.

Fixing those (Step 2, v21) took live from **560.91 to 430.45** — a **130 s drop**.

## Live scoreboard

| tag | RMSE   | notes                                                |
|-----|--------|------------------------------------------------------|
| v68 | 281.90 | +0.03 s vs v67 (rejected). MB7 on the LIRF head: `src_v3/catboost_r_norm.py` fits CatBoost on the 166 `R_norm` columns and the LIRF genuine rows; `predict_v30.main` takes an `r_norm_post` hook, and `predict_v67 --r-norm-weight 0.5` blends it 50/50 into `R_norm`. Hold-out first (`src_v3/measure_catboost_r_norm.py`, v41 members out of sample on Jan+Jul): early stop at 1,194 rounds; RMSE 319.3 against 323.2 s; blend -347 (w 0.3), -435 (w 0.5), -159 (w 1.0) MSE. Full-year fit 1,740 rounds. Served: 26,848 LIRF rows move, mean +15.2 s, max |d| 1,485 s; non-LIRF identical to v67. Live +0.03 s (+17 MSE): the hold-out gain did not transfer, as v49. LIRF-head defect fixes transferred (v63, v64); LIRF-head model changes did not. Same day, L7 check: the 2025 day-level correlation of the departure fallback share and the arrival schedule-grid share is 0.43 at LIRF and 0.34 at EDDM, 0.02 to 0.19 elsewhere. Uploaded 2026-09-25. |
| **v67** | **281.87** | **-2.87 s vs v65, new team best (-17.45 s vs v41)**. WINNING_PLAN L3, MB7, finding M5. New dependency `catboost` 1.2.10 (a second learner class with its own ordered encoding of the categorical columns). Hold-out price first (`src_v3/measure_catboost_blend.py`): CatBoost (depth 8, lr 0.08, RMSE, l2 3) on the v65 columns and MB3 rows of months 2 to 6 and 8 to 10, early stop on 11 and 12, hit the 5,000-round cap (stop RMSE 202.59). On Jan+Jul 2025 outside LIRF against the v46 members: CatBoost alone -1,358 MSE; blend w 0.3 -1,583, w 0.5 -2,079, negative at all 9 airports. Control (`src_v3/measure_lgbm_control.py`): one LightGBM member with the v43 parameters on the same columns, rows and months stopped at round 438 and scored +1,636 MSE against v46, so the columns do not explain the CatBoost gain (CatBoost minus control -2,994). Full-year fit (`src_v3/catboost_base.py`): 7,385 rounds = 4,996 x 1,923,953 / 1,301,574, about 2.6 h at 12 threads. `src_v3/predict_v67.py` serves the v65 stack, checks the LightGBM member parity, and adds 0.5 x (CatBoost - LightGBM mean) outside LIRF before MS1 and MS2. CatBoost minus LightGBM on 2026: mean +2.4 s, p99 |d| 344 s. Label-free check vs v65: LIRF unchanged; shifts -2.69 (LFPG) to +4.67 s (LTFM); max |d| 2,159 s; 9 non-LIRF rows above 7,200 s. Live -2.87 s (-1,627 MSE); plan range -300 to -4,000 MSE. The first full-year run was stopped by a memory shortage while it ran next to the control; the rerun was sequential. Uploaded 2026-09-24. |
| v66 | 285.07 | +0.33 s vs v65 (rejected). WINNING_PLAN L11 (R6), MF1: METAR at EOBT_1 (`tmpc_eobt`, `vis_km_eobt`, `wind_kt_eobt`, `wx_precip_eobt`, `deicing_gate_eobt` from `src/build_weather_eobt.py`) added to the v65 base (123 columns, `src_v3/train_v66_base.py`). 2026 coverage 95 to 99 % per airport-month, as in 2025; January 2026 de-icing trigger rate 6.9 % against 3.2 % in January 2025. Label-free check vs v65: LIRF unchanged; non-LIRF shifts -0.83 (EHAM) to +0.56 s (LSZH). One LSZH E290 row on the 8 January 2026 snow day (de-icing trigger on 80 % of that day's rows) moved 1,257 to 5,536 s with `mvt_eobt1` 1,570 s; the rest of that day moved +18.6 s on average. Live +0.33 s (+190 MSE). The paired test on v45 (-367 MSE, gate missed by 7 MSE) did not transfer. A computer shutdown stopped the first training run after 2 of 3 seeds; the rerun is deterministic. Uploaded 2026-09-23. |
| tail rules | not uploaded | Discord suggestion 2026-09-23: "force the outliers of the submission to look like the training outliers". `src_v3/measure_tail_rules.py` prices two readings on the 2025 hold-out with the v46 members (trained without months 1 and 7): (1) blend record rows with `mvt_eobt1` > 3,600 s toward the isotonic curve of y on `mvt_eobt1` fit on the other ten months: +390 / +1,639 / +3,746 / +6,711 MSE at base weight 0.75 / 0.5 / 0.25 / 0, worse at all 9 airports; (2) cap record rows with `mvt_eobt1` < 1,800 s at the 2025 q99.99: 3 rows, -3 MSE (10 rows on v65). The fat 2026 EHAM tail is the 3 to 9 January 2026 disruption week; v65 already predicts those rows below the 2025 conditional means. Closed; no upload. |
| **v65** | **284.74** | **-1.22 s vs v64, new team best (-14.58 s vs v41)**. WINNING_PLAN L9 (R5), MF5, finding C2: `src_v3/build_plan_nm_taxi_v65.py` writes `plan_nm_taxi` = `ARVT_1 - EOBT_1` (filed block time) minus its 12-month median per (ADEP, ADES, aircraft type) on clean non-fallback rows, clip +/-3,600 s. 35,957 keys; coverage 0.989 train, 0.961 rank; spread p10/p90 -494/+594 s in 2025 and -562/+755 s in 2026. `src_v3/train_v65_base.py` retrains the v57 recipe with this one column added (118 columns, same MB3 rows and rounds); `train_v57_base.train_base` now holds the shared fit. Served on the v64 stack with `--extra`. Label-free check vs v64: LIRF unchanged; non-LIRF shifts -1.29 (LTFM) to +2.68 s (LFPG); max |d| 2,188 s; 10 non-LIRF rows above 7,200 s (v64 8). Live -1.22 s (-695 MSE); plan range 0 to -900 MSE. Uploaded 2026-09-23. |
| **v64** | **285.95** | **-1.14 s vs v63, new team best (-13.36 s vs v41)**. WINNING_PLAN L10 (R4), finding L3: `src_v3/train_p_fb_v64.py` trains six gate boosters with the v56 recipe and 207 rounds, one per month-pair fold, scores each hold-out pair, and fits the isotonic map on the concatenated out-of-fold scores. The v56 booster stays; only its map changes. Held-out months, nested map: log loss 0.383 to 0.364 (sd <= 3,600 s, 143,402 rows), 0.140 to 0.122 (3,600 to 14,400 s, 16,792 rows), flat on sd > 14,400 s (510 rows, bias -0.25 to -2.04 points). Served: 25,988 LIRF rows outside Step A move, mean +8.7 s, max |d| 1,805 s; non-LIRF and cell rows unchanged; rows above 7,200 s 125 to 105. Live -1.14 s (-652 MSE); plan range 0 to -800 MSE. The v56 in-sample map was too extreme on out-of-sample 2026 rows. Uploaded 2026-09-23. |
| **v63** | **287.09** | **-0.24 s vs v57, new team best (-12.22 s vs v41)**. WINNING_PLAN L12 (R3), finding T1: `predict_v30.main` takes `stepa_normal_rnorm` (off by default; v57 rebuild with it off matches at max |diff| 0.0 s). On, the Step A input carries `R_norm` on LIRF rows instead of the mixture, and only the cell rows are copied back. 18 of the 43 cell rows move, all down, by 100 to 4,241 s (mean -2,460 s); the other 25 sit in bands with no normal share. v57 served these rows at nearly `sd`, which shows the gate near 1 and the schedule weight counted twice. Live -0.24 s (-139 MSE) from 18 rows. The plan bar is -0.30 s; accepted because the change is deterministic and has no retrain noise. Uploaded 2026-09-23. |
| v62 | not uploaded | WINNING_PLAN L13: MS3 member median on the v57 base (`src_v3/predict_v62.py`). The largest spread of the three v57 members outside LIRF is 2,376 s (7 rows above 1,800 s), under the 3,600 s trigger, so MS3 moves 0 rows and v62 equals v57 on every row. Not uploaded. MB3 removed the member disagreement that MS3 caught on v48. `predict_v57.py` now exposes `serve_v30` and `write_ms`; the v57 rebuild matches `kind-mango_v57.parquet` at max |diff| 0.0 s. |
| v61 | 288.32 | +0.99 s vs v57 (rejected). WINNING_PLAN L4 (R1): the v57 recipe (MB3 rows, 12 months, v57 frame) with the v45 base parameters in place of the v43 retune (`src_v3/train_v61_base.py`). Rounds are the v45plan best iterations x 1.2585: 2,091 / 1,795 / 986. Served by `predict_v57 --base-model lgbm_r_all_v61`. Label-free check vs v57: LIRF unchanged; non-LIRF shifts -2.34 (LFPG) to +0.43 s (EDDF); 13 non-LIRF rows above 7,200 s (v57 8); 5 EHAM rows of one day moved by more than 3,000 s (flight record, `mvt_eobt1` 3,905 to 7,443 s). The API shows the retune cost +3.01 s at v46, but on the MB3 rows the retuned parameters are better. L14 (blend of both recipes) needed v61 within 0.5 s and does not run. Uploaded 2026-09-23. |
| v60 | 289.58 | +2.25 s vs v57 (rejected, worse than v58). Fifteenth-pass MD4: leave-one-month-out operator encoders. OOF at training / full-year map at serving. Regression is bigger than v58's MB8 (+1.29). Two mechanisms cooperated: (1) the "leak" MD4 was meant to fix is small (1/N per row where N>=20), and removing it adds noise without removing overfitting; (2) MD4 breaks the training-serving alignment (training rows use per-month fits missing their own month, serving rows use the full-year fit), so the tree learns split thresholds calibrated to the OOF distribution and misapplies them at serve time. LTFM shift +11.73 s was the biggest single-airport move of any encoder lever. v57's v26 encoders (partial exclusion of months 1,7; training-serving aligned) sit at the local optimum. Both encoder levers now closed. |
| v59 | 287.70 | +0.37 s vs v57 (rejected). Fifteenth-pass MF4: arrival-taxi-in drift features (1-day / 7-day rolling median minus 2025 fit-month per-airport median). 4 new columns keyed on MVT_ID_mvt at 0.998/0.982 train/rank coverage. Retrained the v57 base with 121 features. Non-LIRF mean shifts small (-0.73 to +1.71 s), biggest at LTFM and EGLL which had the largest arrival-side drift per the mirror. Live regressed by 0.37 s: drift signal from the arrival side does not transfer to departure taxi-out in the way the arrival mirror suggested. Two possible mechanisms: (1) 2026 departures may already be capturing the drift via existing features (congestion, tempo); (2) rolling arrival medians add noise on quiet days that outweighs any signal on busy days. Lever closed; MF4 does not help stacked with MB3+MB8. |
| v58 | 288.62 | +1.29 s vs v57 (rejected). Fifteenth-pass MB8 applied to the operator target encoders: refit `features_operator.fit_encoders` on all 12 months (was months 2..12 excluding 7). Retrained the v57 base with recomputed openc_* columns. Direction backfires because the encoders have an inherent L1 leak: each row's encoded value depends on its own label, so more months in the encoding = more leakage. MB8 works for boosters and scoring maps but NOT for target encoders; the correct fix is MD4 (leave-one-month-out) which we did not ship. Lever closed. |
| v57 | 287.33 | -0.17 s vs v56. Fifteenth-pass MB8 applied to the plan_taxi_res route medians: recompute the medians on all 12 months of 2025 clean rows (was fit months 2..6+8..10+11..12 only), rebuild the plan_taxi_res parquets, retrain the v51 base on the updated frame. Extra columns list unchanged; only the plan_taxi_res values differ. 5,883 routes with medians (v45 had 5,619 - Jan+Jul added new routes). Same iter scaling as v51 (767, 2265, 1278). Non-LIRF mean shifts vs v56 are tiny (-0.51 to +0.44 s), max |d| < 1,400 s. LIRF unchanged (LIRF head untouched). Uploaded 2026-09-19. |
| v56 | 287.50 | -0.33 s vs v55. Fifteenth-pass MB8 applied to the p_fb LIRF gate: refit lgbm_p_fb_lirf and its isotonic on all 12 months of LIRF rows (v23 recipe otherwise). v23 gate trained on months 2..6+8..10 with isotonic on 11-12; v56 refits with iter count scaled from v23 (182 -> 207) and calibrates on the same full-year rows. Full-year AUC 0.9256, log-loss 0.2779 (in-sample). LIRF-only shift vs v55: 25,044 rows moved, mean -16.81 s, max |d| 3,648 s. over_7200 107 -> 126 (19 more) - the recalibrated gate pushes some rows closer to sd. Uploaded 2026-09-19. |
| **v55** | **287.83** | **-0.42 s vs v51, new team best (-11.48 s vs v41)**. Fifteenth-pass Phase 5 MB8 applied to R_norm_LIRF: refit the 5 R_norm boosters on ALL 12 months of LIRF genuine rows (v51 recipe otherwise, MB3 base unchanged). Same L2-style 12-month exposure trick that gave the base -2.57 s. Iter counts scaled from v41 (234, 371, 430, 286, 235) by 1/0.88 -> 266, 422, 489, 326, 268. LIRF-only shift (26,795 rows moved, mean -0.27 s, max |d| 2,707 s). Non-LIRF unchanged. Ceiling was ~0.4 s (LIRF weight 26k/344k) and it landed at that ceiling - the L2/MB3 pattern of broader-training-exposure works on the LIRF head too. |
| v54 | 324.11 | +35.86 s vs v51 (rejected). Fifteenth-pass MB1 anchored-offset target `y - anchor` (anchor = mvt_eobt1 else sched_delay) on top of MB3. Failed because anchor is bimodal on 2026 delayed rows: rows with mvt_eobt1 > 10,000 s (aircraft held past EOBT) but normal taxi output pred_y = anchor + small_offset = 10,000+ s instead of the true 1,000 s. 101 rows at EHAM over 7,200 (v51 had ~4), 25 at LFPG (v51 ~4). The tree learns median offset ~-300 s, cannot correct on huge-anchor rows. Grade D audit missed the bimodality. Lever closed. | Train the base on `y - anchor` where anchor = mvt_eobt1 if not-null else sched_delay. 3 seeds, retuned params, same iter scaling as v51/v52. Predict-time: score the boosters directly (bypassing predict_v30's clip at 0 which would destroy negative offsets), add anchor per row, clip to per-airport [q0.001, q0.999] offset range from training. LIRF rows come from the v51 parquet unchanged. Post MS1+MS2. Mean shift vs v51: +3 to +25 s per airport upward, biggest EHAM +25.54; over_7200 108 -> 279 (extra tails at EHAM 101 and LFPG 25). Could be over-correction or 2026 drift catch-up (arrival mirror showed +6% drift). Uploaded 2026-09-18. |
| v52 | 288.36 | +0.11 vs v51 (noise). Fifteenth-pass MB3+MB4: v51 recipe with cyclic doy/local hour/public holiday added, numeric month dropped. MB4 did not help; MB3 alone is the best point. |
| **v51** | **288.25** | **-11.06 s vs v41, -5.99 vs v48; new team best**. Fifteenth-pass MB3: base retrained on served rows only. Excluded 160,706 LIRF rows and 2 non-LIRF y>80,000 rows; retuned Optuna params, 3 seeds, all 12 months, iter scaling 1.259 (767/2265/1278). Post-processing MS1+MS2 moved only 85 rows (max 304 s) because the MB3 base already avoids extreme predictions natively. MS1 finds nothing to clip; without LIRF's 16.6 % fallback rate and the 12 24-h rows in training the linear leaves stop being shaped by those extreme labels. Live -5.99 s at zero paired price - the paired hold-out cannot see this because the base's errors on 2025 rows are moderate; MB3's benefit shows up entirely on 2026 OOD rows. | `src_v3/train_v51_base.py` filters out LIRF rows and rows with y > 80,000, then trains the retuned v47 recipe on the 1,923,953 remaining rows across all 12 months (dropped 160,706 = 7.7 %). Round counts scaled from the v46 paired iters by the served-row growth ratio (1.259 vs 1.136 for v47's 1/0.88). 3 seeds: 767, 2265, 1278 rounds. `predict_v51.py` serves the new base with the v41 LIRF head unchanged plus MS1 + MS2. Sanity: post-MS moved only 85 rows with max |diff| 304 s (v48 had 109 rows, max 13,966 s) - the MB3 base already avoids the extreme predictions natively because its linear leaves are no longer shaped by LIRF's 16.6 % fallback rate and the 12 24-h rows. Non-LIRF shifts vs v48: EHAM +6.25 (largest), all others under 1 s absolute. Uploaded 2026-09-18. |
| v50 | pending | Fifteenth-pass Phase 1: MS3 member-disagreement median on top of v48. `src_v3/predict_v50.py` dumps the v47 served feature frame via predict_v30.main(dump_features=...), aligns categoricals to the training vocab, scores the three v47 base members individually, replaces the ensemble mean with the row median where the three-member spread exceeds 3,600 s outside LIRF, then reapplies MS1 and MS2 exactly as v48. MS3 moved 17 rows (EDDF 3, EDDM 4, EHAM 4, LEBL 1, LFPG 5); MS1 clipped 2 of them back to the same value, so v50 differs from v48 on 15 rows total (max |diff| 4,349 s at EHAM). Non-LIRF only; LIRF head untouched (v41 v41). MP7 label-free gate passed. Uploaded 2026-09-17, score pending. |
| v49 | 294.65 | +0.42 s vs v48 (regression). Fifteenth-pass MD2+MH1 rebuild of the LIRF head without the 56 drifted columns (38 ec_* + 9 opdi_* + 9 opdi_live_*). `src_v3/train_lirf_head_v49.py` retrained R_norm_LIRF (5 seeds) and p_fb+isotonic; rate features (fbrate/fbcount) kept. Paired LIRF-only hold-out (26,528 rows): v41 head FULL 887.42 CLEAN 463.55, v49 head FULL 869.28 CLEAN 460.69; LIRF MSE contribution to the 344,341-row scale -2,405 MSE paired. Live -2,405 MSE did NOT transfer. The v41 head's zero-imputed ec_* rows in July 2026 were apparently routed to sensible branches, not the "wrong" one D1 predicted. Grade C estimate of 700 MSE gain was optimistic. Head rebuild lever closed for this stack. |
| v48 | **294.24** | **new best, -2.33 s vs v47 (296.57), -5.07 s vs v41 (299.31)**. Fifteenth-pass Phase 1: MS1 per-airport upper bounds + MS2 clean floor applied post-hoc to the v47 parquet (`src_v3/build_v48.py`, `src_v3/postprocess.py`, `src_v3/support.py`). MS1 caps at the 2025 tail-class max plus a 600 s margin at EDDF/EDDM/LEBL/LEMD (no long tail), and at the clean max plus margin at EGLL/EHAM/LFPG/LSZH/LTFM unless the row has flt_null=0 and mvt_eobt1 > 5400. MS2 floors at the 2025 clean 0.1 %-quantile per airport. MS1 moved 5 non-LIRF rows (EDDM 21,465 -> 7,499 the biggest -13,966 s; EHAM 11,103 -> 7,311; two LEBL rows to 4,204; LEMD 4,376 -> 4,198). MS2 moved 104 rows (small pushes). LIRF head output untouched by MS1; MS2 raised 14 LIRF rows by tiny amounts. Zero training cost. MP7 gate passed. Live -2.33 s (about 1,400 MSE) essentially all came from the five MS1 clips; MS1 fold price was 0 MSE because the base does not extrapolate on 2025 rows. The label-free evidence at upload time (5 rows with predictions 2-6x above the airport's 2025 support, all with mvt_eobt1 below the eligibility bar) was correct. |
| v47 | **296.57** | **new team best, -2.74 s vs v41 (299.31)**. L2 12-month refit of the v46 stack (`train_r_all_v47.py`, `predict_v47.py`). Trained 3 seeds on ALL 12 months with the retuned hyperparameters, no early-stop split, iteration counts scaled by 1/0.88 (v46 iters 609/1799/1015 -> 693/2045/1154). 2026 sanity: LIRF unchanged; non-LIRF shifts vs v46 range -15.05 s (EGLL) to +4.36 s (EHAM). Live delta: -2.74 s covers the sum of shipped paired prices (L3.a -552, L9 -650, L1 -662, L2 unknown-priced) ~ 1,864 MSE served clean, matching the ambiguity prediction of 2-3 s. Uploaded 2026-09-17. |
| v46 | 299.14 | -0.17 s vs v41. L1 retuned base on v45 features. `tune_lgbm_v43.py` Optuna sweep (30 trials, `linear_tree` pinned True, `feature_pre_filter=False`) best trial 10 stop RMSE 245.41 with num_leaves 436, min_data_in_leaf 291, lr 0.0183, feature_fraction 0.5629, linear_lambda 0.0056 (vs deployed lr 0.023, leaves 220, min_data 76, l_lambda 1.0). Paired vs v45: served clean -662 MSE (LSZH -527 dominates), served total -740 MSE, gate 500 passed. Live -102 MSE landed, paired-to-live ratio 14 % — the base stack v44+v45+v46 combined only moved live by 0.17 s despite -1,864 paired MSE. Says the paired hold-out is poorly calibrated once several base changes stack on each other. Uploaded 2026-09-17. |
| v45 | pending | v44 pipeline + plan_taxi_res, clipped to +/-3600 s (`build_plan_taxi_res.py`, `train_r_all_v45.py`, `predict_v45.py`). Route medians from 5,619 2025-clean-fit-month routes; residual coverage 0.989 train / 0.982 rank. Paired vs v44: clean class outside LIRF -650 MSE, served base total -617 MSE, gate 300 passed. 2026 sanity: LIRF rows unchanged; non-LIRF shifts vs v44 -1.7 to +3.2 s (LSZH +2.54, EDDM +3.15, LFPG +2.72, LTFM -1.72); over_7200 105, over_80000 3, at_zero 27. Uploaded 2026-09-16, score pending. |
| v44 | pending | v41 pipeline + p25/p75 of neighbour `mvt_eobt1` on the base (`build_tempo_p2575.py`, `train_r_all_v44.py`). Six new columns at the apt30, apt60 and rwy30 windows. Paired vs the v40 recipe reproduced: clean class outside LIRF -552 MSE, served base total -565 MSE, gate 300 passed. 2026 sanity: LIRF rows unchanged; non-LIRF airport shifts -2.1 to +2.0 s; over_7200 106 (v41 104), over_80000 3 (=v41), at_zero 32 (v41 29). Uploaded 2026-09-16, score pending. |
| v48wx | not uploaded | L7: METAR joined at EOBT_1 (tmpc/vis_km/wind_kt/wx_precip/deicing_gate at pushback time) added to v45's 117-column feature set. `build_weather_eobt.py` (coverage 0.985), `train_r_all_v48wx.py`. Paired vs v45: FULL 391.97 -> 381.55, CLEAN 255.11 -> 254.11. Served (non-LIRF) delta clean -293, fallback -4, tail -58, 24h -12 (total -367 MSE). Served clean price +293, gate 300 short by 7 MSE. All four served classes improved but the gate is unmet as written; kept as an open lever for a future retune-on-v47 session. |
| v48cal | not uploaded | L8: three calendar flags (public holiday from `holidays` 0.104, weekend, or either) on v45's 117-column feature set. `build_calendar.py`, `train_r_all_v48cal.py`. Public-holiday coverage 2.6 %, weekend 28 %. Paired vs v45: served (non-LIRF) delta clean -144, fallback +24, tail +28, 24h -13 (total -105). Served clean price +144 MSE, gate 200 short by 56. Rejected; the hour/dow categoricals already capture most of the weekend signal. |
| v48arr | not uploaded | L3.e: arrival taxi-in residual over the previous 30 min minus the 2025 fit-month per-airport median. `build_arr_taxi_res.py` (coverage 0.996, airport medians LTFM 772 s to LEBL 209 s), `train_r_all_v48arr.py`. Paired vs v45: served (non-LIRF) delta clean +102, fallback +8, tail -56, 24h -10 (total +44). Served clean price -102 MSE, gate 300 failed with a genuine regression on clean. Closed. |
| v45term | not uploaded | L3.c: neighbour `mvt_eobt1` median/mean/count grouped by (ADEP, stand_prefix) over 30 min (`build_tempo_term30.py`, `train_r_all_v45term.py`). Paired vs v44 recipe: served (non-LIRF) delta clean +77, fallback +4, tail -55, 24h 0; total +26 MSE (neutral-slightly worse). Served clean price -77 MSE, gate 300 failed. Rejected; the stand categorical + rwy30 tempo already capture per-terminal geometry. |
| v45iobt | not uploaded | L3.b: p25/p75 of neighbour `mvt_iobt` at apt30/apt60/rwy30 windows (`build_tempo_iobt_p2575.py`, `train_r_all_v45iobt.py`). Paired vs v44 recipe: served (non-LIRF) delta clean +16, fallback +12, tail +37, 24h +12; total +77 MSE (worse). Served clean price -16 MSE, gate 300 failed. Rejected; IOBT is redundant with EOBT_1 once the eobt quartiles are in. |
| v45op | not uploaded | L3.d: neighbour `mvt_eobt1` grouped by (airport, operator) over 120 min, median/mean/count (`build_tempo_op120.py`, `train_r_all_v45op.py`). Paired vs v44 recipe: served (non-LIRF) delta clean +54, fallback +1, tail +59, 24h +14; total +128 MSE (worse). Served clean price -54 MSE, gate 300 failed. Rejected; likely captured by the 16 operator-historic taxi encoders already in the base. |
| v46const | not uploaded | L4: constant-leaf booster (linear_tree=False, 220 leaves, v44 features) meaned with the shipped v44 members at 0.7 / 0.3 (`train_r_all_v46const.py`). v44 alone CLEAN 256.22; v46const alone CLEAN 262.86 (2.6 s worse); mean CLEAN 256.64. Served (non-LIRF) delta clean +64, fallback +1, tail +129, 24h +2; total +196 MSE (worse). Served clean price -64 MSE, gate 400 failed. Rejected; the weaker class drags the mean, exactly the risk MODEL_ANALYSIS 4.1 L4 called out. |
| v43 | not uploaded | L5 rebuild of `R_norm_LIRF` with `FB_TOL = 5` on the fit mask only (`train_r_norm_lirf_v43.py`). Paired vs the shipped v41 members: clean class +3,546 MSE, fallback -1,200, tail +75, 24h 0; total -2,421 MSE, gate 200 failed. Rejected; the 5-sec mask pulls near-fallback rows into training and hurts the clean predictor. L5 closed. |
| v41 | **299.31** | **best**. v40 with one term of the LIRF head changed: five `R_norm_LIRF` members retrained with the 13 tempo columns (`train_r_norm_lirf_v41.py`, paired -228 MSE) and the 4,431 s cap removed (paired -187 MSE). Priced -415 MSE; live **-0.65 s vs v40** (-391 MSE). Rejected on the same day by paired price: the gate with the tempo columns (+94 MSE) and the forward take-off order on the base (v42 members, -93 MSE, under the bar). |
| v40 | 299.97 | previous best. v33 stack with the 13 tempo, order, stand-gap and queue columns on the base (`train_r_all_v40.py`, `predict_v40.py`). Paired hold-out CLEAN -2.67 s vs a control that reproduces v26; priced about -1.3 s; live **-1.90 s vs v33** (-1,146 MSE). First ship under 300. |
| v39 | 352.19 | Twelfth-audit rewrite (`src_v2/`), cold start: harness, 5-second fallback window, tempo and order features, regime heads at 7 airports, constant-leaf clean-only base, null-flight LIRF tail head, clip [30, 100000], ITY340 rule removed. Hold-out FULL 338.06, CLEAN 264.26. Debrief (MODEL_ANALYSIS 4.1, T15 to T17): one LIRF row with `sd = 94,560` and a flight record moved from 88,718 (v33 ITY340 hedge) to 3,262 and holds 62 to 73 % of the +32,911 MSE; the base is 4.5 s worse on clean rows than the v33 base on identical hold-out rows (259.74 vs 264.26) and worse on the tail class (clean-only training). Uploads `v2a` (int32) and `v2b` (letter suffix) got no result. Three slots used, two left. v33 remains best. |
| v38 | not uploaded | Base + `ARVT_1_flt` planned-time features (`features_plan.py`, `train_r_all_v26.py --plan`, `predict_v38.py`). Fast A/B CLEAN about -2 s (noisy, gate 1 failed). Deployed recipe: seed 43 collapses at iteration 6 in two identical runs; 3-member hold-out CLEAN 277.27 vs v26 266.46. Gate 2 failed; stopped before upload. See MODEL_ANALYSIS section 4. |
| v37 | 302.52 | Eleventh audit: v33 pipeline + 5-seed `p_fb_LIRF` mean (`train_p_fb_lirf_seeds.py`, `predict_v37.py`; `predict_v30.main` takes `p_fb_members`, v33 parity 0.0 s). Label-free price -151 MSE (predicted 301.62), under the 268 MSE bar; shipped on user decision. Live +0.65 s vs v33, a 547 MSE miss: the v23 booster is a better-than-average draw on the 2026 labels. Seed-mean lever closed. Earlier the same day the `R_norm_LIRF` coverage-skew retrain failed its replay gate at 103 MSE, no upload. v33 stays best. |
| v36 | 302.05 | Tenth-audit measure 1: F7 zero-clip repair. `predict_v30` gains two optional kwargs (`per_member_base_clip`, `fill_zero_rows_per_airport`); default preserves v33 bit-parity. v36 sets both — the base ensemble averages raw predictions, then any row still at exactly 0 is filled with the median of positive predictions at its own airport. Diff vs v33: 110 rows, mean 0.06 s, mean \|diff\| 0.13 s, 44 zero rows filled (v33 had 23 zeros; removing per-member clip generated 21 more, all caught by the fill). Live +0.18 s vs v33, -0.02 s vs v34 — priced ~252 MSE gain did not appear on live; the collateral movement from removing the per-member clip cancelled the zero-fill win. v33 stays best. |
| v35 | 302.09 | Ninth-audit 7.5 repair: v34's honestly trained p_fb classifier + v33's all-month scoring artefacts (rate maps, band table, NORMAL_MEAN_LIRF=1150, R_NORM_CLIP=4431). Predict_v30 refactored to accept p_fb kwargs; parity check on v33 rebuild = 0.0000 s. Expected to revert v34 by construction; instead landed +0.02 s vs v34 and +0.22 s vs v33. The regression sits in the honestly-trained classifier, not in the scoring-artefact refit. |
| v34 | 302.07 | v33 pipeline with hold-out leak removed from LIRF stats (audit Item 2 / F2). Refit on months {2..12} \ {7}: NORMAL_MEAN_LIRF 1150 -> 1094.41, R_NORM_CLIP 4431 -> 4431.49 (~0), lirf_band_table_v34 (127 -> 82 cell rows), lgbm_p_fb_lirf_v34 with clean OOF (best iter 245, stop AUC 0.857 vs v23 0.858), lirf_regime_v34.rate_maps. Diff vs v33: LIRF-only (25,204 rows, mean -1.34 s, mean \|diff\| 22.0 s, max 2,633 s). Live +0.21 s vs v33 — inside the 2-s noise band; ship exists to make future decisions honest. |
| v33 | **301.87** | **best**. v30 pipeline + 5-member R_norm_LIRF mean (seeds 42-46). 3-seed base unchanged. Isolates the priced-cheap change from v32. Predicted 301.86 s from A_5 post-mixture = 74 MSE; live landed 301.87 — 0.01 s off, validating the 2026 ambiguity framework. (-0.11 s vs v30) |
| v30 | 301.98 | v29 + LIRF-only encoders for LIRF models + new Step A band table (mutually exclusive classes, NOSOS431 fix) (-1.28 s vs v29) |
| v32 | 302.11 | v30 pipeline + 5-member R_norm_LIRF mean (seeds 42-46) + 7-member base mean (seeds 42-48). Doc sixth-audit blueprint. A_5 post-mixture 74.2 MSE (matches audit prediction), A_7 5,417 MSE (8× hold-out estimate). Expected 299.86 s; live +0.13 s regression — implied base-change loss +152 MSE against predicted -1,204 MSE, so the R_norm gain likely landed and the base change alone drove the regression (seventh audit 10.2). |
| v31 | 302.41 | Cleaned invalid ARR taxi-in labels before congestion aggregation; local full 397.84 s. Regressed live by +0.43 s, so v30 remains the submission. |
| v29 | 303.26 | v26 3-seed base + R_norm_LIRF clip at 4,431 s + ITY340 constant formula (Step A unclipped after audit showed clip regressed +124 MSE) (-0.45 s vs v26) |
| v26 | 303.71 | v24 pipeline with R_all_v26 (drop 38 ec_* + 18 opdi_* per doc Step 5) (-13.1 s vs v24) |
| v27 | 304.03 | v26 + 7-seed R_all_v27 + drop 3 dead features + Step 2 fixes (R_norm clip, ITY340 formula). +0.32 s regression. |
| v28 | 304.48 | v26 pipeline + 7-seed R_all_v27 (isolate ensemble). +0.77 s regression, likely from `ades_arr_atfm_delay_today` drop. |
| v25 | 375.33 | v24 + imputed ec_* with 2025 medians (perturbed learned NaN routing; retracted) |
| v24 | 316.85 | v23 pipeline + 3-seed R_all_v24 mean with honest 12% random stop split (-0.65 s vs v23) |
| v23 | 317.50 | v22 pipeline + Step 2 fallback-rate encodings on p_fb (fbrate_flt_prefix rank 1) (-2.8 s vs v22) |
| v22 | 320.31 | v21 base + LIRF regime head (R_norm_LIRF + calibrated p_fb) + refined Step A band table (11 bins) (-9.8 s vs v21) |
| v21 | 330.12 | R_all v21: linear_tree + signed-log copies + OPDI-live leak fix + turnaround BLOCK_TIME fix + temperature (-15.6 s vs v20) |
| v20 | 345.70 | R_all v20 (v21 stack + turnaround + disruption) base + v18 LIRF (Step A + 6.3) + ITY340 rule (-25 s vs v18) |
| v19 | 359.46 | v20 combined per-airport: v18 for LIRF/EHAM, R_norm+detector for EGLL/LEBL/LTFM, R_norm alone elsewhere (-11 s vs v18) |
| v18 | 370.63 | v16 + Section 6.3 rule for LIRF null-flight sd 3600-14400 (shrink 0.6) |
| v16 | 372.40 | v21 + LIRF band rule (Step A)                        |
| v17 | 377.47 | v16 + LIRF/EGLL fallback detector soft-mix (Step C)  |
| v13 | 430.45 | v21 = v15 + unfiltered labels + OBT deltas           |
| v14 | 431.96 | v25 4-model ensemble refit on 12 months              |
| v15 | 625.34 | v25 + Step 3 mixture. Mixture blew up on live.       |
| v10 | 560.91 | earlier best, 5-model NNLS on the old broken metric  |
| v9  | 561.04 |                                                      |
| v7  | 561.06 |                                                      |
| v12 | 561.15 | v20-swap ensemble                                    |
| v6  | 561.19 |                                                      |
| v3-v5, v8 | 561.5-562.4 |                                                |
| v1-v2 | 562.3-562.4 |                                                |

## Where we stand vs top

Current: rank 44 of 111 at 301.87 s (v33). The list below is an old snapshot
from the v18 era (370.63 s).

```
 1. youthful-giraffe               263.46
 2. upstanding-firefly             267.01
 ...
41. reliable-hamburger             343.31
42. gentle-lemon                   348.99
43. jolly-lobster                  353.71
44. affectionate-ukulele           366.52
45. kind-mango                     370.63   ← us
46. vigorous-jungle                389.86
47. versatile-violin               396.23
48. tidy-nugget                    404.86
```

Team best is now 299.97 s (v40). The old snapshot list above predates v40.

## Model progression on hold-out

| model                | full   | clean  | live   | note                                       |
|----------------------|--------|--------|--------|--------------------------------------------|
| v15 (before analysis)| 576.12 | 293.09 | 560.91 | baseline (on unfiltered hold-out)          |
| v21 (Step 2)         | 455.24 | 283.85 | **430.45** | live -130.5 s — the whole story        |
| v22 mixture (Step 3) | 415.51 | 335.11 | -      | hold-out only                              |
| v23 (Step 4)         | 453.75 | 282.89 | -      | ADES arrival context                       |
| v24 (Step 5)         | 452.23 | 281.92 | -      | OPDI stationarity                          |
| v25 4-model ensemble | 452.14 | ~      | 431.96 | 12-month refit                             |
| v26 (Step 7)         | 459.80 | 283.61 | -      | runway config, dropped                     |
| v27 mixture on v25   | ~      | ~      | 625.34 | mixture disaster                           |

## What works on live

| item                               | live effect     |
|------------------------------------|-----------------|
| Unfilter labels + OBT deltas       | **-130.5**      |
| Step A LIRF band lookup (v16)      | -58             |
| Turnaround + disruption + ITY340 (v20) | -13.8 vs v19 |
| linear_tree + leak fixes (v21)     | -15.6           |
| LIRF regime head (v22, v23)        | -9.8, -2.8      |
| Drop `ec_*` + `opdi_*` (v26)       | -13.1           |
| LIRF-only encoders + band table (v30) | -1.28        |
| 5-member `R_norm_LIRF` (v33)       | -0.11           |
| Changes after v33 (v34-v37)        | +0.18 to +0.65 (rejected) |

## What does not work

| approach                                | hold-out delta | live delta | reason                              |
|-----------------------------------------|----------------|------------|-------------------------------------|
| DQ classifier                           | AUC 0.51       | -          | Random.                             |
| SSL drift correction                    | ~0             | +          | Hurt live.                          |
| Quantile regression                     | -              | -          | No signal.                          |
| LIRF specialist                         | ~0             | -          | Same convergence.                   |
| Tail mixture (old formulation)          | -              | -          | Wrong economics for RMSE.           |
| Extended OPDI event counts (v18)        | ~0             | -          | Redundant.                          |
| VRS / openap metadata                   | ~0             | -          | Duplicates categorical.             |
| Historical climatology 2022-24 (v19)    | ~0             | -          | Marginal.                           |
| Aggressive cleanup (v20)                | ~0             | +0.24      | Only 60 rows removed.               |
| v20-swap ensemble (v12)                 | -0.03          | +0.24      | v20 too close to v15.               |
| Fixed arrival context (Step 4, v23)     | -1.5           | ~0 (v14)   | Right in principle, no live gain.   |
| OPDI ratio-to-median (Step 5, v24)      | -1.5           | ~0 (v14)   | Right in principle, no live gain.   |
| 4-model seed/ff ensemble (Step 6, v25)  | -0.1           | +1.5       | Marginal. 12-month refit hurts live.|
| Runway configuration cat (Step 7)       | +7.6           | -          | High cardinality overfits.          |
| Fallback-and-24h classifier (Step 3)    | -40            | +195       | Classifier does not generalize to 2026. |
| Step A: LIRF null-flt band rule (v16)   | -83.6          | **-58**    | Deterministic lookup, big win.          |
| Step A variant, gate 3600 (extended)    | +9.8           | -          | Normal rows in band get wrongly boosted; rejected. |
| Step C: LIRF+EGLL detector soft mix (v17)| -8            | +5         | Classifier overfits 2025 hold-out.      |
| Section 6.3: LIRF no-flight-record group  | -2.3          | **-1.8**   | Shrink 0.6 on detector; mix vs group-mean, not v21. |
| Section 6.2 R_norm alone                  | LIRF clean -247 | in v19   | R_norm removes v21's bimodal bias at LIRF.          |
| v20 combined per-airport winner (v19 sub) | -2.25         | **-11.2** | v18 on LIRF/EHAM, R_norm+detector on EGLL/LEBL/LTFM, R_norm alone elsewhere. |

## Ordered plan status

- [x] Step 1 - Diagnostic. Confirmed 576 s on true metric.
- [x] Step 2 - v21 (unfiltered + OBT + no clip). **Live 430.45. Only real win.**
- [x] Step 3 - Mixture. Tested v15 live 625.34. Broken.
- [x] Step 4 - ADES arrival context. Rolled into v25. No live gain.
- [x] Step 5 - OPDI stationarity. Rolled into v25. No live gain.
- [x] Step 6 - 12-month refit + ensemble. v14 live 431.96. No live gain.
- [x] Step 7 - Runway config. Hurt hold-out. Dropped.
- [x] Step A - LIRF band lookup on v21. Hold-out full 455.24 -> 371.67. **Live v16 372.40, -58 s vs v13.**
- [x] Step C - Fallback detector LIRF+EGLL. Hold-out 371.67 -> 363.52. Live v17 377.47 (+5 s worse than v16). Detector overfits again; rejected.
- [x] Section 6.3 - LIRF no-flight-record group (3600 < sd <= 14400). Detector on flight-number prefix, stand prefix, aircraft, runway, hour, sd. Mix p*sd + (1-p)*1150. Shrink 0.6. Hold-out 371.67 -> 369.33. **Live v18 370.63 (-1.77 s vs v16).**
- [x] Section 6.4 - Taxi-in drift meter. 2025 hold-out 202.40 -> 2026 214.96 (+6.21 %). Drift confirmed. Per airport: EHAM +128.9, LIRF +28.9, LFPG +24.1; EGLL -40.9, EDDF -13.3, LEBL -27.6.
- [x] v41 blueprint (thirteenth pass, MODEL_ANALYSIS 4.3) - `R_norm_LIRF` members with the 13 tempo columns (`train_r_norm_lirf_v41.py`, LIRF clean -255 MSE paired) and no 4,431 s cap (-187 MSE paired, first isolated price of the v29 cap). Gate refit with the columns rejected (+94). v42 base with the forward order set aside (-93). `predict_v41.py` serves through `predict_v30.main(r_norm_features=..., r_norm_clip=None)`. Gates: LIRF rows only differ, mean shift -5 s. **Live 299.314 s, -0.65 s vs v40**. Five slots used on 2026-09-13.
- [x] v40 blueprint (twelfth audit 4.2) - v33 stack with the 13 tempo, order, stand-gap and queue columns on the base. `src/train_r_all_v40.py` trains the v26 recipe paired against a control on the cached 97-column frame (`models/v36_tune_cache.parquet`, ids in `v36_tune_cache.ids.parquet`); the control reproduces the shipped v26 members exactly (iters 1275, 890, 1240; FULL 392.33, CLEAN 260.13). v40: CLEAN 257.46 (**-2.67 s**), FULL 393.01 (+0.68, all at LIRF where the head serves), clean class -906 MSE outside LIRF, no airport worse by 500 MSE, iters 1070, 773, 1099. Price about -1.3 s live. `src/predict_v40.py` serves it through `predict_v30.main(extra_columns=...)`. Gates 3 to 5 passed: parity 0.0 s on 344,841 rows; serve clean (0 LIRF rows differ, 29 zero rows, 101 over 7,200); 2026 mean shift per airport within -15.1 (LFPG) to +3.8 s (EGLL). Uploaded 2026-09-13: **live 299.965 s, -1.90 s vs v33**. Superseded by v41 the same day.
- [x] v33 end-to-end hold-out (twelfth audit P1) - `src/eval_v33_holdout.py`, `models/v33.holdout.json`: FULL **321.74**, CLEAN 253.29; class MSE clean 61,554, fallback 7,746, tail 12,654, 24-h 21,464 (LFPG 20,191 undetectable, LIRF 1,273). v39 on the same rows: 338.06 / 264.26. The repo had no such number before v39.
- [x] v34 blueprint (eighth audit Item 2) - Refit hold-out-leaking statistics on months {2..12} \ {7}. Files: `build_lirf_band_table_v34.py`, `train_lirf_regime_v34.py`, `predict_v34.py`. Artefacts: `lirf_band_table_v34.json`, `v34_constants.json` (NORMAL_MEAN_LIRF=1094.41, R_NORM_CLIP=4431.49), `lgbm_p_fb_lirf_v34.txt` (best iter 245, stop AUC 0.857), `lirf_regime_v34.rate_maps.pkl` (base_rate=0.2062 on fit months). Base R_all_v26 and R_norm_lirf_s{42..46} unchanged — their training and encoders already exclude {1, 7}. Diff v34 vs v33: LIRF-only, 25,204 rows, mean -1.34 s, mean \|diff\| 22.0 s, max 2,633 s. **Live 302.07 (+0.21 s vs v33)** — inside the 2-s noise band, consistent with the audit's "score-neutral in expectation" call for a hygiene ship. v33 remains team best.
- [x] v36 blueprint (tenth audit measure 1) - F7 zero-clip repair. `predict_v30.main` gains `per_member_base_clip` (default True) and `fill_zero_rows_per_airport` (default False); defaults keep v33 bit-parity. `predict_v36.py` sets both to False/True: base ensemble averages raw predictions (removes per-member `np.clip(x, 0, None)`); then any row at exactly 0 gets the median of positive predictions at its own airport. Diff vs v33: 110 rows, mean 0.06 s, mean \|diff\| 0.13 s. 44 zero rows filled (v33 shipped 23 zeros; removing per-member clip pulled 21 more rows to exactly 0, all caught by the fill). **Live 302.05 (+0.18 s vs v33, -0.02 s vs v34).** Priced ~252 MSE gain did not materialise on live: the per-member-clip removal moved 66 non-zero rows too (max |d| 3,802 s at EHAM), and their collateral cost cancelled the zero-fill benefit. v33 remains team best.
- [x] v35 blueprint (ninth audit section 7.5) - v34's honestly trained p_fb + v33's all-month scoring artefacts. Refactored `predict_v30.main` to take p_fb kwargs (backward-compatible; parity rebuild of v33 = 0.0000 s). `predict_v35.py` swaps in `lgbm_p_fb_lirf_v34.txt`, `lirf_regime_v34.features.txt`, `lirf_regime_v34.isotonic.pkl`; leaves rate maps, band table and constants at v33 defaults. Diff v35 vs v33: LIRF only, 25,202 rows, mean -0.11 s, mean \|diff\| 1.62 s. Diff v35 vs v34: 5,898 rows, isolates the scoring-artefact revert. **Live 302.09 (+0.22 s vs v33, +0.02 s vs v34).** The audit's decomposition was wrong: reverting the scoring artefacts recovered essentially nothing. The honestly-trained classifier is what drove the +0.21 s regression, not the fit-month lookups. v33 remains team best.
- [x] v33 blueprint (seventh audit) - `predict_v33.py` calls `predict_v30.main` with the same 5-member R_norm files as v32 but keeps the 3-seed base default. No new training. Isolates the priced-cheap R_norm gain (post-mixture A_5 = 74 MSE, scaled to all rows) from the priced-risky base expansion (A_7 = 5,417 MSE on 2026, 8× its hold-out estimate; live behaviour showed base-change bias dominates). Diff v33 vs v30: LIRF only (26,899 rows, mean |diff| 18.8 s, max 3,235 s); non-LIRF identical. Diff v33 vs v32: non-LIRF only. Live **301.87 (-0.11 s vs v30)**, exact match to the priced 301.86 s (off by 0.01 s). Ambiguity framework validated.
- [x] v32 blueprint (sixth audit) - 5-member `R_norm_LIRF` (seeds 42-46, deployed recipe reproduced exactly: best iters 251, 417, 294, 294, 297) + 7-member base R_all_v26 (seeds 42-48, seed 47 needed one retry because multi-threaded linear_tree LGBM was non-deterministic; final iter 1249). Repo repair: restored `models/v26_pre_v31/` boosters to `models/`, moved v31 arrival-cleaning retrain to `models/v31_arrclean/`, reverted `features_congestion_v2.py` leftover line, removed env-var/record rule from `predict_v30.py`, deleted `predict_v31.py`. Acceptance A: `predict_v30.py` default reproduces v30 to 0.0000 s on all 344,841 rows. Ambiguity on 2026 ranking: raw A_5 = 3,475 MSE (LIRF only), post-mixture-and-scaled-to-all-rows A_5 = 74.22 MSE (matches audit's 74 prediction), A_7 = 5,417 MSE (much higher than the audit's hold-out estimate of ~690). Predicted v32 RMSE 299.86 s; **live 302.11 (+0.13 s regression)**. Lottery draw lost — audit priced the spread at ~268 MSE (~0.44 s), so this outcome sits comfortably inside the noise band. v30 remains team best.
- [x] v30 blueprint - LIRF-only encoders for LIRF models (fixes defect where all-airport encoders were served to LIRF-trained models) + new Step A band table `lirf_band_table_v30.json` with mutually exclusive classes (NOSOS431 row moves from 121,410 s to sd). Live **301.98 (-1.28 s vs v29)**.
- [x] v29 blueprint (doc 4th audit + correction) - v26 3-seed base + R_norm_LIRF clip at 4,431 s + ITY340 constant formula. Step A base-term clip REMOVED (audit showed it regressed +124 MSE because deployed code feeds the LIRF mixture into the normal term, not the raw base — the doc's Section 3 measured on the wrong quantity). Live **303.26 (-0.45 s vs v26)**, matches predicted 303.3.
- [x] v27 blueprint - Step 3 (7-seed ensemble with feature_fraction_seed varied) + Step 2 four free fixes (ITY340 formula, R_norm_LIRF clip, drop 3 dead features). Hold-out CLEAN -1.52s vs v26. Live 304.03 (+0.32 s). Regressed.
- [x] v28 blueprint - v26 pipeline + 7-seed R_all_v27 (isolates ensemble effect). Live 304.48 (+0.77 s). Ensemble expansion didn't help live; likely `ades_arr_atfm_delay_today` drop hurt.
- [x] v26 blueprint - Step 5: drop 38 ec_* + 18 opdi_* features and retrain R_all (3 seeds, honest stop). Live **303.71 (-13.14 s vs v24)**.
    - Hold-out full 392.33 (v24 was 396.75, -4s). Doc predicted only -1.9s from ec_* fix but LIVE gain was -13s!
    - Root cause confirmed: ec_* features have 0% coverage in July 2026 (55% of scoring set); model routed all July rows to NaN branch, poisoning predictions.
    - v25 (impute with 2025 medians) failed catastrophically at 375.33 (+58s worse); proves ec_* are actively harmful, not just missing.
    - opdi_* dropped too: dead weight (+0.09 CLEAN removal cost), and coverage collapses at 5 airports in 2026.
- [x] v24 blueprint - Step 4: 3-seed R_all with honest 12% random stop split. Live **316.85 (-0.65 s vs v23)**.
    - Hold-out full 396.75 / clean 266.23 (v21 was 425.92 / 274.08). Cumulative -29 full, -7.85 clean vs v21.
    - Per-seed hold-out clean: 271.58, 268.48, 276.56 (mean 266.23). Doc predicted -3.3s CLEAN; measured -7.85s.
    - Each seed runs to 1200-1600 iterations vs 372 for Nov+Dec stop (4x longer as doc predicted).
- [x] v23 blueprint - Step 2 fallback-rate encodings on p_fb (7 keys × 2 = 14 features, OOF leave-one-month-out, K=30 smoothing). Live **317.50 (-2.8 s vs v22)**.
    - fbrate_flt_prefix rank 1/167 (top feature), fbrate_op rank 7, fbrate_destination rank 16.
    - Stop AUC 0.858 (v22: 0.857), logloss 0.369 (v22: 0.371).
    - LIRF CLEAN hold-out 465.88 vs v22 485.52 = -20s on clean subset.
- [x] v22 blueprint - LIRF regime head + refined Step A + retire 6.3 rule. Live **320.31 (-9.8 s vs v21)**.
    - `R_norm_LIRF`: LightGBM linear_tree on LIRF genuine rows (|y-sd|>=60 & y<80000), 127 leaves, best iter 353.
    - `p_fb_LIRF`: LightGBM binary classifier for |y-sd|<60 on all LIRF rows, selected on binary_logloss (AUC 0.857).
    - Isotonic calibration fitted on Nov+Dec stop set.
    - Mixture: `p_fb_cal * sd + (1 - p_fb_cal) * R_norm_LIRF` at LIRF (26,899 rows).
    - Step A refined band table (11 bins per doc appendix); ITY340 rule now uses R_norm_LIRF instead of R_all (bounded pred 94k vs v21's 112k).
    - Section 6.3 rule and 0.6 shrink RETIRED.
- [x] v21 blueprint - linear_tree + signed-log copies + Step 1 defect fixes + Step 7 temperature. Live **330.12 (-15.6 s vs v20)**.
    - R_all v21 hold-out full 425.92 (v20 was 456.40, -30 s). Clean 274.08 (v20 280.17, -6 s).
    - Fix 3.1: OPDI live-taxi lag of 600 s removes label leak.
    - Fix 3.5: turnaround reads BLOCK_TIME_UTC_mvt directly, not landing+clip(taxi_in).
    - linear_tree + signed-log copies: mvt_eobt1_slog rank 8/153, sched_delay_slog rank 10.
    - Temperature: tmpc rank 26 (helps de-icing gate at cold airports). ice_accretion_1hr dead (all NaN).
    - Per-airport (clean): LIRF -25, LFPG -11, EHAM -6, EGLL -5, EDDM -4.
- [x] v20 blueprint - turnaround features + disruption meters + ITY340 rule. Live **345.70 (-24.9 s vs v18)**.
    - R_all v20 hold-out clean 280.17 (vs v21 283.85, -3.7s). Per-airport: EGLL -19.7, LTFM -11.4, LSZH -8.6.
    - Turnaround features (ground_time, slack_eobt, etc.) rank 13/143 for slack_sched, 18/143 for ground_time.
    - Disruption meters (dep_sd_mean_60m, arr_txi_p90_60m) rank 22/143 for dep_sd_mean.
    - ITY340 rule fired on 1 LIRF ranking row (sd=94560): pred 5,465 -> 77,604 (saves ~6.5G MSE on this row).
- [x] Section 6.2 - R_norm regressor + per-airport detectors + v19 combined architecture. Live **359.46 (-11.2 s vs v18)**.
    - R_norm (v21 recipe on |y-sd|>=60 & y<80000) clean RMSE: **LIRF 565.10 -> 318.08 (-247 s)**, LSZH -21.7, LEBL -21.2, others ~unchanged.
    - Per-airport fallback detectors (honest split Feb-Jun+Aug-Oct / Nov+Dec stop / Jan+Jul once). Qualifying (300+ positives): LIRF (AUC 0.887), LEBL (0.880), EGLL (0.862), LTFM (0.853). All pass doc's decile-overshoot < 0.1 check.
    - v20 combined per-airport winner selection (hold-out full 367.08, -2.25 vs v18):
        - LIRF: v18 (Step A + 6.3 rule)
        - EHAM: v18
        - EGLL / LEBL / LTFM: R_norm + detector soft mix
        - EDDF / EDDM / LEMD / LFPG / LSZH: R_norm alone
    - Naive v19 mixture (uniform mixture everywhere) FAILED at LIRF (+5.3G MSE) because R_norm underestimates on high-sd rows and soft mix pulls all rows toward sd.

## Feature ranks in v21 (proof the analysis was right)

```
mvt_eobt1     rank   1   ← highest gain of all 130 features
sched_delay   rank   3
mvt_iobt      rank   5
sd_mod_86400  rank   6
eobt1_iobt    rank  10
eobt1_sched   rank  12
```

## Submission slots

The limit is 5 uploads per UTC day. 2026-09-12 used 4 slots (v34 to v37).
2026-09-13 used 3 slots (v2a int32 — no result; v2b float64 with letter
suffix — no result; v39 float64 numeric — 352.19). 2026-09-16 used 2 slots (v44 and v45, both scores pending; the scorer was
unusually slow overnight). 2026-09-17 used 2 slots (v46 and v47, both scores pending).
One slot remains today.
Naming lesson: the scorer needs `kind-mango_v<int>.parquet`, float64 dtype.

## Next ideas

The fourteenth pass of `docs/MODEL_ANALYSIS.md` (2026-09-13) is the plan
for a model under 290 s: nine priced levers, a build programme with gates,
the data-source table and a stop rule. Earlier passes stay at `b193868`.

Session 2026-09-16 paired-gate log (base v44 = v41 + L3.a):

- L5 `R_norm_LIRF` on `FB_TOL = 5`: failed (-2,421 MSE served clean).
- L3.a p25/p75 of nb `mvt_eobt1` at apt30/apt60/rwy30: **passed** +552 MSE.
  Shipped as v44, score pending.
- L3.b p25/p75 of nb `mvt_iobt`: failed (-16 MSE); redundant with EOBT_1.
- L3.c neighbour EOBT by (ADEP, stand_prefix), 30 min: failed (-77 MSE);
  captured by the stand categorical and rwy30 tempo.
- L3.d neighbour EOBT by (ADEP, operator), 120 min: failed (-54 MSE);
  captured by the 16 operator-historic encoders already in the base.
- L4 constant-leaf booster meaned 0.7/0.3 with v44: failed (-64 MSE);
  constant leaves are 2.6 s worse alone and drag the mean.
- L9 plan_taxi_res clipped to +/-3600 s: **passed** +650 MSE. Shipped as
  v45 (v44 + plan_taxi_res), score pending.
- L1 purified retune (`tune_lgbm_v43.py`, 30 Optuna trials, linear_tree
  pinned): **passed** +662 MSE served clean, mostly LSZH -527. Shipped as
  v46 (v45 features with the retuned base), score pending.
- L2 12-month refit (`train_r_all_v47.py`): the plan's final step per
  section 5.2 step 6 and section 7 stop rule (sum of shipped paired prices
  ~1,922 MSE, under the 3,000 threshold). Removes the hold-out. Shipped as
  v47, live 296.57 (-2.74 s vs v41, the dominant live driver).
- L8 calendar (`build_calendar.py`, `train_r_all_v48cal.py`, holidays 0.104
  package): paired vs v45, served clean +144 MSE, gate 200 short by 56.
  Rejected; only 2.6 % of rows hit a public holiday and 28 % a weekend,
  and the row-hour categorical already captures most of the weekend
  signal.
- L7 weather at EOBT (`build_weather_eobt.py`, `train_r_all_v48wx.py`):
  paired vs v45 with tmpc/vis/wind/wx_precip/deicing_gate joined at
  EOBT_1 (in addition to the deployed MVT_TIME join), served clean +293
  MSE, gate 300 short by 7. All four served classes improved (total -367),
  but rejected as written. Practically at the bar; a fresh paired run on
  v47 features could tip it over if it survives the L1 retune and L2
  refit — kept as an open lever for a future session.
- L3.e arrival taxi-in residual (`build_arr_taxi_res.py`,
  `train_r_all_v48arr.py`): median arrival TAXITIME_SEC_mvt at same
  airport in previous 30 min minus 2025 fit-month median, coverage 0.996.
  Paired vs v45, served clean -102 MSE (regression); gate 300 failed.
  Airport medians look sensible (LTFM 772 s, LIRF 592, LEBL 209) but the
  residual does not carry signal for departure taxi that is not already
  in the disruption / congestion / plan_taxi_res columns. Closed.

Every step of the fourteenth-pass §5.2 build programme is now complete:
L1 shipped (v46), L5 rejected, L3.a shipped (v44), L3.b/c/d/e/f rejected
or skipped, L4 rejected, L6 skipped (H4 coverage), L7 rejected (7 MSE
short), L8 rejected (56 MSE short), L9 shipped (v45), L2 shipped (v47).
- L3.f runway-use shares: skipped, duplicates `runway_diversity_prev_30m`
  already in features_advanced.py.
- Open work: L3.e arrival taxi-in residual, L6 OPDI@LSZH tempo (H4 says
  January 2026 coverage is 25 % below 2025; the January gate fails as
  written), L7 weather at EOBT, L8 calendar (`holidays` package), L1
  purified retune (`tune_lgbm_v43.py`, 3 h), L2 12-month refit last.
- `ARVT_1_flt` signal in one new form: `plan_taxi_res` alone, clipped to
  ±3,600 s, without the raw `plan_block` and `arvt1_mvt`. Run fresh gates
  before any upload (MODEL_ANALYSIS section 4).
- Do not ship a change priced under the 268 MSE lottery. Do not replace a
  shipped single model on a label-free price alone (v32, v37).

## Discord confirmations

### 2026-09-08 (espinielli)

- `_mvt` values are airport-reported (~validated by Eurocontrol). `_flt` values come from the Network Manager; no post-ops adjustments.
- `BLOCK_TIME_UTC_mvt` should be the airport's actual off-block. `SCHED_TIME_UTC_mvt` carries the schedule.
- Organiser is not providing any value as ground truth for submissions. Any provided value is fair to use as a proxy (or as itself) if it helps.

### 2026-09-09 (espinielli, three-question thread with romano)

- **BLOCK==SCHED fallback rows ARE in the ranking scoring set** — "the ranking comes from whatever we got from the relevant airports for Jan/Jul 2026". Validates Step A, Section 6.3, and R_norm regime head.
- **Open data allowed if declared in the documentation** — METAR, OPDI, OurAirports, OSM, Eurocontrol NM, VRS all fine.
- No known recurring operational cause for the 630 "neither" rows (BLOCK_TIME precedes AOBT by ~50 min at EGLL/LFPG/LIRF/EHAM). Some may be event-linked, others messy airport/NM data.
- **The decision on how to deal with strange/noisy/messy data is a full part of the challenge and a modelling decision** (espinielli reply to Sam re LIRF null-block anomalies). Directly validates our regime-head + Step A approach over any attempt to reverse-engineer the labels.
- Sensitive/state/military flight exclusions are NOT uniform across airports or hours; the fraction varies month-to-month; total volume is small. Congestion counts carry a small, uneven downward bias.
- **IOBT / EOBT / LOBT semantics** (JohnMar1 question):
  - `IOBT` — initial off-block calculated from the flight plan.
  - `EOBT` — estimated off-block from later messages/updates.
  - `LOBT` — latest calculated off-block value.
  - All three are operational values, no post-ops adjustments. Ethics stance on `LOBT_flt` unchanged — it equals `AOBT_3_flt` on 4.3 % of rows.
- **Feature engineering from earlier movement rows (`MVT_TIME < T`) is permitted** — "your model could take into account what is occupied, what is coming, ...". Validates congestion / disruption / OPDI-live features.

### 2026-09-11 (espinielli)

- **No phase 2 is planned**, though the organiser reserves the right to add one "if reverse engineering the ranking is too easy" — "which we despise". Strong external validation for our permanent exclusion of `AOBT_3_flt` and `LOBT_flt` and for the audit rule against setting row-level predictions from live scores.
- **Trino data access is NOT allowed for the challenge** (Piyush Patil raised; espinielli confirmed). Any Trino-derived feature would disqualify.

### 2026-09-17 to 2026-09-18 (espinielli, fl340, others)

- **EUROCONTROL AIU daily airport files and ADSB.lol (ODbL 1.0) count as open
  data for the challenge if declared with attribution**, despite
  EUROCONTROL's non-commercial-use disclaimer and ADSB.lol not being OSN.
  espinielli: "can be considered open data for the challenge." Does not
  extend to OSN state vectors (ground truth there is airport-reported).
- **X1/MX1 resolved.** fl340 raised the same leak concern the audit's X1
  finding names: `AOBT_3_flt`/`MVT_TIME_UTC_mvt` sit in the ranking set and
  their difference nearly reveals the target, and ADS-B trajectory archives
  (adsb.lol) can back out off-block time directly at well-covered airports
  (~40 % of EDDM flights have a first observation near the gate). espinielli:
  "the model is for post-ops, not for tactical use" and "there are no such
  restrictions: if you can find open trajectory data with good ground
  coverage and extract off-block times, we are ok. Practically speaking it
  won't be possible" (poor coverage at LFPG, LIRF, LEMD, LFTM named as the
  reason it is impractical). No formal rule bans either input; the organiser
  is relying on the coverage gap, not a prohibition. We keep our stricter
  exclusion — see `README.md` Ethics section and `docs/MODEL_ANALYSIS.md`
  section 4.9 (X1) / 5.8 (MX1).
- **Training-data corruption may also affect the ranking set.** lbragado
  reported corrupted rows in training data; isaacoluwafemiog said prior
  organiser replies imply the ranking set likely has the same corrupt
  instances. We have not audited our own pipeline for this yet.
- **Domain feature ideas from team GREKI [VTSM]** (their numbers, unverified
  on our data): wake-turbulence take-off sequencing on the same runway
  (medians 113/148/182/241 s by ICAO wake-class pair) priced at only +0.25 s
  live for them, explaining 15 % of raw taxi-out variance but 0.18 % of
  their residual — mostly redundant with congestion + WTC features.
  Target-encoding the (airport, stand, runway) triple on median historical
  taxi time substitutes for gate-to-runway distance without GIS data.
- **Team correspondents share MinIO access keys with teammates**, or create
  dedicated ones; MinIO CLI is the recommended client (espinielli).

## Prize window

Closes **2026-10-31 23:59 CET**. GPLv3 licence + open data only.
