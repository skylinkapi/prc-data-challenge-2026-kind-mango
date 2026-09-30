# PRC Data Challenge 2026 — team `kind-mango`

Solo entry. Predicts **taxi-out time in seconds** (`TAXITIME_SEC_mvt`) for
departing flights at 10 major European airports. Metric is RMSE against
airport-reported truth.

- **Challenge home:** https://ansperformance.eu/study/data-challenge/dc2026/
- **This repo:** https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango
- **Deadline:** 11 October 2026, 23:59:59 CET (challenge site source
  `index.qmd`, read on 2026-09-30). Upload the last file before 21:00 UTC on
  that day.
- **Current leaderboard best:** **264.34 s RMSE** (`kind-mango_v89.parquet`,
  2026-09-28). v89 mixes one LightGBM member per airport 50/50 into both
  LightGBM terms of v85. From v73 on, the base reads the NM clocks of the
  scored flight (track B, see the Stance section). Before track B, the best
  was 281.87 s (v67, 2026-09-24). The results table below gives every step
  and its live delta.
- **Plan to the deadline:** `docs/WINNING_PLAN.md` section 0 (2026-09-30).

## License

**GNU GPLv3.** See `LICENSE`. Prize-eligible per the challenge rules: source
public, open-data only, documented reproduction path.

## Stance on `AOBT_3_flt` and `LOBT_flt` (changed 2026-09-26)

From v73 on, the base reads the NM clocks of the scored flight. Up to v72,
no served file read `AOBT_3_flt` or `LOBT_flt` (v39 read `LOBT_flt`; see
WINNING_PLAN 3.1). The team lifted that rule on 2026-09-26 and accepts the
risk of a later organiser ruling.

Facts that support the change:

1. The data page blanks only `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`
   on ranking DEP rows. `AOBT_3_flt` and `LOBT_flt` are listed flight
   columns with no restriction.
2. Organiser rulings on Discord: "any provided value is fair to use as a
   proxy (or as itself) if it helps" (2026-09-08). For off-block times,
   "there are no such restrictions" (2026-09-18, MX1).
3. `AOBT_3_flt` does not reconstruct the label. On 2025 departures,
   `MVT - AOBT_3` has a median absolute error of 175 s against the label,
   21 % of rows lie within 60 s, and the raw clock scores RMSE 376 s. On
   the 584 rows with a label above 7,200 s, the clock median is 1,082 s:
   it does not see long holds.
4. The quoted warning in earlier versions of this section came from the
   team's own brief (`docs/PRC_Data_Challenge_2026_BRIEF.md`), not from an
   organiser page.

Forward windows (changed 2026-09-28): from v85 the base reads the NM
clocks of departures that take off after the row (WINNING_PLAN L8). The
team stance MF6 kept such columns out. The organiser allows features from
other movements ("what is occupied, what is coming", 2026-09-09).

Risk: on 2026-09-11 the organiser said that reverse engineering of the
ranking is a practice "we despise" and kept the right to add a phase 2. A
phase 2 without NM actuals cancels this gain. No submission sets a row
value from a leaderboard score.

## What the model is

The scoring stack of v89 (current best) has a base outside LIRF and a LIRF
head. `src_v3/predict_v67.py` serves it. `REPRODUCE.md` section 6 lists
every artefact and its producer.

**Outside LIRF (317,942 rows):** `0.5 x LightGBM term + 0.5 x CatBoost term`.

| term | plain member, weight 0.5 | anchored member, weight 0.5 |
|---|---|---|
| LightGBM, 131 columns | 0.5 x 3-seed global model `lgbm_r_all_v85` + 0.5 x one model per airport `lgbm_pa_v89` | `clip(MVT - AOBT_3, 0, 7,200)` plus an offset: 0.5 x `lgbm_r_all_v85a` + 0.5 x `lgbm_pa_v89a` |
| CatBoost, 123 columns | `catboost_r_all_v73` | `clip(MVT - AOBT_3, 0, 7,200)` plus the offset of `catboost_r_all_v75a` |

- The 131 LightGBM columns are the 118 columns of v65, the 5 NM clocks of the
  scored flight (`build_nm_clocks_v73.py`) and the 8 forward-window clocks
  (`build_nm_forward_v85.py`). CatBoost reads the first 123 of them.
- Rows without `AOBT_3_flt` get the plain member of each term.
- Where the 3 seeds of a global LightGBM model differ by more than 3,600 s,
  the term reads their median (MS3).
- All LightGBM models use `linear_tree` and the MB5 parameters of
  `models/tune_lgbm_v76.tuned_params.json`. Every base model trains on 12
  months of 2025, without LIRF rows and rows with `y > 80,000` (MB3).

**LIRF (26,899 rows), unchanged since v64:**

| rows | scoring component |
|---|---|
| null flight record, `sd > 14,400` (43 rows) | Step A band table: `p_fb * sd + p_24h * (86,400 + mean_24h_extra) + p_norm * R_norm` |
| `sd > 70,000`, outside Step A (1 row) | ITY340 constant `(5/6) * (86,400 + 1,150) + (1/6) * sd` |
| all other rows | `p_fb * sd + (1 - p_fb) * R_norm` |

`R_norm` is the mean of 5 members (`lgbm_r_norm_lirf_v55_s{42..46}`).
`p_fb` is the v56 gate with the out-of-fold isotonic map of v64. Both read
the LIRF-only encoders in `models/lirf_regime.encoders.pkl`.

**Post-processing, every row:** MS1 caps per airport outside LIRF and MS2
floors at the 2025 clean 0.1 % quantile (`src_v3/postprocess.py`).

## Live leaderboard progression

Only `kind-mango_v*.parquet` uploads are shown. All are scored on the same
344,841-row 2026 Jan+Jul ranking set. RMSE decreases are the improvement.

| upload | live RMSE | change | headline |
|---|---|---|---|
| kind-mango_v1 to v10 | 560-562 | — | Old metric era; label filter hid the tail |
| kind-mango_v13 | 430.45 | -130 | Unfiltered labels + unclipped `sd` + `EOBT_1_flt`, `IOBT_flt` deltas |
| kind-mango_v16 | 372.40 | -58 | + LIRF band-lookup rule for `sd > 14,400` null-flight rows |
| kind-mango_v18 | 370.63 | -1.8 | + Section 6.3 LIRF null-flight `3,600 < sd <= 14,400` classifier |
| kind-mango_v19 | 359.46 | -11.2 | + per-airport R_norm switch on the wins |
| kind-mango_v20 | 345.70 | -13.8 | + turnaround family + disruption meters + ITY340 rule |
| kind-mango_v21 | 330.12 | -15.6 | + linear_tree + signed-log copies + OPDI-live leak fix + temperature |
| kind-mango_v22 | 320.31 | -9.8 | + LIRF regime head (`R_norm_LIRF` + calibrated `p_fb`) + refined Step A band table |
| kind-mango_v23 | 317.50 | -2.8 | + fallback-rate encodings on `p_fb` (7 keys × 2 = 14 features) |
| kind-mango_v24 | 316.85 | -0.65 | + 3-seed base with honest 12 % random stop split |
| kind-mango_v26 | 303.71 | -13.1 | + drop 38 `ec_*` and 18 `opdi_*` features (0 % 2026 coverage) |
| kind-mango_v29 | 303.26 | -0.45 | + `R_norm_LIRF` clip at 4,431 s + ITY340 constant formula |
| kind-mango_v30 | 301.98 | -1.28 | + LIRF-only encoders for LIRF models + Step A band table with exclusive classes |
| kind-mango_v32 | 302.11 | +0.13 | + 7-seed base + 5-member `R_norm_LIRF` — base expansion regressed (see MODEL_ANALYSIS 10) |
| kind-mango_v33 | 301.87 | -0.11 | + 5-member `R_norm_LIRF` only, 3-seed base kept; priced 301.86, live 301.87 |
| kind-mango_v34 to v36 | 302.05-302.09 | +0.18 to +0.22 | hold-out-leak refits and the zero-clip repair; rejected (see RECAP) |
| kind-mango_v37 | 302.52 | +0.65 | + 5-seed `p_fb` mean; priced 301.62, regressed; rejected (see MODEL_ANALYSIS 4.1) |
| **kind-mango_v41** | **299.31** | **-0.65** | v40 + `R_norm_LIRF` members with the 13 columns and no 4,431 s cap (`train_r_norm_lirf_v41.py`); paired -415 MSE; see MODEL_ANALYSIS 4.3 |
| kind-mango_v44 | 299.85 | +0.53 vs v41 | v41 base retrained with 6 new columns: 25th and 75th percentiles of neighbour `mvt_eobt1` at the apt30, apt60 and rwy30 windows (`build_tempo_p2575.py`, `train_r_all_v44.py`). Paired -565 MSE on the base's served rows; gate 300 passed. Uploaded 2026-09-16. |
| kind-mango_v45 | 296.13 | -3.72 vs v44 | v44 base retrained with `plan_taxi_res` (clipped +/-3600 s), the row's plan-block minus its 2025 route median (`build_plan_taxi_res.py`, `train_r_all_v45.py`). Paired -617 MSE on the base's served rows (-650 on clean); gate 300 passed. Uploaded 2026-09-16. |
| kind-mango_v46 | 299.14 | +3.01 vs v45, -0.17 vs v41 | v45 features (117 cols) with the L1 retuned base: 30-trial Optuna sweep on `tune_lgbm_v43.py` picked num_leaves 436, min_data 291, lr 0.018, linear_lambda 0.006 (vs deployed 220 / 76 / 0.023 / 1.0). Paired -1,864 MSE served clean cumulative for the L3.a+L9+L1 stack; only -102 MSE landed live. |
| **kind-mango_v47** | **296.57** | **-2.57** | **L2: 12-month refit of the v46 recipe (`train_r_all_v47.py`).** 3 seeds on ALL months with the retuned params, iterations scaled by 1/0.88 (v46 iters 609/1799/1015 -> 693/2045/1154). Removed the 2025 hold-out. Live -2.57 s vs v46 shows most of the reachable gain came from training on months 1 and 7 directly, not from feature engineering or hyperparameter tuning. |
| **kind-mango_v48** | **294.24** | **-2.33** | **Fifteenth-pass Phase 1: MS1 per-airport upper bounds and MS2 clean-floor post-processing on the v47 parquet (`src_v3/build_v48.py`).** 5 non-LIRF extreme rows capped (biggest EDDM 21,465 -> 7,499 s). MP7 label-free gate passed. Zero training cost. Live -2.33 s (about 1,400 MSE) essentially all from the five MS1 clips - MS1 fold price was 0 MSE because the base doesn't extrapolate on 2025 rows, but does on 2026 out-of-distribution rows. |
| kind-mango_v49 | 294.65 | +0.42 | Fifteenth-pass MD2+MH1 rebuild of the LIRF head without the 56 drifted columns; paired LIRF hold-out -2,405 MSE did not transfer live. Head rebuild lever closed. |
| kind-mango_v50 | 293.82 | -0.42 vs v48 | Fifteenth-pass Phase 1: MS3 member-disagreement median on top of v48. 17 rows moved (EDDF 3, EDDM 4, EHAM 4, LEBL 1, LFPG 5) where the 3 v47 base members disagreed by more than 3,600 s outside LIRF. 15 remain distinct after MS1's cap. MP7 gate passed. |
| **kind-mango_v51** | **288.25** | **-5.99** | **Fifteenth-pass MB3: base retrained on served rows only** (LIRF and y > 80,000 excluded). Biggest single lever of the season. -5.99 s live at zero paired price - the MB3 base avoids the extreme predictions v47/v48 relied on MS1 to clip. |
| kind-mango_v52 | 288.36 | +0.11 vs v51 (noise) | MB3 + MB4 (cyclic doy, local hour, holidays, drop numeric month). MB4 does not help on top of MB3; the numeric month wasn't the defect P2 warned about once L2 gave the base all 12 months. |
| kind-mango_v54 | 324.11 | +35.86 vs v51 (rejected) | MB1 anchored-offset target broke on 2026 delayed rows: pred_y = anchor + small_offset fails when anchor is 10,000+ s but the true taxi is 1,000 s. Lever closed. |
| **kind-mango_v55** | **287.83** | **-0.42 vs v51 (new best, -11.48 s vs v41)** | MB8 12-month refit applied to R_norm_LIRF. Landed at the ~0.4 s ceiling for a LIRF-only change (26k of 344k rows). Confirms the L2/MB3 pattern of broader training exposure transfers to the LIRF head. |
| **kind-mango_v56** | **287.50** | **-0.33 vs v55 (new best, -11.81 s vs v41)** | MB8 12-month refit applied to the p_fb LIRF gate + isotonic. Pattern held at four levels now: L2 base -> MB3 base filter -> MB8 R_norm -> MB8 gate. |
| **kind-mango_v57** | **287.33** | **-0.17 vs v56 (new best, -11.98 s vs v41)** | MB8 for `plan_taxi_res` route medians: recompute on all 12 months (5,883 routes vs v45's 5,619) and retrain the v51 base on the updated frame. Pattern held at five levels now: L2 base -> MB3 filter -> MB8 R_norm -> MB8 gate -> MB8 route medians. |
| kind-mango_v58 | 288.62 | +1.29 vs v57 (rejected) | MB8 for `features_operator` target encoders. Backfires because the encoders have an inherent L1 leak that MB8 exacerbates. Lever closed. Fix is MD4 (leave-one-month-out) which we did not ship. |
| kind-mango_v59 | 287.70 | +0.37 vs v57 (rejected) | MF4 arrival drift features. Signal did not transfer to departure taxi-out; other features (congestion, tempo) already capture the drift. Lever closed. |
| kind-mango_v60 | 289.58 | +2.25 vs v57 (rejected) | MD4 OOF encoders regressed worse than v58's MB8. Alignment between training and serving encoders matters more than leak removal. Both encoder levers now closed; v57's v26 encoders sit at the local optimum. |
| kind-mango_v61 | 288.32 | +0.99 vs v57 (rejected) | WINNING_PLAN L4: the v57 recipe with the v45 base parameters (220 leaves, `linear_lambda` 1.0, `BEST_PARAMS`) in place of the v43 retune; rounds 2,091 / 1,795 / 986. Run: `python -m src_v3.train_v61_base`, then `python -m src_v3.predict_v57 --base-model lgbm_r_all_v61 --pre-ms kind-mango_v61_pre_ms.parquet --out kind-mango_v61.parquet`. On the MB3 rows the retuned parameters are better. |
| kind-mango_v62 | not uploaded | 0 (equals v57) | WINNING_PLAN L13: MS3 member median on the v57 base (`python -m src_v3.predict_v62`). The v57 members never differ by more than 3,600 s outside LIRF (max 2,376 s), so no row moves. |
| **kind-mango_v63** | **287.09** | **-0.24 vs v57 (new best, -12.22 s vs v41)** | WINNING_PLAN L12, finding T1: Step A reads `R_norm` as its normal term, not the mixture, so the schedule weight counts once. 18 LIRF cell rows move, all down (100 to 4,241 s); no retrain. Run: `python -m src_v3.predict_v57 --stepa-normal-rnorm --pre-ms kind-mango_v63_pre_ms.parquet --out kind-mango_v63.parquet`. |
| **kind-mango_v64** | **285.95** | **-1.14 vs v63 (new best, -13.36 s vs v41)** | WINNING_PLAN L10, finding L3: the v56 gate booster with an isotonic map fit on out-of-fold scores of six month-pair fold boosters (`python -m src_v3.train_p_fb_v64`), not on its own in-sample scores. 25,988 LIRF rows outside Step A move, mean +8.7 s. Run: `python -m src_v3.predict_v57 --stepa-normal-rnorm --gate-iso lirf_regime_v64.isotonic.pkl --pre-ms kind-mango_v64_pre_ms.parquet --out kind-mango_v64.parquet`. |
| **kind-mango_v65** | **284.74** | **-1.22 vs v64 (new best, -14.58 s vs v41)** | WINNING_PLAN L9, MF5: new base column `plan_nm_taxi` = filed block time `ARVT_1 - EOBT_1` minus its 12-month median per (ADEP, ADES, aircraft type), clean rows, clip +/-3,600 s (`python -m src_v3.build_plan_nm_taxi_v65`; coverage 0.989 train, 0.961 rank). Base retrained on the v57 recipe with 118 columns (`python -m src_v3.train_v65_base`). Run: `python -m src_v3.predict_v57 --base-model lgbm_r_all_v65 --stepa-normal-rnorm --gate-iso lirf_regime_v64.isotonic.pkl --extra models/plan_nm_taxi_rank_v65.parquet --pre-ms kind-mango_v65_pre_ms.parquet --out kind-mango_v65.parquet`. |
| kind-mango_v66 | 285.07 | +0.33 vs v65 (rejected) | WINNING_PLAN L11, MF1: five METAR columns at EOBT_1 on the v65 base (`python -m src_v3.train_v66_base`; serve with a second `--extra models/weather_eobt_rank.parquet`). The -367 MSE paired price on v45 did not transfer. |
| **kind-mango_v67** | **281.87** | **-2.87 vs v65 (new best, -17.45 s vs v41)** | WINNING_PLAN L3, MB7: CatBoost (depth 8, RMSE, ordered target statistics on the 10 categorical columns; `catboost` dependency for a second learner class) on the v65 columns and MB3 rows, 7,385 rounds on 12 months (`python -m src_v3.catboost_base`). Base outside LIRF = 0.5 LightGBM mean + 0.5 CatBoost. Priced first on the 2025 hold-out (`python -m src_v3.measure_catboost_blend`, then `measure_lgbm_control`). Run: `python -m src_v3.predict_v67 --weight 0.5`. |
| kind-mango_v68 | 281.90 | +0.03 vs v67 (rejected) | CatBoost member blended 50/50 into the LIRF `R_norm` term (`python -m src_v3.catboost_r_norm`, then `python -m src_v3.predict_v67 --weight 0.5 --r-norm-weight 0.5 --out kind-mango_v68.parquet`). Hold-out -435 MSE did not transfer. |
| kind-mango_v71 | 281.85 | -0.015 vs v67 (inside noise, not accepted) | MF2 runway-queue columns (`python -m src_v3.build_runway_queue_v71`, `python -m src_v3.train_v71_base`) on the LightGBM half of the v67 blend. Hold-out -560 MSE at equal rounds did not transfer. |
| kind-mango_v73 | 270.58 | -11.29 vs v67 | WINNING_PLAN L1, track B: five NM clock columns of the scored flight (`mvt_aobt3`, `aobt3_eobt1`, `aobt3_sched`, `mvt_lobt`, `lobt_aobt3`; `python -m src_v3.build_nm_clocks_v73`) on the v65 LightGBM base (123 columns, `python -m src_v3.train_v73_base`). Hold-out first (`python -m src_v3.measure_nm_clocks`): -5,652 MSE against an equal-round control, negative at all 9 airports; neighbour clocks (L2) cost 730 MSE at LFPG and stay out. Base outside LIRF = 0.7 LightGBM (member median where the members disagree by more than 3,600 s, MS3) + 0.3 v67 CatBoost. Run: `python -m src_v3.predict_v67 --weight 0.3 --ms3 --base-model lgbm_r_all_v73 --catboost-tag v67 --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet --pre-ms kind-mango_v73_pre_ms.parquet --out kind-mango_v73.parquet`. |
| kind-mango_v74 | 266.84 | -3.73 vs v73 | v73 plus two members. (1) CatBoost with the five clock columns, 7,385 rounds (`catboost_r_all_v73`). (2) WINNING_PLAN L5: LightGBM on `y - clip(MVT - AOBT_3, 0, 7,200)` with the v73 columns (`python -m src_v3.train_v74_anchor`); it serves `anchor + offset`, and the v73 term where `AOBT_3` is null. Hold-out first (`python -m src_v3.measure_nm_anchor`): 0.5 anchored + 0.5 clock member -818 MSE against the clock member, all 9 airports. Base outside LIRF = 0.5 x (0.5 v73 LightGBM + 0.5 anchored, MS3 on each) + 0.5 CatBoost. Run: `python -m src_v3.catboost_base --tag v73 --rounds-from v67 --extra models/nm_clocks_train_v73.parquet`, then `python -m src_v3.predict_v67 --weight 0.5 --ms3 --base-model lgbm_r_all_v73 --catboost-tag v73 --anchor-model lgbm_r_all_v74a --anchor-weight 0.5 --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet --pre-ms kind-mango_v74_pre_ms.parquet --out kind-mango_v74.parquet`. |
| kind-mango_v75 | 266.15 | -0.70 vs v74 | v74 plus a CatBoost member on `y - clip(MVT - AOBT_3, 0, 7,200)` (v67 recipe with the clock columns, 7,305 rounds on the 1,903,118 rows with `AOBT_3`), blended 0.5 into the CatBoost term as the anchored LightGBM is blended into the LightGBM term. No hold-out price (a CatBoost hold-out fit takes about 2 h). Run: `python -m src_v3.catboost_base --tag v75a --rounds-from v67 --extra models/nm_clocks_train_v73.parquet --anchor-col mvt_aobt3`, then `python -m src_v3.predict_v67 --weight 0.5 --ms3 --base-model lgbm_r_all_v73 --catboost-tag v73 --catboost-anchor-tag v75a --anchor-model lgbm_r_all_v74a --anchor-weight 0.5 --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet --pre-ms kind-mango_v75_pre_ms.parquet --out kind-mango_v75.parquet`. |
| kind-mango_v76 | 265.27 | -0.88 vs v75 | MB5: a 24-trial Optuna study (seeded TPE sampler; v73 columns, MB3 rows, fit months 2 to 6 and 8 to 10, early stop on 11 and 12; trial 0 = v43 parameters) picks `linear_lambda` 15.1 (v43: 0.0055), 613 leaves, learning rate 0.015, 2,284 rounds. Hold-out against the served-style clock member (v43 parameters, 2,000 rounds): -965 MSE, all 9 airports (`python -m src_v3.measure_tune_v76`); inside the v75 blend about -240 MSE for the plain member alone. Both LightGBM bases (plain and anchored) refit with these parameters, 3,377 and 3,340 rounds. Run: `python -m src_v3.tune_lgbm_v76`, `python -m src_v3.train_v76_base plain`, `python -m src_v3.train_v76_base anchored`, then `python -m src_v3.predict_v67 --weight 0.5 --ms3 --base-model lgbm_r_all_v76 --catboost-tag v73 --catboost-anchor-tag v75a --anchor-model lgbm_r_all_v76a --anchor-weight 0.5 --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet --pre-ms kind-mango_v76_pre_ms.parquet --out kind-mango_v76.parquet`. |
| kind-mango_v85 | 264.55 | -0.72 vs v76 | WINNING_PLAN L8, forward windows: mean and count of `MVT - AOBT_3` of the other departures at the airport and on the runway that take off in `(t, t + 15 min]` and `(t, t + 30 min]` (`src_v3/build_nm_forward_v85.py`). Hold-out: the MB5 member -604 MSE, 8 of 9 airports; as the plain LightGBM term of the v76 blend -208 (`python -m src_v3.measure_extra_v81 v85 models/nm_forward_train_v85.parquet`). Both LightGBM bases refit with the 8 columns; CatBoost members as v76. Run: `python -m src_v3.build_nm_forward_v85`, `python -m src_v3.train_v76_base plain --tag v85 --extra models/nm_forward_train_v85.parquet`, `python -m src_v3.train_v76_base anchored --tag v85 --extra models/nm_forward_train_v85.parquet`, then `python -m src_v3.predict_v67 --weight 0.5 --ms3 --base-model lgbm_r_all_v85 --catboost-tag v73 --catboost-anchor-tag v75a --anchor-model lgbm_r_all_v85a --anchor-weight 0.5 --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet models/nm_forward_rank_v85.parquet --pre-ms kind-mango_v85_pre_ms.parquet --out kind-mango_v85.parquet`. |
| kind-mango_v87 | 264.86 | +0.31 vs v85 (rejected) | Forward-window columns in both CatBoost members too (`catboost_base --tag v86c` and `v86ca`); not priced on the hold-out first. |
| **kind-mango_v89** | **264.34** | **-0.22 vs v85 (new best, -34.98 s vs v41)** | One LightGBM member per airport (v85 columns, MB5 parameters, one seed), mixed 50/50 with the global member in the plain and the anchored LightGBM terms. Hold-out (`python -m src_v3.measure_per_airport_v89`): alone +1,112 MSE, the 50/50 mean -657 MSE at all 9 airports, about -150 in the blend for the plain term alone. Run: `python -m src_v3.train_per_airport_v89 plain`, `python -m src_v3.train_per_airport_v89 anchored`, then `python -m src_v3.predict_v67 --weight 0.5 --ms3 --base-model lgbm_r_all_v85 --catboost-tag v73 --catboost-anchor-tag v75a --anchor-model lgbm_r_all_v85a --anchor-weight 0.5 --per-airport-tag v89 --per-airport-weight 0.5 --extra models/plan_nm_taxi_rank_v65.parquet models/nm_clocks_rank_v73.parquet models/nm_forward_rank_v85.parquet --pre-ms kind-mango_v89_pre_ms.parquet --out kind-mango_v89.parquet`. |
| kind-mango_v40 | 299.97 | -1.90 | v33 stack + 13 tempo, order, stand-gap and queue columns on the base (`train_r_all_v40.py`); paired hold-out CLEAN -2.67 s; see MODEL_ANALYSIS 4.2 |
| kind-mango_v39 | 352.19 | +50.32 | twelfth-audit rewrite from a cold start, `src_v2/`; hold-out CLEAN 264 (beat v33's 266) but FULL regressed on live; base lost 37 v33 columns; rejected (see MODEL_ANALYSIS 4.1) |

v38 (base + `ARVT_1_flt` planned-time features) is not in the table. It failed
the deployed-recipe gate, so no upload followed (MODEL_ANALYSIS section 4).

The `RECAP.md` file tracks every attempt, including the failures. The
`docs/MODEL_ANALYSIS.md` file is the fifteenth pass: the audit of v47, the
causes of the remaining error, and the measures for the new model. Earlier
passes stay in git.

## Model card (v21)

This card describes the v21 base. The shipped base `R_all_v26` differs in 3
ways:

- 3 seeds (42-44), early-stopped on a random 12 % of the train rows.
- 97 features. v26 dropped the 38 `ec_*` and 18 `opdi_*` columns.
- Hold-out on Jan+Jul 2025 for the 3-member mean: FULL 392.33, CLEAN 266.46.

The LIRF head still reads 153 columns (MODEL_ANALYSIS F1).

- **Algorithm:** LightGBM regressor, `linear_tree=True`, `linear_lambda=1.0`.
- **Objective:** RMSE.
- **Rows kept:** 2,084,659 departures at the 10 target airports for 2025 (only
  `y > 0`).
- **Split:** train = months 2 to 6 and 8 to 10; early-stop = 11 and 12;
  hold-out = 1 and 7 (Jan+Jul).
- **Features:** 153 total.
  - 10 categoricals: airport, destination, runway, stand, aircraft type, wake,
    market segment, operator, flight rule, flight type.
  - 14 time and schedule: `hour`, `dow`, `month`, `sched_delay`,
    `sd_mod_86400`, `mvt_eobt1`, `mvt_iobt`, `eobt1_sched`, `eobt1_iobt`, the
    signed-log copies of the 5 previous fields, `flt_null`, `flt_id_null`.
  - 18 METAR: crosswind and headwind on the active runway, gust, visibility,
    ceiling, cloud codes, precip / snow / thunder / freezing, and the Step 7
    temperature block (`tmpc`, `dwpc`, `dewpt_spread`, `ice_accretion_1hr`,
    `deicing_gate = wx_precip * (tmpc < 3)`).
  - 13 congestion v2 (ADES-keyed arrival counts, same-runway ARR/DEP counts,
    arrival taxi-in rolling mean, next-10-min queue).
  - 6 turnaround (`ground_time`, `slack_eobt`, `slack_sched`, `in_delay`,
    `taxi_in_prev`, `link_ambiguous`) linking each departure to the last
    arrival on the same stand, using `BLOCK_TIME_UTC_mvt` for the join.
  - 4 disruption meters over a 60-minute window before take-off
    (`dep_sd_mean_60m`, `dep_fnull_60m`, `arr_txi_mean_60m`, `arr_txi_p90_60m`).
  - 18 Eurocontrol daily ATFM per airport.
  - 16 operator historic encoders per (op × airport / runway / hour-bin / stand).
  - Physical: OSM haversine stand→runway, real path length, turn count.
  - 5 advanced physical (scheduled pushback load, runway diversity, secs since
    last DEP on runway, runway-bank intensity, ADES arrival ATFM delay).
  - 9 OPDI raw event counts (entry-runway, entry-taxiway, exit-parking, all in
    prev 15/30/60 min) — leak-fixed by a 600 s lag on the window end.
  - 9 OPDI live-taxi rolling stats (mean / median / count of ACTUAL observed
    taxi times of OTHER completed flights, 30/60/120 min).
- **Hyperparameters (Optuna trial 31, kept, plus `linear_tree`):**
  - `learning_rate` 0.0228, `num_leaves` 220 (halved from 440 for linear
    leaves), `min_data_in_leaf` 76, `feature_fraction` 0.674, `bagging_fraction`
    0.937, `lambda_l1` 0.089, `lambda_l2` 0.342, `min_gain_to_split` 1.83.
  - Best iteration on Nov+Dec early-stop: 372.

- **Hold-out RMSE on Jan+Jul 2025** (single number is misleading — see
  `docs/MODEL_ANALYSIS.md` section 2 on the 1.5 s measurement noise floor):

  | metric | value |
  |---|---|
  | FULL RMSE (all rows) | 425.92 |
  | CLEAN RMSE (`30 <= y <= 7,200`) | 274.08 |
  | per-airport clean (LIRF / EGLL / LFPG / LEMD) | 546.6 / 278.4 / 275.5 / 188.5 |

## Tail rules of v21 (historical)

The v33 rules are in the delegation table above. v22 replaced the rules below
with the LIRF regime head and retired the Section 6.3 classifier. The text
stays for the record. See `src/predict_v21_final.py`.

1. **Step A band table.** LIRF, null flight record, `sd > 14,400`. Prediction:
   `P(fb) * sd + P(24h) * (86,400 + 1,150) + (1 - P(fb) - P(24h)) * v21_base`.
   Bands and probabilities are in the appendix of `docs/MODEL_ANALYSIS.md`.
   Covers 43 rows on the ranking set.
2. **Section 6.3 classifier.** LIRF, null flight record, `3,600 < sd <= 14,400`.
   A LightGBM binary classifier for `|y - sd| < 60` trained on the LIRF-null
   subset with flight-number prefix, stand prefix, aircraft, runway, hour and
   `sd`. Applied as `p * sd + (1 - p) * 1,220` (mean of normal LIRF-null rows).
   The raw probability is shrunk by 0.6 (a conservative de-calibration).
   Covers 305 rows.
3. **ITY340 rule.** LIRF, any flight record, `sd > 70,000`, outside the Step A
   cell. Prediction: `(5/6) * (86,400 + R_all_v21) + (1/6) * R_all_v21`.
   Fires on 1 row per year at LIRF. This row alone is worth about 18 s of
   full RMSE in expectation.

## Data sources (all open)

| Source | Data | Licence |
|---|---|---|
| PRC / OpenSky Network | Movements at 10 EU airports (2025 train, 2026 ranking) | Challenge data licence |
| Iowa State ASOS | METAR reports, hourly, per station | Public domain |
| OurAirports | Runway coordinates, headings, lengths | Public domain |
| OSM Overpass | `aeroway=parking_position`, `taxiway`, `runway` | ODbL |
| Eurocontrol NM | Daily ATC pre-dep delay, ATFM slot adherence, airport traffic | Open (non-commercial-use clause; challenge use confirmed by organiser, 2026-09-17) |
| OPDI (PRC + OSN) | Flight events (entry-runway, entry-taxiway, exit-parking, …) | Open (CC-BY 4.0, confirmed by organiser) |
| VRS StandingData | Aircraft-type metadata | Open |

## Quick start

Python 3.13, one virtualenv. Setup:

```bash
git clone https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango.git && cd prc-data-challenge-2026-kind-mango
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned versions, Python 3.13
```

Then follow `REPRODUCE.md`:

1. Get the organiser data and the open data (sections 2 and 3).
2. Build the frames and serve v89 from the trained artefacts (sections 4 and
   6.1), or retrain every artefact first (sections 5 and 6.2).
3. Check the file against the recorded hash:
   `python -m src_v3.test_parity --tag v89`. The v89 record does not exist
   yet (see Reproduction).
4. Check a candidate against the best file without labels:
   `python src/check_submission_2026.py submission/kind-mango_vNN.parquet --ref submission/kind-mango_v89.parquet`.
5. Upload: `python -m src_v2.cli upload kind-mango_vNN.parquet` (5 uploads per UTC day).

The v33 and v41 pipelines under `src/` stay for the paper trail. Their
commands are in git history (`git show 2a4ec1b:README.md`).

### Rewrite (`src_v2/`, v39, live 352.19)

Cold-start rebuild of the section-4 measures from `docs/MODEL_ANALYSIS.md`.
One evaluation harness (P1), one data layer (A1–A8), neighbour EOBT tempo
(B1), take-off order (B2), stand re-occupation gap (B3), queue between
EOBT_1 and MVT (B4), plan-taxi residual (B5), anchor deltas (B6); 3-seed
constant-leaf base on clean rows (C2, C3, C5); regime heads at seven
airports (C1); null-flight LIRF tail head (D1); post-processing (E1–E3).
Ships as `kind-mango_v39.parquet`. Regressed live by +50 s vs v33. The
debrief in `docs/MODEL_ANALYSIS.md` 4.1 puts most of it on the removed
ITY340 hedge and the rest on the clean-only, cold-start base. The package
stays as the source of the 13 tempo, order, stand-gap and queue columns
that `src/train_r_all_v40.py` grafts onto the v33 base, and as the
cached-frame harness for its own stack. It cannot rebuild v33.

```bash
python -m src_v2.cli build      # cache 2 feature frames (~4 min)
python -m src_v2.tests          # F4 label/coverage assertions
python -m src_v2.cli full kind-mango_v39.parquet   # fit + hold-out + predict
python -m src_v2.cli upload kind-mango_v39.parquet # MinIO upload
```

## Layout

```
src/
  # v21 to v41 pipeline; predict_v30.py and the features_*.py modules still serve v89
  features_weather.py           # METAR + crosswind + Step 7 temperature block
  features_congestion_v2.py     # arrivals keyed on ADES_mvt, same-runway ARR/DEP counts
  features_congestion.py        # v1 keyed on ADEP_mvt (needed by v21_old booster)
  features_eurocontrol.py       # daily ATFM per-airport features
  features_operator.py          # operator historic taxi encoders
  features_taxi_distance.py     # OSM stand->runway haversine
  features_advanced.py          # physical proxies (queue, runway diversity, ades delay)
  features_osm_path.py          # OSM taxiway graph shortest-path
  features_opdi.py              # OPDI rolling event counts
  features_opdi_live.py         # OPDI rolling live-taxi mean/median (600 s lag fix)
  features_turnaround.py        # aircraft-on-stand link, uses BLOCK_TIME_UTC_mvt
  features_disruption.py        # 60-min disruption meters
  train_r_all_v26.py            # base regressor, 3 members
  train_lirf_regime.py          # LIRF feature list and LIRF-only encoders
  train_r_norm_lirf_seeds.py    # 5 R_norm_LIRF members
  train_lirf_regime_v23.py      # LIRF p_fb gate, fallback-rate maps, calibrator
  build_lirf_band_table_v30.py  # Step A band table, exclusive classes
  predict_v30.py                # shared prediction pipeline
  predict_v33.py                # v33 submission
  price_ensemble.py             # label-free ambiguity price on the ranking set
  eval_v33_holdout.py           # v33 stack scored end to end on the 2025 hold-out (P1)
  train_r_all_v40.py            # tempo columns on the v33 base, paired against a control
  predict_v40.py                # v33 stack with the v40 base members
  train_r_norm_lirf_v41.py      # LIRF regressor members with the tempo columns, paired
  train_p_fb_lirf_v41.py        # LIRF gate with the tempo columns, paired (rejected)
  features_order_next.py        # forward take-off order column (v42, set aside)
  train_r_all_v42.py            # forward order on the base, paired (set aside)
  predict_v41.py                # v41 submission

  # fourteenth-pass harness (H1 to H5)
  build_h1_frame_cache.py       # cached frame with the movement id inside (h1_frame_cache.parquet)
  eval_v33_holdout.py           # scores any stack per class per airport on months 1 and 7
  check_submission_2026.py      # label-free 2026 checks against the best file
  check_coverage.py             # per-column coverage 2025 vs Jan/Jul 2026 + OPDI counts
  test_v33_parity.py            # rebuilds v33 to 0.0 s; writes the 2026 feature dump

  # rejected ships v34-v38, kept for the paper trail (see RECAP)
  build_lirf_band_table_v34.py, train_lirf_regime_v34.py,   # v34/v35 fit-month refit
  tune_lgbm_v36.py,                                          # purified tuner
  train_p_fb_lirf_seeds.py, price_p_fb_seeds.py,             # v37 p_fb seed mean
  features_plan.py,                                          # v38 ARVT_1 planned-time features
  predict_v34.py .. predict_v38.py

  # older versions kept for the paper trail
  train_lgbm_v{1..20}.py, train_r_all_v20.py, features_*_v2.py,
  ensemble_*.py, predict_v*.py, mixture_*.py

  # diagnostics used by MODEL_ANALYSIS.md
  eval_full.py, eval_v21_stepA.py, eval_v21_stepAC.py, eval_v20_final.py,
  eval_v19.py, eval_v20_combined.py, train_drift_meter.py,
  train_fallback_detector.py, train_v18_detectors.py

src_v2/                         # v39 cold-start rewrite (live 352.19); builds the frame caches the served path reads
  cli.py                        # build, full and upload commands
  frame.py, features.py         # tempo, order, stand-gap and queue columns (served from v40 on)

src_v3/                         # served path since v48
  predict_v67.py                # serves v67 to v89: CatBoost blend, anchored and per-airport members
  predict_v57.py                # serve_v30 wrapper and the MS1/MS2 writer
  postprocess.py, support.py    # MS1 caps, MS2 floors, MS3 member median, MS4 checks
  build_nm_clocks_v73.py        # NM clocks of the scored flight
  build_nm_forward_v85.py       # forward-window clocks (v85 served; v86 and v88 priced)
  build_plan_nm_taxi_v65.py     # planned-taxi column
  train_v76_base.py             # global LightGBM bases, plain and anchored
  train_per_airport_v89.py      # per-airport LightGBM members
  catboost_base.py              # CatBoost members, plain and anchored
  tune_lgbm_v76.py              # MB5 study of the LightGBM parameters
  measure_*.py                  # hold-out price of each lever (reports in models/v3/)
  test_parity.py                # rebuilds a served file and checks its recorded hash
  config.py                     # constants with their provenance

external/                       # open-data caches (gitignored)
  metar/       Iowa State ASOS CSVs per ICAO
  eurocontrol/ 5 NM performance Excels + consolidated daily parquet
  osm/         OSM stand positions + taxiway graphs
  airports/    OurAirports runways.csv + airports.csv
  vrs/         VRS StandingData aircraft types
  opdi/        OPDI event chunks + consolidated events_all.parquet + taxi_out_v2.parquet
training/                       # organiser-provided 2025 monthly parquets (gitignored)
submission/                     # ranking.parquet, submitting.parquet, kind-mango_v*.parquet
models/                         # trained boosters + encoders + JSON config
docs/                           # PRC brief + MODEL_ANALYSIS.md (progress, fifteenth-pass findings) + WINNING_PLAN.md (plan to 11 October in section 0, gap analysis, ranked levers) + AUDIT_PROMPT.md (sixteenth-pass brief, superseded)
RECAP.md                        # session log for every attempt, submissions and scores
REPRODUCE.md                    # reproduction guide for v67 and v89
requirements.txt                # pinned library versions of the v67 to v89 pipeline
```

## Approaches tried and RULED OUT

For transparency, and so the next replicator does not waste days on the same
dead-ends:

- **Fallback heads outside LIRF (MH2)** — mixing a calibrated P(|y - sd| < 60) toward `sd`
  costs +292 MSE on the 2025 hold-out and is worse at all 9 airports
  (`python -m src_v3.measure_mh2`).
- **Tail rules that pull predicted outliers toward the 2025 label tail** (Discord
  suggestion, 2026-09-23) — +1,639 MSE on the 2025 hold-out for a 50/50 blend
  with the 2025 curve of y on `mvt_eobt1`; a cap on normal rows moves 3 rows
  (`python -m src_v3.measure_tail_rules`).
- **Huber loss and log-target regression** — 408 s hold-out, much worse than MSE.
- **Tighter target range `[30, 21600]`** — hurt RMSE 6+ s.
- **LIRF specialist model** — converged in the same range as the general model.
- **Old tail-mixture (P(y > 3600) classifier + hard swap)** — v22/v27; live went
  from 372 to 625. Hard swaps have unbounded downside.
- **`P_tail` as a regressor feature** — target leakage (in-fold) or distribution
  shift (out-of-fold).
- **Data-quality classifier (BLOCK==SCHED detector)** — AUC 0.51 on the wrong
  target. The right target is `|y - sd| <= 1` per airport, deployed at LIRF only.
- **Semi-supervised drift features from ranking rows** — hurt both hold-out and
  live.
- **Quantile regression (q = 0.5 / 0.55 / 0.60) ensemble** — near-zero effect.
- **Extended OPDI event counts (exit-taxiway, entry-threshold)** — the counts
  collapse between 2025 and 2026 at 5 airports (see MODEL_ANALYSIS 7); use
  the raw counts at EDDF and LSZH only, with the OPDI-live 600 s lag fix.
- **VRS aircraft-type derived features (manufacturer, engines)** — overfit,
  duplicates the aircraft-type categorical.
- **Post-hoc airport-level shift calibration** — near-zero effect.
- **12-month refit + 3-seed seed/ff ensemble (v25)** — NNLS collapsed to one
  member, live +1.5 s inside noise.
- **Runway-configuration string categorical (v26)** — hundreds of levels,
  overfit, +7.6 s full hold-out.
- **Step A extended to `sd > 3,600`** — v21 already holds the conditional mean
  in that band; the extended rule cost 9.8 s on the hold-out.
- **Per-airport fallback detectors at EGLL, LEBL, LTFM (v18/v19 mix)** — with
  the corrected fallback definition (`|y - sd| <= 1` at 9 airports and `< 60`
  only at LIRF) these detectors model punctuality, not reporting failure. The
  R_all regressor already holds that signal.
- **Hourly / stand-occupancy / EOBT-copy detector families** — measured and
  worth 48 to 229 MSE at stake per airport, not worth a component.
- **7-seed base mean (v32)** — priced -1,204 MSE; live +0.13 s. The base
  expansion drove the regression.
- **Fit-month refit of the LIRF head statistics (v34, v35)** — live +0.21 and
  +0.22 s vs v33.
- **Zero-clip repair on the base mean (v36)** — live +0.18 s vs v33.
- **5-seed `p_fb` mean (v37)** — priced -151 MSE; live +0.65 s vs v33.
- **`R_norm_LIRF` retrain without the columns that lose 2026 coverage** —
  priced at 103 MSE, under the 268 MSE lottery; no upload.
- **`ARVT_1_flt` planned-time features in the base, raw form (v38)** — seed 43
  stops at iteration 6; 3-member hold-out CLEAN +10.8 s worse; no upload.
- **Runway configuration or 3-lane submission variants that differ on a handful
  of rows** — the brief forbids learning from the ranking process.

## Reproduction

See [`REPRODUCE.md`](REPRODUCE.md). Tier 1 serves a file from the trained
artefacts; Tier 2 retrains every served artefact. Section 6 covers the best
file (v89). `python -m src_v3.test_parity --tag v67` rebuilds v67 and checks it
against the recorded hash in `submission/kind-mango_v67.parity.json` (bit
parity on 2026-09-26). The v89 hash is not recorded yet: run
`python -m src_v3.test_parity --tag v89 --record` on the machine that holds the
uploaded file, then commit the record.

## Acknowledgements

- Eurocontrol Performance Review Commission (PRC), OpenSky Network — challenge
  data and infrastructure.
- Iowa State University Environmental Mesonet — METAR archive.
- David Megginson — OurAirports open data.
- OpenStreetMap contributors — stand and runway geometry.
- OPDI project (PRC + OSN) — ADS-B derived flight events.
- Virtual Radar Server community — StandingData aircraft-type lookup.
- The idea to use `MVT - AOBT_3_flt` as a taxi-out clock is public in the
  GPLv3 repos of other teams (for example `Phoenix-Ops-LTD/prc2026-taxiout`
  and `ChandanHegde07/OpenAir`). v73 reimplements the idea in this
  pipeline and copies no code.
