# kind-mango recap

Status as of 2026-09-13. Team best **299.31 s** (v41, below 300). 111 teams on
leaderboard. v34 to v37 regressed +0.18 to
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
suffix — no result; v39 float64 numeric — 352.19). Two slots remain.
Naming lesson: the scorer needs `kind-mango_v<int>.parquet`, float64 dtype.

## Next ideas

The twelfth audit (2026-09-13) holds the measures in `docs/MODEL_ANALYSIS.md`
section 4, corrected in 4.1.4 after v39: keep the LIRF `sd > 70,000` hedge,
train the base on all `y > 0` rows, no upper clip under 180,000 s, paired
control for every change. Section 4.2 holds the v40 plan: the v33 stack with
the 13 tempo, order, stand-gap and queue columns grafted onto the base
(`src/train_r_all_v40.py`, paired against a control on the cached
97-column frame). The open item below stays in the list as measure B5:

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

## Prize window

Closes **2026-10-31 23:59 CET**. GPLv3 licence + open data only.
