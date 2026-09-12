# Model audit — tenth pass. Source of truth for the next build

Updated 2026-09-12 for the v34/v35 debrief and a full re-read of the shipped
pipeline. Status: live best **301.87 s (v33)**, rank 44 of 111 teams,
**1.87 s from breaking the 300 s barrier**. v34 scored 302.072 (+0.202 s)
and v35 scored 302.0897 (+0.017 s vs v34, +0.219 s vs v33). Both rejected.

This audit rereads the shipped pipeline end to end: training scripts, the
feature modules, the prediction scripts, the models on disk, and the v33
submission row by row. It corrects the ninth-audit findings where the new
evidence requires it, and it prices two new defects measured on the 2026
ranking set. Section 3 is the ordered bullet list of measures for the next
model. No code lives here.

## 1. What ships today (v33), exactly

The submission applies four components row-wise. `src/predict_v30.py` owns
the pipeline. `src/predict_v33.py` swaps one component of it.

1. **Base regressor.** A 3-member mean of LightGBM (`linear_tree=True`)
   `lgbm_r_all_v26_s{42,43,44}`. About 97 features. Trained by
   `src/train_r_all_v26.py` on the 10 target airports. Target filter `y > 0`.
   Stop set: fixed 12 % random mask (`default_rng(1234)`). Hold-out
   months {1, 7} excluded from training.
2. **LIRF regime head.** On every LIRF row:
   `p_fb_cal * sd + (1 - p_fb_cal) * min(R_norm, 4431)`. `R_norm_LIRF` is a
   5-member mean (seeds 42-46). `p_fb` is a binary classifier for
   `|y - sd| < 60` with 14 fallback-rate features, isotonic-calibrated
   (`src/train_lirf_regime_v23.py`).
3. **Step A band table** (`models/lirf_band_table_v30.json`). LIRF rows with
   null flight record and `sd > 14,400` take a deterministic
   `p_fb * sd + p_24h * (86400 + extra) + p_norm * base` lookup. 43 ranking
   rows covered.
4. **ITY340 rule.** LIRF, `sd > 70,000`, outside the Step A cell:
   `(5/6) * (86400 + 1150) + (1/6) * sd`. Fires on about 1 ranking row.
   Final clip at 0. NaN filled with the prediction median.

Feature families from open data: METAR weather, congestion windows,
Eurocontrol daily ATFM, operator target encodings, OSM stand-runway
geometry, OSM taxiway paths, OPDI event counts, OPDI live taxi tempo,
turnaround links, 60-minute disruption meters.

## 2. Findings — flaws that explain the remaining RMSE

Each finding carries the evidence, the mechanism, and an uncertainty note
where the audit cannot price the damage offline.

### 2.1 Data integrity

**F1. Fake fallback labels contaminate base training at 9 airports.**
The `y == sd` rows enter the base as genuine labels at every airport except
LIRF. Measured on the 2025 hold-out months {1, 7}: `|y - sd| < 1 s` at
0.08-0.49 % at the 9 non-LIRF airports, and `|y - sd| < 60 s` at 5.5-10.2 %
there. LIRF measures 1.06 % / 15.32 % in the same months. Only LIRF has a
regime gate. The base learns a conditional mean pulled toward `sd` for the
class it should separate. Uncertainty note: the exact-imputed class (`< 1 s`)
is small; the near-punctual class (`< 60 s`) is mostly genuine. The priced
ceiling of a perfect `< 1 s` gate is 0.35 s (ninth audit). The `< 60 s`
definition at 9 airports is unpriced.

**F2. Hold-out leakage is now a policy, not a defect.**
Ninth-audit section 7 settled the class of the leak: the classifier training
pool excludes {1, 7}; the scoring maps, the band table, and the numeric
constants keep all 12 months. Verify every artefact against this policy at
manifest time. Closed.

**F3. NaN-to-zero injection biases two feature families.**
- `src/features_congestion_v2.py` line 86: `arr_taxi.fillna(0)`. The
  60-minute taxi-in mean includes synthetic zeros. Systematic downward bias.
- `src/features_disruption.py` line 74:
  `(mvt_ts - sched_ts).clip(-1800, 14400).fillna(0)`. Missing schedule times
  read as on-schedule.

**F4. Two parallel frames with a positional alignment assert.**
`predict_v30.py` applies the base encoders on `dep` and the LIRF encoders on
`dep_lirf` (lines 125-133), joined by an assert on `MVT_ID` order. The v29
defect shipped live through this pattern. Duplicate congestion v1 and v2 run
in the same pipeline (lines 95-104). Structural defect class.

**F5. The label filter is nearly empty, by design.**
`y > 0` drops 0.023 % of rows. Zero labels are 0.001 %. The third-audit
verdict stands: the residual floor of about 1,192 MSE is a modelling gap,
not a data-cleaning gap. Closed.

**F6. No coverage monitor between train and ranking.**
`ec_*` had 0 % coverage in July 2026 (55 % of the scoring set). `opdi_*`
collapsed at 5 airports. Both shipped for months until v26 removed them. The
v25 imputation attempt perturbed the learned NaN routing and failed by
+58 s. The predict script still prints no coverage report.

**F7. The clip at 0 ships 23 ranking rows predicted as exactly 0 s.
[NEW, tenth pass]** `predict_v30.py` clips each base member at 0 before
averaging (line 141). On the v33 submission, 23 rows are exactly 0 s:
18 at LSZH, 3 at LTFM, 2 at LEBL. All three members produced 0 or negative
raw values on those rows. If the true values sit near 1,000 s, this class
carries about 252 MSE of the live 91,192. The clip destroys the calibrated
floor of the ensemble. Fix path: clip after averaging, or floor by a
positive quantile, or shift the ensemble rather than truncate it.

**v36 outcome:** paired fix (remove per-member clip + per-airport fill of
residual zeros) shipped. Removing the per-member clip pulled 66 extra
non-zero rows away from v33 (max |d| 3,802 s at EHAM); the zero-fill caught
44 rows (23 original v33 zeros plus 21 new zeros produced by the raw
averaging). Live 302.05 (+0.18 s vs v33, -0.02 s vs v34): priced 252 MSE
gain did not materialise. Lesson: the two fixes must be split. Option 3
alone (fill only the 23 v33 zero rows, keep per-member clip) is the
narrowest test and remains untried.

**F8. Extreme-sd rows at non-LIRF airports have no specialist guard.
[NEW, tenth pass]** The ranking set carries 54 rows with `sd > 70,000`;
51 of them are outside LIRF (EHAM 24, EDDF 20, LFPG 4, EDDM 3). The 2025
corpus holds 251 such rows at the 9 non-LIRF airports. 100 % of them are
genuine: no fallback, no 24h offset. Mean `y = 1,130 s`. The `sd` baseline
costs 100,516 s RMSE on them. ITY340 guards only the 3 LIRF rows. The base
alone must compress a 70,000-170,000 s schedule-delay feature to about
1,130 s. Candidate classes below. Uncertainty note: unpriced offline, high
class risk per the closed-door list.

### 2.2 Machine-learning methods

**M1. Hyperparameters descend from the censored pipeline, selected on the
hold-out.** `src/tune_lgbm.py` filters `y` to `[30, 7200]`, clips
`sched_delay` to `[-1800, 3600]`, and uses months {1, 7} for early stopping
and trial selection (lines 67-68, 78-79, 119-131). `BEST_PARAMS` inherits
that pipeline everywhere. `num_leaves` was halved from 440 to 220 by hand.
`linear_lambda = 1.0` was never searched. Optuna selected on the same months
that later evaluated the model, which is selection leakage. Biggest open
method lever.

**M2. Family members early-stop on the hold-out.** `src/train_lgbm_v21.py`
sets `dvalid` to months {1, 7} (lines 126-152). The v21 family numbers are
optimistic. v24 fixed the base with the random 12 % mask. The LIRF
auxiliaries and the tuner keep the old pattern.

**M3. The isotonic calibrator is fitted on the early-stop set.**
`train_lirf_regime.py` line 169 and `train_lirf_regime_v23.py` line 198 fit
the calibrator on the same Nov-Dec rows that early-stopped the classifier.
Calibration error is measured on the fit rows.

**M4. The Step A table ships hard 0/1 probabilities from tiny counts.**
`models/lirf_band_table_v30.json` carries 11 bands from 1 to 27 rows,
hand-chosen edges and a hand-chosen gate at `sd > 14,400`. Bands with n = 1
declare `p_fb = 1` or `p_24h = 1`. At its variance limit.

**M5. A post-hoc cap re-introduced the worst defect class.**
`R_NORM_CLIP = 4431` clips the `R_norm` mean after averaging
(`predict_v30.py` line 156). The v21 audit priced a cap as the single
largest defect (130 s of live score). The 4,431 s cap was priced at v29. It
is still the same mechanism.

**M6. Single-observation constants carry unbounded row leverage.**
ITY340 constants `P24_ITY = 5/6`, threshold `70,000`, `86400 + 1150` come
from one training row. One ranking row equals about 18 s of full RMSE in
expectation.

**M7. Ensemble variance is 8x the hold-out estimate, and the pricing
formula is inexact.** Seventh audit measured `A_7 = 5,417` MSE on the
2026 ranking vs ~690 hold-out. Multi-threaded `linear_tree` training is not
bit-deterministic (v32 seed 47 needed a retry). `price_ensemble.py` uses
`gain = A * 4/18`; the variance-of-mean factor from 3 to 7 members is
`4/21`. The v32 base gain was overpriced by about 172 MSE on that formula
alone. Uniform member weighting; earlier NNLS collapsed to one member.

**M8. The largest structured residual is the LIRF genuine-row gap.**
Clean rows: model vs `sd`-baseline ratio 0.92 at LIRF, 0.55-0.65 at the
other 9. Gap about 13,700 MSE (sixth audit). The LIRF head reads the base
feature stack; only the fallback-rate encodings are LIRF-specific.

**M9. Feature drift between 2025 and 2026 is unmeasured.**
The sixth-audit taxi-in drift meter found 2026 gaps per airport: EHAM
+128.9, LIRF +28.9, LFPG +24.1, EGLL -40.9, EDDF -13.3, LEBL -27.6. The
model has no shift monitor. v14 12-month refits failed live (+1.5 s).

**M10. The mixture head exists only at LIRF.** Nine airports carry no
regime gate (see F1). Detectors at EGLL/LEBL/LTFM overfit and were dropped
(RECAP, Step C, +5 s live). The concept is right; the `< 1 s` definition
failed at 9 airports; the `< 60 s` definition worked only at LIRF.

**M11. Pool-shrink "honesty" costs real signal. [NEW, tenth pass]**
v34 excluded months {1, 7} from the classifier OOF pool. Months {1, 7} hold
the only Jan/Jul seasonal regime that matches the ranking window. The
honestly trained classifier lost +0.2 s live against v33. The same
regression carried v35 to a rejection. New rule: shrink only the pool that
trains or measures a supervised model; keep inference-time lookups on all
months.

### 2.3 Mathematical constants and multiplicators

Model decisions as constants. Verdicts: OK (priced), suspect (never priced),
fix (move to config or repair).

| constant | file | value | verdict |
|---|---|---|---|
| `FB_TOL` (fallback gate) | `train_lirf_regime*.py` | 60 s | suspect; `< 1 s` at other airports |
| `P24_ITY` | `predict_v30.py` | 5/6 | suspect; one row |
| `ITY_SD_THRESHOLD` | `predict_v30.py` | 70,000 s | suspect |
| `R_NORM_CLIP` | `predict_v30.py` | 4,431 s | fix; see M5 |
| `NORMAL_MEAN_LIRF` | `predict_v30.py` | 1,150 s | fix; M3-style leak |
| `86400 + mean_24h_extra` | band table builder | per band | OK-ish; full year |
| `GATE_SD` | `build_lirf_band_table_v30.py` | 14,400 s | suspect; hand-chosen |
| `signed_log(x) = sign(x)*log1p(|x|)` | `train_r_all_v21.py` | — | OK |
| `linear_tree` linear-feature subset | `train_r_all_v21.py` | 10 hand-picked | suspect; never searched |
| `linear_lambda` | all R_all | 1.0 | suspect; never searched |
| `K_SMOOTH` | `train_lirf_regime_v23.py` | 30 | OK-ish |
| `MIN_COUNT` | `features_operator.py` | 20 | suspect; finest key cutoff |
| 12 % stop mask, `default_rng(1234)` | `train_r_all_v26.py` | fixed | OK |
| uniform `np.mean` over 3/5 seeds | `predict_v30.py` | — | suspect |
| clip at 0, NaN to median | `predict_v30.py` | — | broken; see F7 |
| `sd_mod_86400` | base feature | — | OK |
| mean_24h_extra default | `predict_v23.apply_stepA_v22` | 1,150 | fix; duplicates a constant |
| gain multiplier `A * 4/18` | `price_ensemble.py` | 0.222 | fix; exact factor is `4/21` |

The two transform multiplicators are structurally sound: the
`p * sd + (1 - p) * model` mixture, and the signed-log copy. The risk sits
in the hard numerics.

### 2.4 Codebase and pipeline integrity

**P1. Twelve feature-pipeline copies.** Every training script duplicates
the feature build and its import list. The v29 encoder defect and the v31
repository damage came from drift between copies.

**P2. Duplicate congestion computed twice.** `predict_v30.py` calls both V1
and V2 congestion (lines 102-104). Only the V2 list survives into `feat_all`.

**P3. No manifest.** `models/` holds about 150 boosters and tune-era caches
with no shipped-file manifest. The v31 event overwrote boosters and was
repaired from `v26_pre_v31/`.

**P4. Stale reproduction guide.** `REPRODUCE.md` describes the v15/v16
ensemble era. The v30/v33 steps are missing.

**P5. One mutable entry point.** `predict_v32.py`, `predict_v33.py`,
`predict_v35.py` mutate kwargs into `predict_v30.py`. Every submission since
v30 depends on the same mutable entry.

## 3. Ordered bullet measures for the next model

Adopt the measures in this order. Ship one change per upload.

1. **Repair the zero-clip defect (F7).** Move the per-member clip after the
   ensemble mean, or floor predictions at a small positive quantile of the
   training label, or fill the exactly-0 rows with the ensemble median of
   the member spread. Verify by ranking-set diff against v33: zero rows must
   disappear.
2. **Build one canonical feature pipeline (P1, P2, F4).** One
   `build_features(dep, ctx)` function imported by every training and
   prediction script. One encoder set per consumer. Assert names, not
   positions. First task: parity reproduction of v33 to 0.0000 s.
3. **Re-tune the hyperparameters on the purified protocol (M1).** Fresh
   Optuna run on the current feature set: `y > 0`, no `sd` clip, blocked
   early-stop months {11, 12}, hold-out months {1, 7} excluded from every
   training decision, `linear_tree` flags inside the search (`num_leaves`,
   `linear_lambda`, linear-feature subset). Freeze the result. Ship only if
   the ranking-set ambiguity price clears the 2-s CLEAN evidence bar.
4. **Remove NaN-to-zero injections (F3).** Compute masked means in
   `features_congestion_v2` and `features_disruption`. Add one coverage flag
   per family. Report the per-feature NaN share at prediction time.
5. **Refit the numeric constants and move them to config (M5, M6, M3).**
   `R_NORM_CLIP`, `ITY_*`, `P24_ITY`, band edges, `GATE_SD`,
   `NORMAL_MEAN_LIRF` move to one JSON. Each value carries the fit months
   and the last priced delta. Re-estimate or drop the 4,431 s cap under a
   priced test.
6. **Route the non-LIRF extreme-sd rows (F8).** 51 ranking rows at
   `sd > 70,000`. 2025 evidence: all genuine, mean `y = 1,130 s`. Prefer a
   learned gate over a hand rule. Ship only with calibrated probabilities
   and a priced dominance check.
7. **Attack the LIRF oracle gap (M8).** 13,700 MSE target. Candidates:
   per-airport regressors, a residual model `y - sd` restricted to genuine
   rows, a gated two-head LIRF model. Gate every ship by label-free
   ambiguity pricing.
8. **Fit the isotonic calibrator on a dedicated split (M3).** Example
   binds: train on months 2-10, early-stop on 11-12, calibrate on a month
   block the classifier did not stop on. Report calibration error on the
   split the calibrator did not see.
9. **Add a shipped-file manifest (P3, F4).** A JSON in `models/` lists the
   files a submission reads with SHA-256. The predict script refuses to run
   on a mismatch. Move tune-era caches out of `models/`.
10. **Monitor feature coverage at prediction time (F6, M9).** Print the
    non-null share of each family on the ranking frame. Compare it to the
    training share. A drop of more than 20 points falls back to a train
    median or drops the feature. Never perturb the learned NaN routing
    without a priced test (the v25 rule).
11. **Correct the ambiguity pricing formula (M7).** Use `gain = A * (1/k_old
    - 1/k_new)` with the exact member counts. Record the iteration count and
    worker seeds so any recipe is bit-reproducible. Ship member changes one
    at a time.
12. **Keep the pooling policy (M11).** Exclude {1, 7} from supervised
    training and measurement pools only. Keep scoring maps, band tables and
    constants on all 12 months. Mark the fit months inside every artefact.
13. **Sync the documentation (P4, P5).** Update `REPRODUCE.md` and
    `README.md` in the same change as the model. One predict entry point per
    version; no mutation of a shared default.

Do not reopen these doors without a priced test: the 9-airport fallback gate
(ceiling 0.35 s, perfect `< 1 s` detector), the Step A clip, the 12-month
refit, seasonal weights, per-airport caps, XGBoost or CatBoost in the base,
`AOBT_3_flt`, `LOBT_flt`, leaderboard-derived row values.

## 4. Verification protocol for the next build

1. Reproduce v33 to 0.0000 s before any pricing.
2. Price every change label-free through the ambiguity formula before any
   upload.
3. Ship one change per upload. Stacked changes regress (v32).
4. The noise band for the hold-out is 2 s of CLEAN RMSE.
5. A hold-out difference under that band is not evidence.
6. The guardrails hold: no `AOBT_3_flt`, no `LOBT_flt`, no leaderboard
   values, no 0/1 probabilities from finite counts.
7. Checkpoint every artefact against the pooling policy of measure 12.

## 5. What the floor is made of

About 28 % of the residual MSE sits in extreme rows with no observable
predictor. About 13,700 MSE is the LIRF genuine-row oracle gap. The rest is
the 9-airport local optimum plus the small, biased classes above. Measures
6 and 7 attack the two biggest buckets. The rest is hygiene for decision
quality, not for the score itself.

## 6. The v34/v35 debrief — the falsified decomposition

v34 refit two pools together: the classifier OOF pool (correctly dropped
months {1, 7}) and the scoring artefacts (wrongly dropped {1, 7}). v35
reverted the scoring artefacts and kept the honest classifier. Live results:
v34 302.072 (+0.202 vs v33); v35 302.0897 (+0.017 vs v34; +0.219 vs v33).
The scoring-artefact revert recovered 0.017 s. The regression sits in the
classifier alone. The ranking months Jan and Jul carry the only seasonal
window that resembles the ranking set. Dropping them from the training pool
deleted the signal the classifier needed at inference. Rule carried forward
as measure 12.

## 7. Decision status of open levers

| lever | ceiling | verdict |
|---|---|---|
| Item 3, hyperparameter retune | untried | **biggest open method lever; first big ship** |
| Item 7, 9-airport fallback gate | 0.35 s perfect `< 1 s` gate | deprioritized |
| Item 6, LIRF oracle | ~13,700 MSE | **biggest score-attacking lever; after Item 3** |
| F8, extreme-sd non-LIRF | unpriced | high ceiling, high class risk |
| F7, zero-clip | ~252 MSE raw count | hygiene, repairs the floor |
| Extreme rows, learned gate | 28 % of residual | attach only with calibrated probabilities |

Sections 1-6 of this audit supersede the ninth pass where they differ. The
closed-door list in section 3 stands.
