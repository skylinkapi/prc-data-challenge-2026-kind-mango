# PRC Data Challenge 2026 — team `kind-mango`

Solo entry. Predicts **taxi-out time in seconds** (`TAXITIME_SEC_mvt`) for
departing flights at 10 major European airports. Metric is RMSE against
airport-reported truth.

- **Challenge home:** https://ansperformance.eu/study/data-challenge/dc2026/
- **This repo:** https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango
- **Audit, 2026-10-09:** `docs/MODEL_ANALYSIS.md` twenty-second pass, a coverage audit. The ADS-B and OPDI links used the NM callsign, which no departure without an NM record has; a flight-number fallback with an IATA-to-ICAO airline map from the record rows links 1,712 of them (v127, -0.95 s).
- **Audit, 2026-10-08 evening:** `docs/MODEL_ANALYSIS.md` twenty-first pass. On the user's directive, the live score judges each upload; no hold-out price comes first. Strategy: a neural-forward blend (v123 TabM share 0.65; v124 a fourth TabM family). No measured lever reaches 220 s.
- **Audit, 2026-10-08:** `docs/MODEL_ANALYSIS.md` twentieth pass. The organiser times are whole minutes plus noise of at most 6 s at seven airports. At LFPG a take-off time off that grid marks a long label (rule P1, hold-out -250 MSE, v121 -235 live with the seed change). No measured lever reaches 220 s.
- **Audit, 2026-10-07:** `docs/MODEL_ANALYSIS.md` nineteenth pass. Every 2025 departure with an NM record has `|BLOCK - LOBT| <= 3,606` s, so the label lies within 1 h of `MVT - LOBT`. Two rules from that window (WINNING_PLAN J1, J2) price -564 hold-out MSE with no fit: `python -m src_v3.measure_nm_join`. No measured lever reaches 220 s; the top is 213.52 s.
- **Final phase, 2026-10-12:** `submission/kind-mango_final.parquet` is built and not uploaded yet: the v132 recipe on the 670,790 departures of January, February, June and July 2026 (`python -m src_v3.release_final tables`, then `serve`; `REPRODUCE.md` section 8). On the 344,841 departures of the ranking phase it differs from v132 by 6.6 s on average, because the organisers drew new timestamp noise. The same two commands rebuild v132 from the ranking-phase files.
- **Candidate v133, 2026-10-12, not uploaded:** the stack reads ADS-B segments that end up to 3,600 s before the take-off (`build_adsb_clock --first-gap 3600`, tables in `external/adsblol/v133`): 14,749 more ranking departures with a track, hold-out price -1,081 MSE (bar -150). The TabM families (`tabm_*_k9`) and the plain tree members (tag `v133`) refit on those tables.
- **Candidates v134 and v135, 2026-10-13:** v135 scored 232.45 s live (-3.09 s vs v132); v134 is not uploaded. v134 is v133 with the five TabM families refit on the v133 tables (hold-out -523 MSE against v133). v135 is v134 with the plain tree members refit on them (-58 MSE more, below the bar). The final-frame files are `submission/kind-mango_final134.parquet` and `kind-mango_final135.parquet` (`release_final serve --stack-adsb external/adsblol/v133 --tabm tabm_*_k9 --trees v133`; `REPRODUCE.md` section 8). `kind-mango_final.parquet` (v132 recipe) stays the fallback until a candidate is chosen.
- **v139, 2026-10-13, 231.90 s live (-0.13 s vs v138):** v138 with the month-level gap fit on the share of departures that the EUROCONTROL file drops (`release_final serve --month-level kept`); LFPG enters with +18 s in June and +32 s in July. `submission/kind-mango_final139.parquet` is the final-frame file, and `submission/kind-mango_final.parquet` holds a copy of it for the final upload.
- **v138, 2026-10-13, 232.03 s live (-0.17 s vs v137):** v137 with the month level (`python -m src_v3.month_level`, `release_final serve --month-level constant`). The June and July values of six airports move by one shift per airport and month to the mean taxi-out that EUROCONTROL publishes; the file is an aggregate of the labels and the organisers allow it. Hold-out -15 MSE in July 2025; `submission/kind-mango_final138.parquet` is the final-frame file.
- **v137, 2026-10-13, 232.20 s live (-0.25 s vs v135):** v135 with two member columns in the stack (the neural term minus the tree blend, the TabM mean minus the MLP; `release_final serve --member-columns`). Hold-out -76 MSE against v135, below the bar. `submission/kind-mango_final137.parquet` is its final-frame file. `docs/MODEL_ANALYSIS.md` twenty-third pass holds the budget of v135 and the probes that closed.
- **Current leaderboard best:** **231.90 s RMSE** (`kind-mango_v139.parquet`, 2026-10-13), -0.13 s vs v138, under the 0.3 s acceptance bar: LFPG enters the month level through a gap fit on the kept share of the EUROCONTROL file (live -59 MSE). Before that: **232.03 s RMSE** (`kind-mango_v138.parquet`, 2026-10-13), -0.17 s vs v137, under the 0.3 s acceptance bar: the July values of six airports shifted to the EUROCONTROL monthly mean taxi-out (live -81 MSE). Before that: **232.20 s RMSE** (`kind-mango_v137.parquet`, 2026-10-13 09:24 UTC), -0.25 s vs v135, 0.05 s short of the 0.3 s acceptance bar, team rank 21: two member columns in the stack (live -116 MSE, hold-out -76). Before that: **232.45 s RMSE** (`kind-mango_v135.parquet`, 2026-10-13 08:40 UTC), -3.09 s vs v132, accepted, team rank 22: the stack, the five TabM families and the plain tree members read the ADS-B segments that end up to 3,600 s before the take-off (`build_adsb_clock --first-gap 3600`; 14,749 more ranking rows with a track, live -1,445 MSE, hold-out sum about -1,660). One upload holds the three steps; v133 and v134 have no live score. Before that: **235.54 s RMSE** (`kind-mango_v132.parquet`, 2026-10-11 19:54 UTC), -0.41 s vs v131, accepted: plain LightGBM, plain CatBoost and per-airport LightGBM refit on the v130 ADS-B tables, so the tree base also sees the flight-number and lift-off links (live -194 MSE). Before that: **235.95 s RMSE** (`kind-mango_v131.parquet`, 2026-10-10 21:50 UTC), -2.30 s vs v130, accepted: the five TabM families refit on the v130 tables, so the networks also see the lift-off links (live -1,092 MSE). Before that: **238.25 s RMSE** (`kind-mango_v130.parquet`, 2026-10-10 09:11 UTC), -1.47 s vs v129, accepted: departures that no callsign links take the ADS-B aircraft of their type that lifts off at `MVT` (`build_adsb_clock --link-takeoff`; 1,016 more ranking rows with a track, live -703 MSE). Before that: **239.72 s RMSE** (`kind-mango_v129.parquet`, 2026-10-10 01:11 UTC), -0.87 s vs v128, accepted: the five TabM families refit on the clock tables in which the departures without an NM record link to their tracks (live -419 MSE). Before that: **240.60 s RMSE** (`kind-mango_v128.parquet`, 2026-10-09 05:17 UTC), the stack tuned for the 2026 coverage mix (live -52 MSE). Before that: **240.70 s RMSE** (`kind-mango_v127.parquet`, 2026-10-09 05:08 UTC), -0.95 s vs v126, accepted: departures without an NM record link to their own ADS-B and OPDI tracks by flight number (twenty-second pass). Before that: **241.66 s RMSE** (`kind-mango_v126.parquet`, 2026-10-09 00:06 UTC), a fifth TabM family `tabm_s` on the v124 recipe (live -6 MSE, inside noise); v125 (NM clocks in the LIRF gate) lost +0.67 s. Before that: **241.67 s RMSE** (`kind-mango_v124.parquet`, 2026-10-08 19:52 UTC), a fourth TabM family `tabm_plr_j` at the share 0.65 (live -57 MSE). Before that: **241.79 s RMSE** (`kind-mango_v123.parquet`, 2026-10-08 17:42 UTC), the TabM share 0.65 on the v122 families (live -27 MSE). Before that: **241.84 s RMSE** (`kind-mango_v122.parquet`, 2026-10-08 16:52 UTC), -0.18 s vs v121, under the 0.3 s bar: the v121 recipe with `tabm_j` as a third TabM family at the share 0.5 (hold-out -42 MSE, live -88). Before that: **242.03 s RMSE** (`kind-mango_v121.parquet`, 2026-10-08 08:39 UTC), -0.49 s vs v119, accepted: P1 (the 19 LFPG departures whose `MVT` lies off the minute grid, raised to `MVT - LOBT` minus 242 s; `python -m src_v3.mvt_precision`) bundled with six TabM seeds per family. Before that: **242.51 s RMSE** (`kind-mango_v119.parquet`, 2026-10-07 23:00 UTC), -0.295 s vs v118, 0.005 s short of the 0.3 s acceptance bar: J2 (the LIRF gate at 0 where the arrival-side `LOBT` shows a schedule copy would have joined, `predict_v67 --copy-linked`; 20 rows) bundled with a second TabM family, `tabm_plr` (periodic numeric embeddings), averaged with `tabm_v116`. Before that: **242.81 s RMSE** (`kind-mango_v118.parquet`, 2026-10-07 15:56 UTC), -0.85 s vs v116, accepted, team rank 34 of 235: J1 on the v116 recipe. `predict_v67 --copy-window` sets the LIRF gate to 0 where a schedule copy breaks the NM join window, and `python -m src_v3.lobt_window` projects every row with `LOBT` onto `MVT - LOBT` plus or minus 3,606 s as the last step. Hold-out -371 MSE, live -411. Before that: **243.65 s RMSE** (`kind-mango_v116.parquet`; v117 243.74 s, +0.09 s, rejected), 2026-10-06 16:34 UTC), -0.035 s vs v115, inside noise: X-Plane stands also at LEMD, LEBL and LFPG. The accepted recipe stays v115. Before that: **243.69 s RMSE** (`kind-mango_v115.parquet`, 2026-10-06 12:57 UTC), -0.33 s vs v114 and the accepted best: X-Plane Scenery Gateway stand positions where OSM lacks the assigned stand at EDDM and EHAM (stand-clock share in 2026 from 0.40 to 0.73 at EDDM and 0.52 to 0.76 at EHAM) feed the ADS-B clocks of the stack, the stand bound and the TabM member; the TabM member also reads the EUROCONTROL ATFM slot adherence per airport and day. Hold-out full path -33 MSE against v114. Before that: **244.02 s RMSE** (`kind-mango_v114.parquet`, 2026-10-06 03:17 UTC), -0.27 s vs v112: the 32-epoch TabM member, three seeds, at 0.4 (hold-out full path -380 MSE against v105, -60 against v112). It misses the 0.3 s acceptance bar of AGENTS.md rule 11.5, so the accepted recipe stays v112. Before that: **244.29 s RMSE** (`kind-mango_v112.parquet`,
  2026-10-06 00:57 UTC), -0.94 s vs v111: the TabM member with a 24-epoch schedule at 0.35 (hold-out full path -320 MSE against v105, -89 against v111), plus the Step A date-crossing fix on one LIRF row (`src_v3/step_a_date.py`); the live gain cannot be split between the two. Before that: **245.23 s** (`kind-mango_v111.parquet`,
  2026-10-05 23:21 UTC), -1.25 s vs v105: a TabM member (k = 16 MLPs with shared weights and
  piecewise-linear numeric embeddings, `src_v3/tabm_member.py`, trained on the GPU) at 0.3 on top
  of the v105 base. Hold-out price of the full serve path -231 MSE (January -129, July -102); live
  about -614 MSE. Leaderboard top 213.92 s, kind-mango rank 35 before v111 (API, 2026-10-05).
  `docs/MODEL_ANALYSIS.md` eighteenth pass. Before that:
  **246.48 s RMSE** (`kind-mango_v105.parquet`,
  2026-10-04): v97 with a LIRF fallback gate that reads 2026 arrival information
  (the airline's and operator's share of arrivals with in-block on the schedule,
  and the previous arrival at the stand; `src_v3/build_arr_artefact.py`, gate g3)
  and the MLP member retrained on the v97 columns at 0.2. Path from v97: v101
  247.69, v102 247.14, v104 246.95, v105 246.48 s (RECAP rows). Rejected since: v108 246.49 s (N2 LightGBM members) and v109 247.37 s (N7, a CatBoost member in the LIRF gate, `src_v3/catboost_gate.py`) and v110 246.57 s (N3, NNLS weights of the base members per coverage class, `src_v3/blend_nnls.py`); `docs/MODEL_ANALYSIS.md` seventeenth pass. Before that:
  **247.89 s RMSE** (`kind-mango_v97.parquet`, 2026-10-03), -1.50 s vs v95; team rank 31. The leaderboard top is 216.57 s
  (API, 2026-10-03). **Next:** `docs/WINNING_PLAN.md` third edition, the path
  from 247.89 s toward 210 s. `python -m src_v3.measure_budget_v97` and
  `python -m src_v3.measure_lirf_gap` measure where the v97 error sits
  (`docs/MODEL_ANALYSIS.md` sixteenth pass). v97 refits every base member (global, anchored
  and per-airport LightGBM, plain and anchored CatBoost) with the OPDI and
  ADS-B pushback columns of 350 days of 2025, then serves the v95 stack on top.
  v95 (249.39 s) trains the pushback stack on all 62
  January and July 2025 days of ADSB.lol data; `src_v3/auto_release.py`
  built, gated and uploaded it unattended. v94 (250.51 s) blends the MLP member into the base at
  0.15 outside LIRF. v93 (251.44 s) trains the pushback stack on 21 ADSB.lol
  days of 2025 instead of 7. v92 (253.82 s) serves the v91 stack on all 62 ranking
  days of ADSB.lol data and floors each row at the last time the aircraft
  was at its assigned stand (WINNING_PLAN second edition, A5). v91 (259.39 s) adds the ADSB.lol pushback clock of the
  scored flight to the stack (WINNING_PLAN A2), on 24 of 62 ranking days. v90 (263.17 s) stacks the OPDI pushback clock of the
  scored flight on v89 (WINNING_PLAN A1, second edition). v89 (264.34 s) mixes one LightGBM member per airport
  50/50 into both LightGBM terms of v85. v85 (264.55 s) added forward windows (WINNING_PLAN
  L8): the NM clocks of the departures that take off in the 15 and 30 min
  after the row. v76 (265.27 s) refit both LightGBM bases with the
  parameters of a new Optuna study on the clock-era columns (MB5). v75
  (266.15 s) added a CatBoost member on the residual from the `AOBT_3`
  anchor; v74 scored 266.84 s. v73 (270.58 s) added the NM clocks of the
  scored flight (`MVT - AOBT_3_flt`, `LOBT_flt` deltas) to the LightGBM base
  (WINNING_PLAN L1, track B). v74 adds the clocks to the CatBoost member and
  a LightGBM member on the residual from the `AOBT_3` anchor (L5). The team changed its stance on these fields
  on 2026-09-26; see the Stance section below. Before v73, the best was
  281.87 s (`kind-mango_v67.parquet`, 2026-09-24). Path from v57 (287.33 s): Step A normal
  term reads `R_norm` (v63, 287.09 s), out-of-fold isotonic map on the LIRF
  gate (v64, 285.95 s), planned-taxi column `plan_nm_taxi` (v65, 284.74 s),
  CatBoost second model class blended 50/50 with the LightGBM base outside
  LIRF (v67, 281.87 s). See `docs/WINNING_PLAN.md`. Path: L3.a/L9/L1 stacked base changes
  (v44-v46, live -0.17 s), L2 12-month refit (v47 = 296.57, -2.57 s),
  fifteenth-pass MS1+MS2 bounds (v48 = 294.24, -2.33 s), fifteenth-pass
  MB3 base retrained on served rows only, LIRF and y>80,000 excluded
  (v51 = 288.25, -5.99 s vs v48). MB3 is the biggest single lever of
  the season: -5.99 s live at zero paired price, because the base's
  linear leaves were being shaped by LIRF's 16.6 % fallback rate and
  the 12 24-h rows and extrapolating badly on 2026 out-of-distribution
  rows. See `docs/MODEL_ANALYSIS.md` fifteenth pass and `RECAP.md` for
  the full decomposition per ship.
- **After v33 (status 2026-09-13):** v34 to v37 scored 302.05 to 302.52 and
  are rejected. v38 failed its hold-out gate. v39, the cold-start rewrite
  under `src_v2/`, scored 352.19 s live: one LIRF row with `sd = 94,560`
  lost the ITY340 hedge and holds 62 to 73 % of the regression, and the
  rewritten base is 4.5 s worse on clean rows than the v33 base on identical
  hold-out rows. v33 stays best. See `docs/MODEL_ANALYSIS.md` section 4.1.

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

Pushback clocks from open trajectory data (changed 2026-09-29): from v90
the served file reads the pushback time of the scored flight from OPDI
flight events, and from v91 also from ADSB.lol traces. The organiser
allowed this on 2026-09-18: "if you can find open trajectory data with good
ground coverage and extract off-block times, we are ok" (`RECAP.md`
Discord confirmations). Raw OSN state vectors and Trino stay out.
The clocks are not the label: after a bias, 57 to 86 % of the linked
rows lie within 60 s, and a stack model learns when to trust them.

Risk: on 2026-09-11 the organiser said that reverse engineering of the
ranking is a practice "we despise" and kept the right to add a phase 2. A
phase 2 without NM actuals cancels this gain. No submission sets a row
value from a leaderboard score.

## What the model is

The scoring stack (v41, current best) is v33 with the base and the LIRF
regressor retrained on 13 extra columns. It is documented in
`docs/MODEL_ANALYSIS.md` sections 4.2 and 4.3. In one line:

**Three-seed LightGBM regressor with `linear_tree` (`R_all_v40`: the v26 recipe
plus 13 tempo, order, stand-gap and queue columns), plus a LIRF
regime head (5-member `R_norm_LIRF` mean + calibrated `p_fb` mixture), plus
the Step A LIRF band lookup, plus the ITY340 offset rule.**

Per-airport delegation. `src/predict_v33.py` calls `src/predict_v30.py:main`
with the 5 `R_norm_LIRF` members:

| airport | scoring component |
|---|---|
| LIRF, `sd > 14,400`, null flight | Step A band table (`p_fb * sd + p_24h * (86,400 + mean_24h_extra) + p_norm * mixture`), exclusive classes since v30 |
| LIRF, `sd > 70,000`, outside Step A | ITY340 constant `(5/6) * (86,400 + 1,150) + (1/6) * sd` |
| LIRF, all other rows | `mixture = p_fb_cal * sd + (1 - p_fb_cal) * R_norm_LIRF`, five v41 members, no cap |
| every other airport | 3-seed mean of `lgbm_r_all_v26_s{42,43,44}` |

Both LIRF-head boosters read the LIRF-only encoders in
`models/lirf_regime.encoders.pkl`. Serving them the all-airport encoders was
the v30 defect fix. The v33 build swaps `R_norm_LIRF` for a 5-member mean
(seeds 42-46). The price was 74 MSE of variance reduction. The live score
confirmed it. `mean_24h_extra` is a per-band value in
`models/lirf_band_table_v30.json`; the fallback is 1,150 s.

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
| kind-mango_v44 | pending | — | v41 base retrained with 6 new columns: 25th and 75th percentiles of neighbour `mvt_eobt1` at the apt30, apt60 and rwy30 windows (`build_tempo_p2575.py`, `train_r_all_v44.py`). Paired -565 MSE on the base's served rows; gate 300 passed. Uploaded 2026-09-16. |
| kind-mango_v45 | pending | — | v44 base retrained with `plan_taxi_res` (clipped +/-3600 s), the row's plan-block minus its 2025 route median (`build_plan_taxi_res.py`, `train_r_all_v45.py`). Paired -617 MSE on the base's served rows (-650 on clean); gate 300 passed. Uploaded 2026-09-16. |
| kind-mango_v46 | 299.14 | -0.17 | v45 features (117 cols) with the L1 retuned base: 30-trial Optuna sweep on `tune_lgbm_v43.py` picked num_leaves 436, min_data 291, lr 0.018, linear_lambda 0.006 (vs deployed 220 / 76 / 0.023 / 1.0). Paired -1,864 MSE served clean cumulative for the L3.a+L9+L1 stack; only -102 MSE landed live. |
| **kind-mango_v47** | **296.57** | **-2.57** | **L2: 12-month refit of the v46 recipe (`train_r_all_v47.py`).** 3 seeds on ALL months with the retuned params, iterations scaled by 1/0.88 (v46 iters 609/1799/1015 -> 693/2045/1154). Removed the 2025 hold-out. Live -2.57 s vs v46 shows most of the reachable gain came from training on months 1 and 7 directly, not from feature engineering or hyperparameter tuning. |
| **kind-mango_v48** | **294.24** | **-2.33** | **Fifteenth-pass Phase 1: MS1 per-airport upper bounds and MS2 clean-floor post-processing on the v47 parquet (`src_v3/build_v48.py`).** 5 non-LIRF extreme rows capped (biggest EDDM 21,465 -> 7,499 s). MP7 label-free gate passed. Zero training cost. Live -2.33 s (about 1,400 MSE) essentially all from the five MS1 clips - MS1 fold price was 0 MSE because the base doesn't extrapolate on 2025 rows, but does on 2026 out-of-distribution rows. |
| kind-mango_v49 | 294.65 | +0.42 | Fifteenth-pass MD2+MH1 rebuild of the LIRF head without the 56 drifted columns; paired LIRF hold-out -2,405 MSE did not transfer live. Head rebuild lever closed. |
| kind-mango_v50 | pending | — | Fifteenth-pass Phase 1: MS3 member-disagreement median on top of v48. 17 rows moved (EDDF 3, EDDM 4, EHAM 4, LEBL 1, LFPG 5) where the 3 v47 base members disagreed by more than 3,600 s outside LIRF. 15 remain distinct after MS1's cap. MP7 gate passed. |
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
| **kind-mango_v90** | **263.17** | **-1.17 vs v89 (new best)** | WINNING_PLAN A1: the OPDI pushback clock of the scored flight. `src_v3/build_opdi_own.py` links a departure to its OPDI flight by airport, `CALLSIGN_flt` and the runway entry closest to `MVT`, and writes three clocks (first, last and own-stand exit before the runway entry). A LightGBM stack on `y - base` (`src_v3/stack_pushback_v90.py`) reads the clocks, `MVT - AOBT_3` and the airport; rows without a clock keep v89. Hold-out, month cross-fit on the out-of-sample v89-style blend: -566 MSE, LSZH RMSE 167.9 to 126.0 s on 15,431 rows. It moves 21,498 ranking rows (LSZH 14,269, EDDF 7,124). Run: `python -m src_v3.build_opdi_own`, then `python -m src_v3.stack_pushback_v90 serve --sources opdi --out kind-mango_v90.parquet`. |
| **kind-mango_v91** | **259.39** | **-3.78 vs v90 (new best)** | WINNING_PLAN A2: the ADSB.lol pushback clock of the scored flight, added to the v90 stack. `src_v3/fetch_adsb_days.py` downloads a `globe_history` day (about 4 GB), `src_v3/build_adsb_points.py` keeps the points near the 10 airports below 5,000 ft, and `src_v3/build_adsb_clock.py` links a departure to its aircraft by callsign, cuts the last ground segment before take-off and writes candidate clocks (first point, first move, first move near the assigned stand, first move after a long stop) with their quality columns. Hold-out (7 days of 2025, day-parity cross-fit): -937 MSE; covered-row RMSE EHAM 141 to 105 s, EDDF 182 to 151 s. Served on 24 of the 31 July 2026 days (148,836 rows); moves 97,793 rows. Run: `python -m src_v3.fetch_adsb_days --start 2026-07-01 --end 2026-07-31`, `python -m src_v3.build_adsb_clock`, then `python -m src_v3.stack_pushback_v90 serve --sources both --out kind-mango_v91.parquet`. |
| **kind-mango_v132** | **235.54** | **-0.41 vs v131 (new best, accepted)** | v131 with the three plain tree members refit on the v130 ADS-B tables (flight-number and lift-off links): `lgbm_r_all_v130` (three seeds), `catboost_r_all_v130c` and the per-airport members `lgbm_pa_v130_*`. The three anchored members keep the v97 fits and the v97 ADS-B table (`predict_v67 --anchor-adsb`, `--per-airport-anchor-tag v97a`): a departure without an NM record has no `AOBT_3` anchor, so the new links reach the plain members only, and a refit of all six members takes 12.5 h (v117). The MS3 guard is on for the per-airport mix (`--pa-guard`). The stack learns from the hold-out blend of the same members (`measure_adsb_base_v97 price --tag v130 --pa-guard --keep-v97 lgb_anchor cat_anchor`, `blend_v130_holdout.parquet`). The TabM term, the stack options and the rules are those of v131. Hold-out, 2025: plain CatBoost -435 MSE, per-airport LightGBM -385 and plain LightGBM -366 against their v97 members; blend -415; blend plus the v95 stack -144 (`budget_v130.holdout.json`: stand-clock rows -28, segment rows -136, rows without a track +22). Live -194 MSE. |
| kind-mango_v131 | 235.95 | -2.30 vs v130 (accepted) | v130 with the five TabM families refit on the K8 tables: column sets v116k8, v120k8 and v126k8 are the K7 sets with the v130 ADS-B tables, in which departures that no callsign links take their lift-off aircraft (`tabm_s_k8`, `tabm_j_k8`, `tabm_plr_j_k8`, `tabm_v116_k8`, `tabm_plr_k8`, three seeds each; one fifth per family at the share 0.65). The stack, the stand bound and the rules are those of v130. No hold-out price before the upload (user directive); the five refits ran as one detached chain of 9.8 h. Live -1,092 MSE. |
| kind-mango_v130 | 238.25 | -1.47 vs v129 (accepted) | v129 with the stack and the A5 stand bound on the v130 ADS-B tables. A departure that no callsign links (F22.1 left 3,578 of 5,290 rows without an NM record unlinked) takes the aircraft of its type whose first airborne point lies within 45 s of `MVT` plus the airport offset (`build_adsb_clock --link-takeoff`; offsets -24 to +30 s, std 9 to 22 s on 2026 callsign links; time and type alone pick the callsign aircraft on 99.4 %). Lift-offs of aircraft that a callsign links are skipped. `--keep external/adsblol/v127` keeps every row with an NM callsign or a segment. New segments: 1,016 ranking rows (EHAM 697, LSZH 234, EDDF 85; 537 stand clocks) and 2,832 rows of 2025; on 2025 stand clocks the label minus `adsb_own_move` has a median of -5 s (callsign links +25 s). TabM, the trees and the rules are those of v129. No hold-out price before the upload (user directive). Live -703 MSE. |
| kind-mango_v129 | 239.72 | -0.87 vs v128 (accepted) | v128 with the five TabM families refit on the K7 tables: column sets v116k7, v120k7 and v126k7 read the v127 ADS-B and OPDI tables, in which the departures without an NM record link to their tracks (`tabm_s_k7`, `tabm_j_k7`, `tabm_plr_j_k7`, `tabm_v116_k7`, `tabm_plr_k7`, three seeds each; one fifth per family at the share 0.65). The stack, the stand bound and the rules are those of v128. No hold-out price before the upload (user directive); the five refits ran as one detached chain of 9.8 h, and `serve` now saves each fit for the final phase. Live -419 MSE. |
| kind-mango_v128 | 240.60 | -0.11 vs v127 (leaderboard best; under the 0.3 s bar) | v127 with the stack tuned for the 2026 coverage mix (`stack_pushback_v90 serve --coverage-weights --leaves 15 --min-data 100 --l2 50 --rounds 400`). Each hold-out clock row carries the 2026-to-2025 share of its airport and coverage class as its weight (bounds 0.2 and 5; no label). The parameters are the best of 54 sets on the weighted cross-fitted objective (`tune_stack_v128.json`): weighted MSE 21,186 against 21,249 for the control; unweighted hold-out gain -113 against -88. Smaller, more regularized stacks won; the most flexible set lost +79. Live -52 MSE. |
| **kind-mango_v127** | **240.70** | **-0.95 vs v126 (new best, accepted)** | v126 recipe; the stack and the A5 stand bound read new clock tables. Departures without an NM record have no `CALLSIGN_flt`, so the builders never linked them to their tracks. `build_adsb_clock --link-flight --keep external/adsblol/v116` and `build_opdi_own --link-flight --keep ...` link them by `FLIGHT_mvt`; an IATA airline code takes the ICAO code paired with it on record rows (98 codes). 2026: 737 stand clocks and 975 segments on 1,712 of 5,290 no-record rows, OPDI 157; rows with an NM callsign keep the v116 and v90 values. Stack `--adsb-train/--adsb-rank external/adsblol/v127/...`, `--opdi-train/--opdi-rank models/opdi_own_*_v127.parquet`. No hold-out price before the upload (user directive). Live -461 MSE. |
| kind-mango_v126 | 241.66 | -0.01 vs v124 (leaderboard best; inside noise) | v124 plus a fifth TabM family, `tabm_s`: the v120 columns plus the structure terms of the record (`python -m src_v3.build_struct_terms`: take-off second off the minute grid, NM default-taxi share of `MVT - AOBT_3`, gaps to the previous and next departure of the same flight number), column set v126, three seeds; one fifth per family at the share 0.65. Live -6 MSE. |
| kind-mango_v125 | 242.35 | +0.67 vs v124 (rejected) | v124 with the LIRF gate g5: the g3 columns plus the five NM clock columns (`train_p_fb_v83 --tag g5 --extra models/arr_art_op_link_train_g3.parquet models/nm_clocks_train_v73.parquet`; 207 rounds, 182 columns; `predict_v67` with the g5 gate files, `--copy-window --copy-linked`). Only LIRF rows move: 25,197 rows, mean +1.3 s, p99 214 s. Live +326 MSE; the hold-out priced +108. |
| kind-mango_v124 | 241.67 | -0.12 vs v123 (leaderboard best; under the 0.3 s bar) | v123 plus a fourth TabM family, `tabm_plr_j`: periodic embeddings on the v120 column set (the v116 columns plus the unclipped `LOBT` terms), three seeds; one quarter per family at the share 0.65 (`combine --tags tabm_v116 tabm_v116_b tabm_plr tabm_plr_b tabm_j tabm_j tabm_plr_j tabm_plr_j --share 0.65`, stack `--mlp-weight 0.72`). No hold-out price before the upload (user directive). Live -57 MSE. |
| kind-mango_v123 | 241.79 | -0.06 vs v122 (leaderboard best; under the 0.3 s bar) | v122 with the TabM share 0.65 instead of 0.5 (`combine --tags tabm_v116 tabm_v116_b tabm_plr tabm_plr_b tabm_j tabm_j --share 0.65`, stack `--mlp-weight 0.72`). No hold-out price before the upload (user directive of 2026-10-08); only the format checks blocked. Live -27 MSE. |
| kind-mango_v122 | 241.84 | -0.18 vs v121 (leaderboard best; 0.11 s under the 0.3 s bar) | v121 plus one change: `tabm_j` (TabM on the v116 columns plus the unclipped `LOBT` terms, the served files of v120) as a third TabM family, one third per family, at the TabM share 0.5 (`combine --tags tabm_v116 tabm_v116_b tabm_plr tabm_plr_b tabm_j tabm_j --share 0.5`, stack `--mlp-weight 0.6`); J3 stays out. Hold-out full path -41.9 MSE (body -40.6; January -26.7, July -15.2; `tabm_j_v122.holdout.json`). Uploaded under the user rule of 2026-10-03 (the hold-out bar does not stop an upload). Live -88 MSE. |
| **kind-mango_v121** | **242.03** | **-0.49 vs v119 (new best, accepted)** | v119 plus two parts, bundled because P1 alone moves 19 rows (AGENTS rule 12). (1) P1, twentieth pass: at LFPG every movement time is a whole minute plus organiser noise of at most 6 s; a departure whose `MVT` lies off that grid has a long label (97 rows of 2025, median 4,453 s against 1,191 s on the grid in the same cell). `python -m src_v3.mvt_precision serve` raises the 19 flagged ranking rows to `MVT - LOBT` minus 242 s, the 2025 median offset, and never lowers a row (18 raised, mean +2,045 s). Hold-out -250 MSE on 17 rows (`mvt_precision.holdout.json`). (2) N10 seeds: TabM members `tabm_v116_b` and `tabm_plr_b` (seeds 45 to 47), so each family averages six seeds (`combine --tags tabm_v116 tabm_v116_b tabm_plr tabm_plr_b --share 0.4`); hold-out +34 MSE, read as stack-refit noise (`tabm_seeds_v121.holdout.json`). Live -235 MSE against -216 priced. |
| kind-mango_v120 | 242.91 | +0.40 vs v119 (rejected) | v119 plus two parts, bundled because J3 alone moves 93 rows (AGENTS rule 12). (1) J3: `python -m src_v3.lobt_window ... --anchor` serves `MVT - LOBT` minus the 2025 median `BLOCK - LOBT` of its band on rows whose `LOBT` lies 2 to 4 h before the take-off (hold-out -255 MSE, all on rows above 3,600 s). (2) `tabm_j`: TabM on the v116 columns plus the unclipped LOBT terms (`python -m src_v3.build_lobt_terms`, column set v120), a third member at the TabM share 0.5 (`combine --tags tabm_v116 tabm_plr tabm_j --share 0.5`, stack `--mlp-weight 0.6`; hold-out -17 MSE). Live +194 MSE; both parts close. |
| kind-mango_v119 | 242.51 | -0.295 vs v118 (leaderboard best; 0.005 s under the 0.3 s bar) | v118 plus two parts, bundled because J2 alone moves 20 rows (AGENTS rule 12). (1) J2: `predict_v67 --copy-window --copy-linked` also sets the LIRF gate to 0 on departures without an NM record whose arrival row at another of the ten airports gives a `LOBT` within 3,606 s of `SCHED`: the departure failed the join, so its off-block is not `SCHED` (`lobt_window.copy_would_join`; hold-out -193 MSE). (2) `tabm_plr`: the TabM member with periodic numeric embeddings (`tabm_member.Spec.embed = "plr"`, stop-month 161.40 s, three seeds), averaged with `tabm_v116` at the TabM share 0.4 (hold-out full path -83 MSE). Run: the v118 base command with `--copy-linked`, `PYTHONPATH=.gpu_site python -m src_v3.tabm_member serve --tag tabm_plr`, `combine --tags tabm_v116 tabm_plr --share 0.4`, the v118 stack command on the new base, `step_a_date`, `lobt_window`. Live -143 MSE. |
| **kind-mango_v118** | **242.81** | **-0.85 vs v116 (new best)** | J1 of the nineteenth pass: every 2025 departure with an NM record has `|BLOCK - LOBT| <= 3,606` s, so the label lies within 1 h of `MVT - LOBT`. (1) `predict_v67 --copy-window` rebuilds the v104 base with the LIRF gate set to 0 on the 1,550 rows where a schedule copy would break that window (`|SCHED - LOBT| > 3,606` s; their mean gate was 0.0083); the base outside LIRF matches v104 at 0.0 s. (2) The v116 stack and `step_a_date`, then `python -m src_v3.lobt_window` as the last step projects every row with `LOBT` onto the window (54 rows). Hold-out -371 MSE (`python -m src_v3.measure_nm_join`); live -411 MSE. 1,123 rows moved against v116, none by more than 3,000 s; rows above 7,200 s 101 to 104. Commands in REPRODUCE.md; artefacts `models/v118.manifest.json`. |
| kind-mango_v117 | 243.74 | +0.09 vs v116 (rejected) | The v116 recipe with every tree member refit on the v116 clock tables. Hold-out members on the v116 tables against the v97 members (same recipes, fixed rounds): plain LightGBM -207, anchored LightGBM -307, plain CatBoost -34, anchored CatBoost -356, per-airport LightGBM +1,057 (six EHAM rows at 4,000 to 13,000 s against about 1,300 s); blend -218, -232 with the MS3 guard on the per-airport mix (`predict_v67 --pa-guard`, `measure_adsb_base_v97 price --pa-guard`). Blend plus stack against v89: -1,205 MSE (v97 members: -983). Run: `python -m src_v3.measure_adsb_base_v97 fit <member> --tag v116 --adsb-train external/adsblol/v116/adsb_clock_train.parquet` for the five members, `price --tag v116 --pa-guard`; full-year `train_v76_base plain|anchored --tag v116`, `catboost_base --tag v116c|v116ca`, `train_per_airport_v89 plain|anchored --tag v116`, all with the v116 clock table in `--extra`; `predict_v67 ... --base-model lgbm_r_all_v116 --catboost-tag v116c --catboost-anchor-tag v116ca --anchor-model lgbm_r_all_v116a --per-airport-tag v116 --pa-guard ...` with the v116 rank table; then the v116 stack command with `--blend-holdout models/v3/blend_v116_holdout.parquet` and `step_a_date`. The hold-out price used the stack on the old clock table; since v115 the served stack, A5 bound and TabM already read the new clocks, so the refit repeats that gain. Built and uploaded by a detached chain (`logs/release_v117.log`). |
| kind-mango_v116 | 243.65 | -0.035 vs v115 (inside noise; leaderboard best) | v115 with X-Plane stand positions also at LEMD (scenery 99851), LEBL (102255) and LFPG (111022); a leading type word is dropped before the match (LEBL `Ramp 287`). Departures that gain a stand position: LEMD 15.6 %, LEBL 9.0 %, LFPG 10.0 %. On 2025 labels the filled stands give clocks as close as OSM stands (within 60 s: LEMD 0.81 against 0.55, LEBL 0.76 against 0.77). 2026 stand-clock share: LEMD 0.138 to 0.182, LEBL 0.346 to 0.367, LFPG 0.001 (no ADS-B ground coverage). Clock tables in `external/adsblol/v116/`; TabM `tabm_v116` (epoch 27, three seeds) at 0.4: -485 MSE on the v105 base against -390 for `tabm_v115`, most of it run-to-run spread at airports whose inputs did not change; the stack on the new tables -1.7 MSE (the 2025 hold-out has almost no LEMD coverage). Gates vs v115 passed. |
| **kind-mango_v115** | **243.69** | **-0.33 vs v114 (new best)** | (1) `src_v3/build_xplane_stands.py`: X-Plane Scenery Gateway `apt.dat` stand positions (GPL v2+) for the stand values OSM lacks at EDDM (69 values) and EHAM (49); median distance to OSM on common stands 28 m and 41 m. The ADS-B clock tables rebuilt with them (`external/adsblol/v115/`, other airports identical): stand-clock share 2026 EDDM 0.396 to 0.726, EHAM 0.519 to 0.763. The tree members keep their old tables; the stack, the A5 bound and TabM read the new ones. (2) `src_v3/build_atfm_slot.py`: departures, regulated share, early and late off-slot shares per airport and UTC day in the TabM member. TabM `tabm_v115`: stop-month 161.6 s at epoch 21, on the v105 base -390 MSE at 0.4 (`tabm_e32`: -386); three seeds at 0.4. Hold-out full path against v114: stack with the new clocks -27, TabM -12, both -33 MSE (EDDM -46, LTFM -94, EGLL +77). Run: `python -m src_v3.build_xplane_stands fetch`, `build`; `python -m src_v3.build_adsb_clock --stands external/xplane/stands_osm_xplane.parquet --out-dir external/adsblol/v115`; `python -m src_v3.build_atfm_slot`; `PYTHONPATH=.gpu_site python -m src_v3.tabm_member measure --tag tabm_v115`, `serve --tag tabm_v115`, `combine --tags tabm_v115 --share 0.4`; the v114 stack command with `--adsb-train external/adsblol/v115/adsb_clock_train.parquet --adsb-rank external/adsblol/v115/adsb_clock_rank.parquet`; then `step_a_date`. Gates vs v114 passed; stack moves 202,073 rows, bound lifts 951. Rebuild from `models/v115.manifest.json` at 2e-13 s. |
| kind-mango_v114 | 244.02 | -0.27 vs v112 (leaderboard best; under the 0.3 s bar) | The v112 recipe with the TabM member `tabm_e32` (32-epoch schedule, best epoch 27, stop-month 162.19 s; three seeds) at 0.4 instead of `tabm_e24` at 0.35; neural term `nn` = 0.229 MLP + 0.771 TabM at 0.52. Hold-out: full path -380 MSE against v105 (v112 -320), negative at all 9 airports; a second seed of the hold-out member adds about -20 MSE. Run: `PYTHONPATH=.gpu_site python -m src_v3.tabm_member serve --tag tabm_e32` (three seeds, epoch 27 of a 32-epoch schedule), `combine --tags tabm_e32 --share 0.4`, the v112 stack command with `--mlp-weight 0.52`, then `step_a_date`. Gates vs v112 passed. |
| kind-mango_v113 | 247.40 | +3.12 vs v112 (rejected) | The v112 recipe with the TabM gate member `tabm_p_fb_lirf_g3t.pt` (`src_v3/tabm_gate.py`) averaged with the g3 LightGBM gate (`predict_v67 --gate-catboost tabm_p_fb_lirf_g3t.pt lirf_regime_g3t.isotonic.pkl`). Hold-out mixture -319 MSE (January -16, July -303). Only LIRF rows moved: 26,576 rows, mean +37.1 s, max 3,929 s; the rest of the file equals v112. The fifth LIRF-head model change to lose live (v49, v68, v100, v109, v113); LIRF-head model classes are closed. |
| **kind-mango_v112** | **244.29** | **-0.94 vs v111 (new best)** | Two changes: (1) the TabM member `tabm_e24` (24-epoch one-cycle schedule, best stop-month epoch 23, 162.68 s) at 0.35 on top of the v105 base instead of `tabm` at 0.3; neural term `nn` = 0.271 MLP + 0.729 TabM at 0.48; hold-out full path -320 MSE against v105 (v111: -231). (2) Step A without the 24-h term where the local dates of the schedule and the take-off do not cross: one LIRF row (TAY9LA, delay 53,520 s) from 64,814 to 53,520 s; 2025 shows no 24-h label without a crossing (0 of 64 null-record rows) and no genuine taxi above 25,000 s. Rule 12 bars a one-row file, so the fix rides on change (1). Run: `PYTHONPATH=.gpu_site python -m src_v3.tabm_member serve --tag tabm_e24`, `PYTHONPATH=.gpu_site python -m src_v3.tabm_member combine --tags tabm_e24 --share 0.35`, the v111 stack command with `--mlp-weight 0.48 --mlp-columns nn --out kind-mango_v112_stack.parquet`, then `python -m src_v3.step_a_date kind-mango_v112_stack.parquet kind-mango_v112.parquet`. Gates of the stack file vs v111: LIRF unchanged, rows above 7,200 s 98, no row moved by more than 3,000 s. |
| **kind-mango_v111** | **245.23** | **-1.25 vs v105 (new best)** | TabM member on the v97 columns (`src_v3/tabm_member.py`; `tabm` and `rtdl_num_embeddings`, Apache-2.0; CUDA build of torch in `.gpu_site`, see `requirements.txt`). Hold-out (`PYTHONPATH=.gpu_site python -m src_v3.tabm_member measure`): stop-month RMSE 163.5 s at epoch 13 (MLP 171.0 s), member RMSE 266.8 s against 265.6 s for the v97 blend, residual correlation 0.98 with the blend and 0.95 with the MLP; on top of the v105 base -214 / -252 / -244 MSE at 0.2 / 0.3 / 0.4; full serve path with stack and A5 bound -231 MSE at 0.3. Served at the pre-registered 0.3 on top of the v105 base: the neural term `nn` = 0.318 MLP + 0.682 TabM at 0.44. Run: `PYTHONPATH=.gpu_site python -m src_v3.tabm_member serve`, `PYTHONPATH=.gpu_site python -m src_v3.tabm_member combine`, then `python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound --rounds 600 --adsb-train external/adsblol/v97/adsb_clock_train.parquet --blend-holdout models/v3/blend_v97_holdout.parquet --mlp-weight 0.44 --mlp-columns nn --base kind-mango_v104_base.parquet --out kind-mango_v111.parquet`. Gates vs v105: LIRF unchanged, rows above 7,200 s 101, no row moved by more than 3,000 s. |
| kind-mango_v98 | 248.25 | +0.36 vs v97 (rejected) | v97 members with serve-time parameters chosen on the hold-out (`src_v3/release_v98.py`): CatBoost share 0.4, anchored 0.5, per-airport 0.7 (blend -65 MSE, better in January and July), MLP 0.1 and 400 stack rounds (-58 MSE). About -120 MSE on the hold-out; did not transfer. |
| **kind-mango_v97** | **247.89** | **-1.50 vs v95 (new best)** | WINNING_PLAN second edition 4.5, last step: every base member refit with the OPDI and ADS-B pushback columns (`opdi_own_train_v90`, `adsb_clock_train` on 350 days of 2025; 95.7 % of training rows on loaded days). Hold-out (`python -m src_v3.measure_adsb_base_v97 fit <member>`, then `price`): one LightGBM member -4,848 MSE against v85; v97 blend -5,377 against the v89 blend; blend plus stack -983 MSE against v95's recipe. Full-year members: `train_v76_base plain|anchored --tag v97`, `train_per_airport_v89 plain|anchored --tag v97`, `catboost_base --tag v97c|v97ca` (all with `--extra models/nm_forward_train_v85.parquet` or the clock table, plus the two pushback tables). Served and uploaded by `src_v3/release_v97.py`; stack trained on `blend_v97_holdout`. Gates vs v95: LIRF unchanged, rows above 7,200 s 107, no row moved by more than 3,000 s. |
| kind-mango_v96 | 249.92 | +0.53 vs v95 (rejected) | v95 plus one stack column: an ADS-B specialist (`src_v3/adsb_specialist.py --tag v96`), a LightGBM on the OPDI, ADS-B and NM clocks trained on 339,400 clock rows of 291 days outside months 1 and 7 (hold-out RMSE 180.2 s without a base). Study (`python -m src_v3.tune_stack_v96`): -302 MSE against v95 on the 2025 hold-out; the price did not transfer. The same study closed 63 leaves (-92), a residual clip of 7,200 s (+19), A5 offsets (+29 to +523) and A4 columns (+37). |
| **kind-mango_v95** | **249.39** | **-1.12 vs v94 (new best)** | The v94 recipe with the stack trained on all 62 January and July 2025 ADSB.lol days (v94: 21). `python -m src_v3.auto_release --version 95 --wait-pid <download pid>` waited for the downloads, rebuilt the clock tables, priced the stack rounds (400: -4,101, 600: -4,048, 900: -3,982 MSE; 600 kept, the 53 MSE difference is under the 150 MSE bar), served, passed the label-free gates against v94 (LIRF unchanged, rows above 7,200 s 106, no row moved by more than 3,000 s) and uploaded at 01:00 local. Record: `submission/kind-mango_v95.recipe.json`; parity `python -m src_v3.test_parity --version 95`. |
| **kind-mango_v94** | **250.51** | **-0.93 vs v93 (new best)** | v93 with the MLP member (`src_v3/mlp_member.py`: embeddings, 1024-512-256, three seeds) blended into the base outside LIRF at 0.15, on the hold-out rows that train the stack and on the ranking rows. 0.15 is the optimum of the quadratic through the hold-out prices -190, -213 and -69 MSE at 0.1, 0.2 and 0.3. Study (`python -m src_v3.tune_stack_v93`): the MLP gain adds to the stack gain (-1,819 against -1,603 at 0.2). Moves 317,018 rows, max 722 s; rows above 7,200 s 105. Live -470 MSE against a hold-out price of about -224. Run: `python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound --mlp-weight 0.15 --adsb-train external/adsblol/v94/adsb_clock_train.parquet --base kind-mango_v89.parquet --out kind-mango_v94.parquet`; parity `python -m src_v3.test_parity --version 94`. |
| **kind-mango_v93** | **251.44** | **-2.38 vs v92 (new best)** | The v92 recipe with the stack trained on 21 ADSB.lol days of 2025 (2025-01-01 to 01-15, 07-01 to 07-05, 07-15; 60,750 stack rows) instead of 7; the stack now cross-fits January against July. Study: stack -1,575 MSE against -937 on 7 days; 1,200 rounds worse than 600; 63 leaves and the A4 neighbour columns within 30 MSE (not adopted). EGLL mean +28.7 s against v92. Live -1,198 MSE against a hold-out difference of about -640. Download fix: some days ship as one unsplit `.tar`; days without a `prod-0` release (2025-05-28 to 06-10) are skipped. |
| **kind-mango_v92** | **253.82** | **-5.57 vs v91 (new best)** | The v91 stack served on all 62 ranking days: `python -m src_v3.build_adsb_clock` read the points of every January and July 2026 day (v91: 24 July days); 60 % of ranking departures carry an ADS-B clock. WINNING_PLAN second edition A5: after MS1 and MS2, each row outside LIRF is floored at `min(adsb_own_last, adsb_push)`, the last point within 100 m of the assigned stand, or the first move after a later stop of 120 s or more at another stand (a tow). On 2025 labels 3.3 % of rows sit below this bound, 0.17 % by more than 300 s; hold-out -15.9 MSE on 155 lifted rows. Served: 202,073 rows moved by the stack, 1,184 lifted by the bound (2 rows by more than 3,000 s, both EDDM rows at the stand until 5,379 and 5,653 s before take-off). Label-free check vs v91: LIRF unchanged, rows above 7,200 s 106 (v91 106). Closed the same day: A3 (first-point lower bound, +4,875 MSE or worse), A4 (neighbour pushback clocks, +2 and +62 MSE), B1 (no Step A pattern outside LIRF). Run: `python -m src_v3.stack_pushback_v90 serve --sources both --stand-bound --base kind-mango_v89.parquet --out kind-mango_v92.parquet`; parity `python -m src_v3.test_parity --version 92`. |
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
| Iowa State ASOS | METAR reports, routine and special, per station; `python -m src_v3.fetch_metar` | Public domain |
| OurAirports | Runway coordinates, headings, lengths | Public domain |
| OSM Overpass | `aeroway=parking_position`, `taxiway`, `runway` | ODbL |
| Eurocontrol NM | Daily ATC pre-dep delay, ATFM slot adherence, airport traffic | Open (non-commercial-use clause; challenge use confirmed by organiser, 2026-09-17) |
| OPDI (PRC + OSN) | Flight events (entry-runway, entry-taxiway, exit-parking, …) | Open (CC-BY 4.0, confirmed by organiser) |
| VRS StandingData | Aircraft-type metadata | Open |
| X-Plane Scenery Gateway | `apt.dat` of EDDM (scenery 111124) and EHAM (111943): stand positions (row 1300) where OSM has no position for the assigned stand; `python -m src_v3.build_xplane_stands fetch` and `build` (v115 on). The `apt.dat` files are kept unchanged with their copyright header in `external/xplane/`. | GPL v2 or later |
| EUROCONTROL `atfm_slot_adherence` (ansperformance.eu CSV) | Departures, regulated departures and departures before, inside and after the slot window, per airport and UTC day, 2025 and 2026; `python -m src_v3.build_atfm_slot` (v115 on). Take-off times only, no off-block time. (c) EUROCONTROL. | Open, with attribution; same family as the daily files confirmed by the organiser, 2026-09-17 |
| EUROCONTROL `taxi_out_additional_time` (ansperformance.eu CSV) | Reference and additional taxi-out time per airport and month, 2025 and 2026; `python -m src_v3.month_level` (candidate v138) downloads the two files. The sum per departure is the monthly mean taxi-out, an aggregate of the labels. (c) EUROCONTROL. | Open, with attribution; the organisers confirmed the use for the scored months on Discord, 2026-10-08 |
| ADSB.lol `globe_history` | Daily ADS-B traces; `src_v3/fetch_adsb_days.py` keeps the low points near the 10 airports and deletes the raw archive. Used for the pushback clock of the scored flight (v91 on). Data (c) ADSB.lol contributors. | ODbL 1.0. The derived tables stay out of git (share-alike). |

## Quick start

Python 3.13, one virtualenv.

```bash
pip install -r requirements.txt   # pinned versions, Python 3.13

# 1. Fetch open external data (~60 min, several rate-limited APIs)
python src/build_eurocontrol_daily.py     # daily ATFM per airport
python src/build_osm_stands.py            # OSM stand centroids
python src/build_osm_taxi_paths.py        # OSM taxiway graph
python src/fetch_opdi_events.py           # OPDI ADS-B events (~50 min, 560 MB)
python src/build_opdi_taxi_v2.py          # OPDI taxi-out record aggregation
python -m src_v3.fetch_metar              # Iowa State ASOS METAR CSVs (~2 min)

# 2. Train the base regressor (3 members, linear_tree)
python src/train_r_all_v26.py

# 3. Train the LIRF regime head, then build the Step A band table
python src/train_lirf_regime.py           # LIRF feature list and LIRF-only encoders
python src/train_r_norm_lirf_seeds.py     # 5 R_norm_LIRF members, seeds 42-46
python src/train_lirf_regime_v23.py       # p_fb gate, fallback-rate maps, calibrator
python src/build_lirf_band_table_v30.py   # band table with exclusive classes

# 4. Predict on the ranking set
python src/train_r_all_v40.py             # base with the 13 tempo columns, paired report
python src/train_r_norm_lirf_v41.py       # LIRF regressor members with the 13 columns, paired report
python src/predict_v41.py kind-mango_v41.parquet

# 5. Run the harness checks (H1 to H5 of the fourteenth pass)
python src/build_h1_frame_cache.py        # cached frame with the movement id inside
python src/eval_v33_holdout.py --stack v41  # score any stack on the 2025 hold-out
python src/test_v33_parity.py             # rebuild v33 to 0.0 s; also writes the 2026 dump
python src/check_coverage.py --dump models/v33_rank_dump.parquet
python src/check_submission_2026.py submission/kind-mango_v41.parquet

# 6. Submit
python -c "\
from minio import Minio; \
c = {l.split('=')[0].strip(): l.split('=',1)[1].strip() for l in open('.osn_credentials.txt') if '=' in l}; \
Minio('s3.opensky-network.org', access_key=c['access_key'], secret_key=c['secret_key'], secure=True) \
    .fput_object(c['bucket'], 'kind-mango_v33.parquet', 'submission/kind-mango_v33.parquet')"
```

Full reproduction steps are in [`REPRODUCE.md`](REPRODUCE.md).

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
  # v33 pipeline (current)
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
  predict_v33.py                # ranking submission (current best)
  price_ensemble.py             # label-free ambiguity price on the ranking set
  eval_v33_holdout.py           # v33 stack scored end to end on the 2025 hold-out (P1)
  train_r_all_v40.py            # tempo columns on the v33 base, paired against a control
  predict_v40.py                # v33 stack with the v40 base members
  train_r_norm_lirf_v41.py      # LIRF regressor members with the tempo columns, paired
  train_p_fb_lirf_v41.py        # LIRF gate with the tempo columns, paired (rejected)
  features_order_next.py        # forward take-off order column (v42, set aside)
  train_r_all_v42.py            # forward order on the base, paired (set aside)
  predict_v41.py                # current best

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

src_v3/                         # current pipeline (v48 on), constants in config.py
  predict_v67.py                # base outside LIRF and the LIRF head; release_v97.py builds the best file
  release_final.py              # final phase: `tables` rebuilds the ranking-side tables, `serve` writes kind-mango_final.parquet (v132 recipe)
  build_gate_g3.py              # column table of the LIRF gate g3: G1 shares, v83 arrival link, operator shares
  stack_pushback_v90.py         # OPDI and ADS-B pushback stack and the A5 stand bound
  build_adsb_clock.py, build_opdi_own.py   # pushback clocks of the scored flight
  measure_budget_v97.py         # v97 error budget per class, airport and ADS-B coverage; 2026 projection
  measure_lirf_gap.py           # LIRF gate ceiling and the arrival-artefact signal (WINNING_PLAN third edition)
  measure_nm_join.py            # J1, J2: hold-out price of the LOBT window rules on the v115 path (nineteenth pass)
  lobt_window.py                # J1 to J3 serve steps: LIRF copy tests for the gate, J3 anchor, LOBT window last (v118 on)
  mvt_precision.py              # P1: LFPG departures with MVT off the minute grid, raised to MVT - LOBT minus 242 s (v121)
  month_level.py                # June and July values of six airports shifted to the EUROCONTROL monthly mean (candidate v138)
  build_lobt_terms.py           # unclipped MVT - LOBT and SCHED - LOBT for the TabM member tabm_j (v120)
  build_arr_artefact.py         # G1: airline arrival artefact shares for the LIRF gate (v101 on)
  train_p_fb_v83.py             # LIRF gate refit with added columns, `--tag g3 --extra ...` (v104, v105)
  catboost_gate.py              # N7: CatBoost member of the LIRF gate, `price` and `fit --tag g3c` (v109)
  blend_nnls.py                 # N3: NNLS member weights per coverage class, `price` and `serve` (v110)
  tabm_member.py                # TabM member on the GPU, `measure`, `serve` and `combine` (v111)
  step_a_date.py                # Step A without the 24-h term where the local dates do not cross
  measure_pseudo_n5.py, measure_member_swap.py, probe_adsb_feeds.py, build_route_clock.py   # closed levers N5, K3, K1

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
.gpu_site/                      # CUDA build of torch and the TabM packages for tabm_member.py (gitignored)
docs/                           # PRC brief + MODEL_ANALYSIS.md (progress to v98, sixteenth pass: error budget of v97, fifteenth-pass findings) + WINNING_PLAN.md (third edition, 2026-10-03: ranked levers from 247.89 s toward 210 s, plan to 11 Oct; second edition in Appendix B) + AUDIT_PROMPT.md (brief of the sixteenth pass)
RECAP.md                        # session log for every attempt, submissions and scores
requirements.txt                # pinned library versions of the v92 pipeline
```

## Approaches tried and RULED OUT

For transparency, and so the next replicator does not waste days on the same
dead-ends:

- **ADS-B specialist as a stack column (v96)** — -302 MSE on the 2025 hold-out,
  +0.53 s live. Derived columns did not transfer this season (v66, v71, v83, v96);
  new row-level observables did (v73, v90 to v95).
- **Lower bound from the first ADS-B ground point (WINNING_PLAN A3)** — the rule
  `max(pred, MVT - first moving point + k)` costs +4,875 to +7,954 MSE on the
  2025 hold-out for every `k`. The first point often moves before the reported
  off-block (41 % of EHAM rows), and where the stack predicts below the bound
  the label also sits below it on 69 % of rows. The stack already reads the
  first-point columns. The stand bound of A5 replaces it.
- **Neighbour pushback clocks (A4)** — the mean and count of other departures'
  OPDI and ADS-B pushback taxi over the v85 windows: +2 MSE as stack columns,
  +62 MSE when rows with neighbours but no own clock join the stack
  (`python -m src_v3.stack_pushback_v90 price --neighbours cols|rows`).
- **Three seeds for the per-airport members (B2)** — -5.2 MSE inside the v89-style
  blend on the 2025 hold-out, airports split both ways
  (`python -m src_v3.measure_per_airport_v89 --seed 43`, then `--seed 44`).
- **Local hour, airport-scoped stand and runway, drift flags (MF3, MD5, MD6)** — as
  columns of a residual stack on all hold-out rows, cross-fitted January against
  July: +723 to +1,267 MSE against +535 for the same stack without them.
- **Step A analogue outside LIRF (B1)** — outside LIRF, 0 of 412 rows of 2025
  with a null flight record and `sd > 14,400` sit within 60 s of `sd`
  (1 of 4,313 above 7,200 s). A rule would move 278 rows of 2026 with no support.
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

See [`REPRODUCE.md`](REPRODUCE.md). Section 8 builds the final file with two commands
(`python -m src_v3.release_final tables`, then `serve`) from the artefacts of `models/v132.manifest.json`.
Tier 1 serves v92 from the trained artefacts;
Tier 2 retrains every served artefact, and step 3 of the guide fetches the
open data, ADSB.lol days included. `python -m src_v3.test_parity --version 92`
rebuilds v92 and checks it against the recorded hash in
`submission/kind-mango_v92.parity.json`; `--version 67` does the same for v67.

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
