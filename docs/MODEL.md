# Model description

This document describes the model of the final file, `kind-mango_final.parquet` (recipe v139). It gives the
parts of the model, the data that each part reads, the fit of each part and the reason for each design
decision. `README.md` has the short summary. `REPRODUCE.md` has the commands. `RECAP.md` has the price and the
live score of each upload.

## 1. Task

The model predicts the taxi-out time of a departure in seconds, `TAXITIME_SEC_mvt`. The label is the take-off
time minus the off-block time that the airport reports. The metric is the RMSE. The data cover 10 airports:
EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM.

| Set | Months | Departures | Use |
|---|---|---|---|
| Training files | 2025-01 to 2025-12 | 2,085,047 (2,062,577 with an NM record) | Fit and hold-out |
| Ranking file | 2026-01 and 2026-07 | 344,841 | Live leaderboard |
| Final file | 2026-01, 02, 06 and 07 | 670,790 | Final rank |

The model works after the flight. It reads the take-off time and each record that exists after the take-off.
It is not a forecast before the off-block.

## 2. Two kinds of label

The label is a report, not a measurement. This fact sets the structure of the model.

| Class | Rule | What it is |
|---|---|---|
| Clean | 30 s to 7,200 s, not on the schedule | A real taxi-out |
| Fallback | Label within 60 s of `take-off - schedule` | The reported off-block is the scheduled time: a punctual pushback, or a copy of the schedule |
| Tail | 7,200 s to 80,000 s | A long hold, a return to the stand or a wrong report |
| 24 h | Above 80,000 s | An off-block with the wrong date |

A copy of the schedule is harmless on a punctual flight and costs hours on a late one. Rome Fiumicino (LIRF) has
8 % of the 2025 departures, but 480 of the 584 labels above 7,200 s, 428 of the 432 schedule copies above
7,200 s and 14 of the 16 labels above 80,000 s. So the model has two branches:

- **Outside LIRF** the model estimates a real taxi-out. The best information is a clock that saw the aircraft
  leave the stand.
- **At LIRF** the model first estimates the probability that the airport copied the schedule. It then mixes the
  schedule delay with a normal taxi estimate.

The squared error gives the long labels a high weight. On the ranking file, LIRF and the labels above 7,200 s
make 48 % of the MSE. The rows without a ground track make 28 % (`README.md`, "Where the error sits").

## 3. Validation

- **Hold-out.** January and July 2025 (344,336 departures) are the hold-out. They are the months of the
  ranking file, one year earlier. Each price in this repo is an MSE change on this scale.
- **Fit months.** A member fits on the other months and stops early on November and December. The served fit
  then uses the 12 months and the round count of the hold-out fit, scaled by the row ratio.
- **Bar.** The default bar for an upload is -150 MSE on the hold-out (`AGENTS.md` section 11).
- **Gate folds.** The LIRF gate uses six folds. Each fold holds out one winter month and one summer month.
- **Live score.** The leaderboard judged a whole recipe, one change per upload. No row value comes from a score.

The hold-out is not exact. Two tree changes with a good price gave no live gain (v68, v71). The neural members
gave two to four times their price. From 2026-10-08 the team judged some uploads on the live score alone
(`RECAP.md` marks them).

## 4. Pipeline

```
organiser files + open data
        |
   19 column tables (158 columns)
        |
        +--> tree members: LightGBM x3 seeds, LightGBM per airport, CatBoost; plain and anchored
        +--> LIRF head: gate + normal model + band table
        +--> neural members: MLP, five TabM families
        |
   blend outside LIRF: 0.28 trees + 0.72 neural
        |
   pushback stack on (label - blend), rows with a ground track; then the stand bound
        |
   month level: one shift per airport and month (June and July, seven airports)
        |
   three row rules: Step A date, LOBT window, LFPG take-off precision
        |
   kind-mango_final.parquet
```

`python -m src_v3.release_final tables` builds the tables. `python -m src_v3.release_final serve` runs the
other steps from the saved fits.

## 5. Columns

The tree members read 158 columns (`models/lgbm_r_all_v133.features.txt`).

| Group | Columns | Content | Source |
|---|---|---|---|
| Categorical | 10 | Airport, destination, runway, stand, aircraft type, wake class, market segment, operator, flight rule, flight type | Organiser |
| Calendar | 3 | Hour, weekday, month | Organiser |
| Schedule and plan | 17 | Take-off minus schedule, minus `EOBT_1`, minus `IOBT`; their signed logs; flags for a missing record; planned-taxi terms from the filed block time | Organiser |
| NM clocks of the flight | 5 | `MVT - AOBT_3`, `MVT - LOBT` and their differences to the plan | Organiser (`build_nm_clocks_v73.py`) |
| Clocks of later departures | 8 | Mean and count of `MVT - AOBT_3` of the departures that take off in the next 15 and 30 min, per airport and per runway | Organiser (`build_nm_forward_v85.py`) |
| Neighbour plan | 18 | Median, mean, quartiles and count of `MVT - EOBT_1` of the departures near in time; the order in the queue | Organiser |
| Traffic | 22 | Departures and arrivals in the last 15, 30 and 60 min, per airport and per runway; taxi-in of recent arrivals | Organiser |
| Turnaround | 5 | Ground time, delay and taxi-in of the arrival that used the stand before | Organiser |
| Operator encoders | 16 | Median, count and spread of the 2025 taxi-out per operator, airport, runway, hour bin and stand | Training labels |
| Weather | 18 | Wind and its runway components, visibility, ceiling, temperature, dew point, precipitation and ice flags | Iowa State ASOS (METAR) |
| Geometry | 9 | Stand and threshold position, straight distance, length and turns of the shortest taxi path | OSM, OurAirports |
| OPDI clocks of the flight | 4 | Runway entry and stand exits of the flight | OPDI (`build_opdi_own.py`) |
| ADS-B clocks of the flight | 23 | Ground segment before the take-off: its start, the first move, the pushback, the stops, the last point at the assigned stand | ADSB.lol (`build_adsb_clock.py`) |

The pushback clocks need three steps:

1. `fetch_adsb_days.py` keeps the points below 5,000 ft in a box of about 9 km around each airport.
2. `build_adsb_clock.py` links a departure to an aircraft. It uses the NM callsign, then the flight number, then
   the lift-off time and the aircraft type. Each fallback added coverage (v127, v130).
3. The builder walks back from the take-off through the ground points. A silence above 300 s ends the segment.
   From v133 a segment can end up to 3,600 s before the take-off, because the receivers lose aircraft on the
   last part of the taxi.

A stand position comes from OSM. Where OSM has no node for the assigned stand, the position comes from the
X-Plane Scenery Gateway (five airports, `build_xplane_stands.py`).

## 6. Tree members

Each tree member fits on the 2025 departures outside LIRF with a label from above 0 to 80,000 s: 1,923,953
rows. LIRF and the 24-h labels are not in the fit. This decision gave -5.99 s on the live score (v51): the base
does not serve those rows, and their labels pulled the trees away from the taxi-out.

| Member | Library | Fit | Parameters |
|---|---|---|---|
| Global LightGBM, 3 seeds | LightGBM, linear leaves | 3,377 rounds | 613 leaves, learning rate 0.015, at least 515 rows per leaf, `linear_lambda` 15.1 (Optuna study v76, 24 trials) |
| LightGBM per airport, 9 airports | LightGBM, linear leaves | Rounds scaled by the row share of the airport | The parameters of the global member |
| CatBoost | CatBoost | 7,385 rounds | Depth 8, learning rate 0.08, ordered target statistics on the categorical columns |

Each class also has an **anchored** member. It learns `label - clip(MVT - AOBT_3, 0, 7,200)`. The NM off-block
`AOBT_3` is a clock with minute resolution and an airport bias, so the anchored member learns the correction of
the clock, and the plain member learns the taxi-out. The two have different errors. The anchored members fit on
the 1,903,118 rows with `AOBT_3`. A row without `AOBT_3` keeps the plain member.

The tree blend outside LIRF, with `g` the global LightGBM mean and `a` the per-airport member:

```
lgb_plain    = 0.5 * g_plain    + 0.5 * a_plain
lgb_anchored = 0.5 * g_anchored + 0.5 * a_anchored
lgb          = 0.5 * lgb_plain  + 0.5 * lgb_anchored
cat          = 0.5 * cat_plain  + 0.5 * cat_anchored
trees        = 0.5 * lgb        + 0.5 * cat
```

Two guards protect the blend. Where the three seeds differ by more than 3,600 s, the blend takes their median
(MS3). Where the per-airport member differs from the global member by more than 3,600 s, the blend keeps the
global member. After the blend, a bound per airport caps a value above the 2025 range of the airport (MS1), and
a floor per airport sits at the 0.1 % quantile of the 2025 clean labels (MS2).

The weights are 0.5 by decision. A grid search of the three weights and an NNLS fit gave no live gain (v98,
v110).

## 7. LIRF head

At LIRF a part of the departures has the schedule as the off-block. Then the label is the schedule delay
`sd = take-off - schedule`. The head has four parts (`src_v3/predict_v67.py`, `src/predict_v30.py`).

1. **Gate.** A LightGBM classifier gives the probability that a row is a fallback row. An isotonic map
   calibrates it on out-of-fold rows. The gate reads 177 columns: the base columns, the EUROCONTROL daily
   counts, the OPDI ground activity, fallback rates per operator, stand and destination, and three arrival
   signals. The arrival signals are the strongest: an airline that reports its arrivals on the schedule also
   reports its departures on the schedule (correlation 0.77 over the airline months of 2025). The ranking file
   keeps the arrival block times, so the 2026 shares exist. No departure label of 2026 enters.
2. **Normal model `R_norm`.** Five LightGBM seeds with linear leaves fit on the LIRF rows that are not fallback
   rows (label below 80,000 s), on the 12 months.
3. **Value.** `p * sd + (1 - p) * R_norm`.
4. **Step A.** A row without an NM record and with `sd` above 14,400 s takes its value from a table with 11
   bands of `sd`. Each band holds the 2025 shares of the fallback, the normal and the 24-h labels.

One fact sets the gate to 0. The organiser joined a movement to its NM flight only if the block time lies
within 3,606 s of `LOBT` (each 2025 row with a record obeys this). So a schedule copy is impossible where the
schedule lies farther from `LOBT` (`lobt_window.py`, `--copy-window`). `--copy-linked` applies the same window to
a departure without a record that links to its arrival row at another of the ten airports.

The value of a LIRF row is a hedge between two outcomes that differ by hours. This is the correct answer under
a squared error, but the error stays high. Each later change of the head lost on the live score (CatBoost gate
+0.89 s, neural gate +3.12 s, gate with ground clocks +0.67 s).

## 8. Neural members

Trees share their errors. A neural member learns a smooth function of the same columns, so it adds new
information to the blend.

| Member | Structure | Fit |
|---|---|---|
| MLP, 3 seeds | Hidden layers 1,024, 512 and 256; batch norm, SiLU, dropout 0.1; one embedding per categorical column; numeric columns as signed log and z-score with a flag for a missing value | 18 epochs, AdamW with a one-cycle schedule, learning rate 0.002, MSE loss |
| TabM, 5 families, 3 seeds each | 16 sub-models with shared weights, 2 blocks of width 512; numeric columns as embeddings on 32 quantile bins | 27 or 30 epochs of 32, learning rate 0.001 |

The five TabM families differ in two ways: the numeric embedding (piecewise linear or periodic) and the column
set (the base columns; plus the unclipped `LOBT` terms; plus four terms that describe how the row was
recorded). TabM is the model of Gorishniy et al. (ICLR 2025, package `tabm`).

```
neural = 0.35 * MLP + 0.65 * mean of the five TabM families
blend  = 0.28 * trees + 0.72 * neural          (outside LIRF)
```

The neural weight grew from 0.15 (v94, the MLP alone) to 0.72 over the later uploads. LIRF keeps the head of section 7.

## 9. Pushback stack

The stack is the part that uses the ground track of the scored flight (`src_v3/stack_pushback_v90.py`).

- **Target.** `label - blend`, clipped at 3,600 s.
- **Rows.** The departures outside LIRF with an OPDI or an ADS-B clock: 216,792 of the 344,841 ranking rows.
  The other rows keep the blend.
- **Fit data.** The January and July 2025 hold-out, where the blend is out of sample. A stack that learns on
  rows that the members fitted on learns a residual that does not exist at serve time.
- **Model.** LightGBM, 15 leaves, 400 rounds, learning rate 0.03, at least 100 rows per leaf, L2 50. The model
  is small because it has two months of rows.
- **Columns.** The airport, each clock as its difference to the blend, the quality terms of the track (points,
  largest gap, distance of the first point to the stand), and two member gaps: neural minus trees, and TabM
  minus MLP.
- **Weights.** The 2025 hold-out has a different coverage mix than 2026 (EGLL had a stand clock on 4 % of the
  departures in 2025 and on 46 % in 2026). Each row gets the 2026 share of its airport and coverage class
  divided by its 2025 share, capped to the range 0.2 to 5. The weights read no label.

The mean move is 9.9 s and the 99th percentile is 185 s. A **stand bound** follows: the taxi-out is at least the
time since the last ADS-B point at the assigned stand. The bound lifts 930 ranking rows.

The stack is a separate step, and the clocks are also columns of each member. Both paid: the stack gave -14.0 s
over five uploads (v90 to v93 and v95), and the clocks as member columns gave -1.5 s more (v97).

## 10. Month level

EUROCONTROL publishes the reference and the additional taxi-out time per airport and month. Their sum per
departure is the mean taxi-out of the month. This is an aggregate of the labels of the scored months. The
organisers confirmed its use (Discord, 2026-10-08).

`src_v3/month_level.py` moves the values of one airport and month by one shift:

```
shift = published mean + gap of the airport - mean of the served values
```

The gap is the 2025 difference between the label mean up to 7,200 s and the published mean. The step moves a
month only if three conditions hold:

1. The month is April to October. In winter the file drops the long departures, and the gap grows to 164 s.
2. The leave-one-month-out error of the gap is at most 8 s. LTFM fails this test.
3. The share of the departures that the file keeps lies in the 2025 range of the airport. EDDF fails this test
   in 2026.

LIRF is not moved. At LFPG the gap is a line on the share of dropped departures. The step moves 7 airport
months of the ranking file and 14 of the final file, by -33 s to +32 s. It gave -0.30 s on the live score (v138
and v139). `release_final serve` without `--month-level` builds the model without this step (v137, 232.20 s).

## 11. Row rules

Three rules set single rows. Each follows from a fact of the 2025 data, not from a score.

| Rule | Fact | Rows moved in the ranking file |
|---|---|---|
| Step A date (`step_a_date.py`) | The 24-h label needs a date change between the schedule and the take-off | 1 |
| LOBT window (`lobt_window.py`) | The label lies within 3,606 s of `MVT - LOBT` on each 2025 row with an NM record | 59 |
| LFPG precision (`mvt_precision.py`) | At LFPG a take-off time off the minute grid marks an off-block near `LOBT`: median label 4,453 s against 1,191 s | 18 |

## 12. Design decisions

| Decision | Reason | Evidence |
|---|---|---|
| Fit the base on the rows that it serves | LIRF and 24-h labels pull the trees away from the taxi-out | v51, -5.99 s |
| Fit the served members on the 12 months | More rows are worth more than an early stop on held-out months | v47 |
| Two tree libraries | LightGBM and CatBoost encode the categorical columns in different ways | v67, -2.87 s |
| Use the NM clocks of the scored flight | The organisers allow `AOBT_3` and `LOBT`; `MVT - AOBT_3` alone scores 376 s, so it is a column, not the answer | v73, -11.3 s |
| Anchored members | A member that corrects the clock errs in other places than a member that learns the label | v76 |
| Clocks of later departures | The departures that take off next stood in the same queue | v85, v89 |
| Ground track of the scored flight | The off-block is a physical event; ADS-B and OPDI see it | v90 to v97, -15.5 s in six uploads |
| Neural members with a high weight | Their errors differ from the tree errors | v94 to v126 |
| Raise the coverage of the ground track | Each new link type moved rows from 217 s RMSE to 93 s or 165 s | v127 to v135, -9.2 s |
| A separate head for LIRF | The label there is a report regime, not a taxi-out | v33 |
| Equal blend weights | Fitted weights did not hold on the live score | v98, v110 |
| Month level in summer only | The published mean matches the labels from April to October only | v138, v139 |

Changes that failed are in `README.md` ("Approaches tried and RULED OUT") and in `docs/WINNING_PLAN.md`.
Three of them explain the structure:

- A base that learns only `label - anchor`, with the plan or the schedule as the anchor, lost 35.9 s (v54):
  the anchor is wrong by hours on delayed rows.
- Out-of-fold operator encoders lost 2.3 s (v60): the encoder at serve time must match the encoder at fit time.
- Rules for long labels outside LIRF priced +1,639 MSE: those labels are not predictable from the columns.

## 13. Limits

- **Coverage.** LTFM has no open ground track. LFPG has one on 18 % of its departures. EDDF has none from
  2026-01-29 to 2026-02-28 (10,367 February departures of the final file). The OPDI archive has no events for
  2025-10-02 to 2025-10-12, so the training rows of those days have no OPDI clock.
- **Stack data.** The stack learns on two months of 2025.
- **Airport-specific parts.** The LIRF head, Step A and the LFPG rule model how an airport fills its off-block
  time. They do not transfer to another airport.
- **Label aggregate.** The month level needs the EUROCONTROL file of the scored month.
- **Long labels.** A label above 7,200 s outside LIRF is not predictable from the columns. The final file holds
  160 values above 7,200 s.

## 14. Compute and determinism

- One full-frame fit of a tree member uses about 11 GB of RAM. The full retrain takes about 20 h on a 16-thread
  CPU, plus about 10 h on a GPU for the TabM families.
- The tree fits are deterministic: fixed seeds, a fixed thread count and the `deterministic` flag of LightGBM.
  `python -m src_v3.test_parity --version 92` rebuilds v92 bit for bit from the saved fits.
- The MLP refit on a CPU is deterministic. The TabM families fit on a GPU. The saved fits of both reproduce the
  served members at 0.0 s.
- `requirements.txt` pins each package. `models/v135.manifest.json` lists the 80 saved artefacts of the recipe
  with their hashes.
