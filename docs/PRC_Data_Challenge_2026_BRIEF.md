# PRC Data Challenge 2026 — Agent Brief

Source of truth: https://ansperformance.eu/study/data-challenge/dc2026/  
Data spec: https://ansperformance.eu/study/data-challenge/dc2026/data.html  
Ranking / submit: https://ansperformance.eu/study/data-challenge/dc2026/ranking.html  
Rationale: https://ansperformance.eu/study/data-challenge/dc2026/rationale.html  
Eligibility: https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html  

This file is written for coding agents. Prefer these facts over assumptions. Official pages can change; re-check them if something conflicts.

---

## 1. One-sentence task

Predict **taxi-out time in seconds** (`TAXITIME_SEC_mvt`) for **departing flights** (`PHASE_mvt = "DEP"`) at **11 major European airports**, using open data, then submit a parquet file scored by **RMSE**.

## 2. Goal (why this exists)

Organiser: EUROCONTROL Performance Review Commission (PRC), with OpenSky Network (OSN).

Taxi-out is hard to obtain and hard to predict. A good model supports post-ops analysis: find congested periods and estimate excess fuel / CO2 vs unconstrained taxi.

Variability drivers called out by organisers:

- airline-specific constraints
- airport procedures and load
- ATFM (Air Traffic Flow Management)

Challenge ethos (required for prizes, expected for a valid entry):

1. Use **open data** only.
2. Publish **code + docs** on a public GitHub repo under **GNU GPLv3**.
3. Ideally write an open-access paper in JOAS.

This is **not** a closed Kaggle-style private-model contest. Reproducibility is part of the product.

## 3. What to predict vs what not to predict

| Item | Spec |
|---|---|
| Target | `TAXITIME_SEC_mvt` |
| Unit | seconds |
| Rows scored | departures in the ranking set (`PHASE_mvt = "DEP"`) |
| Airports | 11 ICAO codes listed below |
| Train period | calendar year **2025** movements at those airports |
| Test / ranking period | **January 2026** and **July 2026** movements |
| Metric | RMSE on submitted taxi-out seconds |
| Ranking rule | **best RMSE across a team's submissions** |

Arrivals (`PHASE_mvt = "ARR"`) are in the data and have taxi-**in** times. They are **not** the submission target. Use them as context (queue, runway load, inbound pressure), not as labels to submit.

### Operational meaning of the target

For a departure, taxi-out is the time from **off-block** to **take-off**.

In this schema that is consistent with:

```text
TAXITIME_SEC_mvt  ≈  MVT_TIME_UTC_mvt  −  BLOCK_TIME_UTC_mvt
                     (take-off)            (off-block)
```

On the ranking set, organisers **blank**:

- `BLOCK_TIME_UTC_mvt` for `PHASE_mvt = "DEP"`
- `TAXITIME_SEC_mvt` for `PHASE_mvt = "DEP"`

So a model **cannot** compute the label from timestamps at inference. It must estimate taxi-out from other features.

`MVT_TIME_UTC_mvt` (take-off for DEP) is still present. Treat it as available context unless you later find it missing in the files. Do **not** assume actual off-block is available on test DEP rows.

## 4. Airports in scope

Total movements in the published corpus: **4,167,797**.

| ICAO | IATA | Airport |
|------|------|---------|
| EDDF | FRA | Frankfurt Main |
| EDDM | MUC | Munich |
| EGLL | LHR | London Heathrow |
| EHAM | AMS | Amsterdam Schiphol |
| LEBL | BCN | Barcelona-El Prat |
| LEMD | MAD | Madrid–Barajas |
| LFPG | CDG | Paris Charles de Gaulle |
| LIRF | FCO | Rome Fiumicino |
| LTAI | AYT | Antalya |
| LTFM | IST | İstanbul |
| LSZH | ZRH | Zürich |

Filtered out by organisers: military, head-of-state, and sensitive movements.

## 5. Files

Access is via OpenSky / MinIO buckets after team approval. 2024 MinIO notes: https://ansperformance.eu/study/data-challenge/dc2024/data.html#using-minio-client  
Generate OSN access keys for the team account.

### Training (labels present)

Monthly parquet, movements at the 11 airports:

- `training_2025-01-01_2025-02-01.parquet` (21M)
- `training_2025-02-01_2025-03-01.parquet` (19M)
- `training_2025-03-01_2025-04-01.parquet` (22M)
- `training_2025-04-01_2025-05-01.parquet` (23M)
- `training_2025-05-01_2025-06-01.parquet` (25M)
- `training_2025-06-01_2025-07-01.parquet` (24M)
- `training_2025-07-01_2025-08-01.parquet` (25M)
- `training_2025-08-01_2025-09-01.parquet` (25M)
- `training_2025-09-01_2025-10-01.parquet` (24M)
- `training_2025-10-01_2025-11-01.parquet` (25M)
- `training_2025-11-01_2025-12-01.parquet` (22M)
- `training_2025-12-01_2026-01-01.parquet` (22M)

### Ranking / inference features

- `ranking.parquet` (27M)
- Same columns as training.
- Jan + Jul 2026 movements.
- For `PHASE_mvt = "DEP"`: `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` blanked.

### Submission template

- `submitting.parquet` (1.1M)
- Columns only: `MVT_ID_mvt`, `TAXITIME_SEC_mvt`
- `MVT_ID_mvt` values = departures in the ranking set.

## 6. Schema

Movements were **left-joined** to Network Manager (NM) flight records. Real-world mismatches exist; organisers did **not** reconcile them. Handle nulls and ADEP/ADES disagreements.

Suffix `_mvt` = reporting airport movement.  
Suffix `_flt` = NM flight list.

### Movement columns (`_mvt`)

| Column | Meaning |
|---|---|
| `MVT_ID_mvt` | Unique movement id. Join key for submission. |
| `FLIGHT_ID_mvt` | NM flight id if matched |
| `FLIGHT_mvt` | Flight number as on boarding pass |
| `FLIGHT_RULE_mvt` | `I` IFR, `V` VFR, `NA` unknown |
| `ADEP_mvt` | ICAO departure aerodrome |
| `ADES_mvt` | ICAO destination aerodrome |
| `PHASE_mvt` | `DEP` or `ARR` |
| `MVT_TIME_UTC_mvt` | Best available movement time: take-off if DEP, landing if ARR |
| `BLOCK_TIME_UTC_mvt` | Off-block if DEP, in-block if ARR. **Hidden on ranking DEP.** |
| `SCHED_TIME_UTC_mvt` | Scheduled departure if DEP, scheduled arrival if ARR |
| `AIRCRAFT_TYPE_mvt` | ICAO type (e.g. `A21N`) |
| `RUNWAY_mvt` | Departure runway if DEP, arrival runway if ARR |
| `STAND_mvt` | Stand id |
| `TAXITIME_SEC_mvt` | Taxi-out seconds if DEP, taxi-in seconds if ARR. **Hidden on ranking DEP.** |

### Flight columns (`_flt`)

| Column | Meaning |
|---|---|
| `LOBT_flt` | Last known off-block time |
| `CALLSIGN_flt` | Callsign |
| `ADEP_flt` | NM departure ICAO |
| `ADES_flt` | NM destination ICAO |
| `ADES_FILED_flt` | Originally filed destination (diversions can differ) |
| `MARKET_SEGMENT_flt` | Mainline, Regional, Low-Cost, Business Aviation, All-Cargo, Charter, Military, Other, Not classified |
| `IOBT_flt` | Initial off-block time |
| `FLIGHT_RULE_flt` | `I` IFR, `V` VFR, `Y` IFR then VFR, `Z` VFR then IFR |
| `FLIGHT_TYPE_flt` | `S` scheduled, `N` non-scheduled, `G` GA, `M` military (filtered), `X` other |
| `AIRCRAFT_TYPE_flt` | ICAO type |
| `WK_TBL_CAT_flt` | Wake: `L` Light, `M` Medium, `H` Heavy, `J` Super |
| `AIRCRAFT_OPERATOR_flt` | **Anonymized** ICAO airline designator |
| `EOBT_1_flt` | Estimated off-block, FPL-based (M1) trajectory |
| `ARVT_1_flt` | Arrival time, FPL-based (M1) |
| `AOBT_3_flt` | Actual off-block, flown (M3) trajectory |
| `ARVT_3_flt` | Arrival time, flown (M3) |

## 7. Critical modelling constraints (read before coding)

1. **Inference-time leakage.** Do not train a model that needs `BLOCK_TIME_UTC_mvt` or the label at test time. Those two fields are blank on ranking DEP rows.

2. **Check `_flt` off-block fields on ranking data.** Organisers only document blanking `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`. `LOBT_flt` and `AOBT_3_flt` may still be present, null, or leaky. Inspect ranking.parquet before using them. If they reconstruct taxi-out almost exactly, using them is probably exploiting the ranking process (explicitly forbidden). Prefer features that would exist **before or without knowing actual off-block**.

3. **What a realistic model may use**
   - airport, stand, runway, aircraft type / wake
   - scheduled time, planned off-block (`IOBT_flt`, `EOBT_1_flt`)
   - airline proxy (`AIRCRAFT_OPERATOR_flt` is anonymized but stable)
   - market segment, flight type, city-pair
   - concurrent demand: other DEP/ARR around the same airport and time
   - inbound/outbound runway configuration
   - calendar: hour, weekday, month, season, holiday
   - weather and other **open** external data (METAR/TAF, airport layouts, etc.) if documented and openly licensed

4. **Arrivals are context.** Ranking file includes ARR rows with taxi-in and block/landing times. Those can describe airport state. Do not submit ARR ids.

5. **Messy joins.** `ADEP_mvt` vs `ADEP_flt`, type codes, and times can disagree. Do not drop all mismatched rows blindly on train if the same mess exists on test.

6. **Target scale.** Unit is seconds. Typical taxi-out is hundreds to a few thousand seconds. Clip only with evidence; RMSE punishes large errors.

7. **No synthetic data from organisers.** All rows are real.

8. **Exploit-the-leaderboard is banned.** “We will monitor submissions for attempts to learn from or exploit the ranking process.”

## 8. Metric and submission contract

RMSE over submitted taxi-out seconds vs hidden labels:

```text
RMSE = sqrt( mean( (y_hat - y)^2 ) )
```

Lower is better. Team rank = best RMSE among that team's valid submissions.

### File to upload

Name:

```text
<team-name>_v<incremental integer>.parquet
```

Example: `adventurous-bicycle_v3.parquet`

Upload to the team's own bucket (same MinIO / OSN pattern as prior years).

Payload columns:

- `MVT_ID_mvt` — exact ids from `submitting.parquet`
- `TAXITIME_SEC_mvt` — predicted seconds

Validation failures (submission rejected / not scored):

- `MVT_ID_mvt` mismatch
- missing rows
- extra rows

Practical implementation rules:

- Start from `submitting.parquet`; do not rebuild the id list.
- Keep row count and id set identical.
- Predictions must be numeric, finite, and in seconds.
- Prefer the same parquet engine/schema as the template.

Leaderboard API (as published on the ranking page):

- competitionId: `bb3693e1-26bc-4a9e-8619-4fe78b4eab0c`
- paginated rankings JSON is linked from the ranking page
- Observable leaderboard widget was broken after 1 Sep 2026 framework migration; use the API if the widget is down
- news: OSN Discord `#prc-data-competition`

## 9. Timeline, prize, licence

- Competition window stated on the site: September 2026 through **end of October 2026, 23:59:59 CET**. Exact start day on the homepage is incompletely rendered (“the of September 2026”); treat October 31 23:59:59 CET as the published close unless Discord/site updates it.
- Combined prize for top 3 teams: **5000 EUR**.
- Prize-eligible solutions must:
  - use only openly accessible, documented extra datasets
  - publish source under **GNU GPLv3**
  - document enough to reproduce
  - be original (re-use allowed only with rights + significant changes; swapping I/O on someone else's model is not enough)
- Sanctioned / ineligible teams: see eligibility page. Participation is broadly open; prize transfer is restricted.

Previous editions (different targets):

- 2024 takeoff weight — https://github.com/prc-data-challenge-2024
- 2025 fuel burn — https://github.com/prc-data-challenge-2025

## 10. Official links

- Home: https://ansperformance.eu/study/data-challenge/dc2026/
- Data: https://ansperformance.eu/study/data-challenge/dc2026/data.html
- Ranking: https://ansperformance.eu/study/data-challenge/dc2026/ranking.html
- Teams / leaderboard notebook: https://observablehq.com/@espinielli/dc26-leaderboard
- Team request form: https://docs.google.com/forms/d/e/1FAIpQLScgRRk0j5Giot8puUAjzXC7ScR926Oupd62LbRVS1g8Y2p4hw/viewform
- JOAS: https://journals.open.tudelft.nl/joas/
- OSN: https://opensky-network.org/
