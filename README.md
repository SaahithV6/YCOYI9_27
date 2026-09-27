# Scrubline

**Pick a rocket and a time. Get a go/no-go with receipts.**

## The short version (read this first)

Launch pads on the Space Coast have shown they can fly 143 times a year. They fly 73.
Since 2020 a Cape launch has missed its announced date **1,502 times** (1,216 slips before
the countdown, 286 scrubs during it). For **1,235 of those (82%) nobody published why**, so
nobody learns from them. There isn't even an agreed definition of what counts as a scrub,
a slip, or a launch that went fine.

We're building a **launch weather officer**: you choose a date, a time and a rocket; it
checks the real launch rules against real weather data and the Space Force forecast, pulls
up every past failure that looked like this one, scores the risk with a small non-LLM model,
and says **GO / NO-GO / UNDECIDED and why**, citing its evidence. When the team corrects it,
the correction is remembered, and the next answer is better.

**For the feasibility check, the questions are:**
1. Is the demo below buildable in 3h45m with the pieces marked "have"?
2. Do we trust the enrichment numbers in section 3 (they are measured, see how)?
3. Which sponsor pieces do we actually wire up: QM, GBrain, River (Memorable optional)?

## 1. The demo (MVP)

1. User sets **date, time, rocket** (and pad).
2. The system gathers **weather** for that moment: live forecast (and the Space Force
   forecast if the launch is within 5 days), or real observations for a past date.
3. It **checks the launch rules**: the 10 Lightning Launch Commit Criteria plus the risks
   the Space Force forecast deliberately leaves out (upper-level wind shear, solar activity,
   recovery weather). A rule with no real measurement says `UNAVAILABLE`, never "fine".
4. It **recalls past failures** with the same pad, rocket or conditions, and their causes.
5. A **non-LLM model** scores the chance of a scrub or slip and lists the top factors.
6. The agent writes **GO / NO-GO / UNDECIDED** with reasons, each one cited. GO requires
   that something was actually decided; otherwise it's UNDECIDED.
7. The team **corrects** it in Slack; the correction goes to memory and the model retrains.

Visual: a CesiumJS globe (the engine GeoFS runs on; GeoFS itself can't be embedded) showing
the pad, the ascent corridor, real lightning flashes and cloud layers at their altitudes.

## 2. Stack and who does what

| Layer | Tool | Job | Status |
|---|---|---|---|
| Data | scrubline | 1,502 non-launch events, METAR / lightning / radar / upper-air per attempt, lightning-rule engine, launch windows | **have** |
| Space Force forecasts | 45th Weather Squadron | Past pre-launch forecasts (probability of violating weather constraints, primary concerns, shear/solar/recovery ratings) | **have**: 465 forecasts parsed so far, 2020-2026 |
| Next-week weather | Open-Meteo + NOAA SWPC | 7-day hourly forecast at CCSFS and KSC incl. 11 pressure levels (winds, temps, heights, shear), freezing level, CAPE, cloud cover; 3-day solar Kp | **have** |
| Prediction | Gradient-boosted trees | Chance of scrub/slip + top factors. Milliseconds, zero tokens | to build |
| Memory | **GBrain** | Every attempt as an entity with sourced facts; lessons when a prediction was wrong; precedents for explanations | to wire |
| Agent + team | **QM** | Launch weather officer in Slack/web; feedback in the thread; nightly retrain | to wire |
| Cause labeler | **River** | Post-train a small open model on evidence-verified causes, label the unknowns (stays "predicted" until confirmed) | stretch |
| Token savings | Memorable | Reuses the agent's own brief-building procedure (optional) | optional |

GBrain is good at shared memory with a source on every fact, links between launches /
boosters / pads / causes, and corrections. It is not a predictor and not a numeric
similarity search; the feature table and model do that.

## 3. Enrichment: why did these launches not go?

Every cause carries a grade saying how direct the evidence is. Nothing below overwrites a
label; these are **candidates** for a person or the River labeler to confirm.

| Pass | What it checks | Validated? | Unknowns it covers |
|---|---|---|---|
| INSTRUMENT | Weather rules fired on real observations at the pad | Yes: fires on 35% of known weather scrubs vs 10% of known non-weather events | 113 |
| CARRY | Same mission's previous event (within 7 days) has a stated cause | Logical continuation | 68 |
| FORECAST | Space Force forecast exists for that mission/date; its probability is attached | Yes: median 60% before known weather scrubs vs 20% before non-weather ones | 40 |
| ~~PAD~~ | Another launch from the same pad within 48 h | **Rejected**: fires 38% on known vehicle/weather events vs 40% on unknowns (no signal) | - |
| ~~RANGE~~ | Another scrub at the Cape in the previous 24 h | **Rejected**: 14% vs 13% (no signal) | - |

**Result: 196 of 1,235 unexplained events now have an evidence-backed candidate cause
(63 of 132 scrubs, 133 of 1,103 slips). 1,039 have no evidence in any data we hold.**
Those need the research pass (webcast recaps, operator posts, press) run by agents, then
the River labeler. Earlier webcast work closed 19% of unknowns, and routine Starlink slips
barely close at all, so we report the close rate rather than promise one.

## 4. The prediction model

- **Features:** pad, vehicle, provider, booster flight count; hour and month; ceiling,
  CAPE, freezing level, nearest lightning, radar echo tops; Space Force launch-day
  probability of violation; max wind shear; hours since the pad last flew; same-range
  traffic (the rejected PAD/RANGE signals are still legitimate model features).
- **Evaluation:** train 2021-2024, test 2025-2026; launches that flew on time are the
  negative class. There is no scrub-reason record before 2021.

## 5. Build order (from 1:15)

1. Feature table: join events + Space Force forecasts + weather (**mostly done**).
2. Gradient-boosted baseline with factor explanations, 45 min.
3. Load attempts into GBrain as sourced facts, 30 min.
4. QM agent: the brief + feedback capture, 60 min.
5. CesiumJS view, 30 min (parallel, second person).
6. Demo run: real upcoming launch, a teammate corrects it, rerun cites the lesson.
7. Stretch: River cause labeler; Memorable config.

## Ground rules

- **No proxies, no invented thresholds.** Missing measurement = `UNAVAILABLE`, named.
- **Forecasts are never graded as observations.** Predictions are never stored as labels.
- **Report what closes.** Measured numbers only, including the ones that don't flatter us.
