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
3. Sponsors: GBrain (memory, precedents, corrections, scheduled refreshes) and River (cause labeler). QM was dropped: GBrain's hosted workspace already has an agent, chat and scheduled runs, so QM only added Slack.

## Why it matters: the cost of a launch that doesn't go

A launch that doesn't go costs more than the day. The range time is spent, crews and
propellant are stood down, the satellite customer's schedule moves, and the pad stays
occupied. That last cost is the one that compounds, because the next rocket can't use
the pad until this one leaves it.

The Space Coast averaged **73 launches a year** in 2021-2026. If every pad kept up
the pace it already manages in its better quarters (its own 25th-percentile gap between
launches), the same pads would fly **143** (a16z_throughput_numbers.md). That is a
pace each pad has already shown, not a sustained rate we promise, and the gaps include
planned downtime. Turnaround is what separates the pads:

| pad | fastest demonstrated turnaround, 2015-2026 | launches 2021-2026 | flights/yr at its own p25 pace |
|---|---|---|---|
| SLC-40 | 45 h | 284 | 92 |
| LC-39A | 126 h | 98 | 41 |
| SLC-41 | 498 h | 26 | 9 |

Sources: turnaround from ~/Documents/scrubline/README.md; launch counts and flights/yr from
~/Documents/jobs/a16z_throughput_numbers.md.

The limit is the pad, not the range. The range has cleared two launches from
different pads 2.9 h apart, but no pad has flown twice in under 45 h, and range-caused
stops are 6 of 223 stopped countdowns (a16z_throughput_numbers.md). So each avoidable
stand-down takes up hours on the operator's scarcest resource. And demand keeps growing:
the FAA expects licensed operations to double by 2029, with 14 licensed spaceports
(a16z_throughput_building_answer.txt). At higher cadence, one lost pad day delays more of
the launches queued behind it.

**What the money figures say, and what they don't.** The earlier pitch used $0.5-1M per
scrub, and up to $1.2M for a scrub after propellant loading. At about 7.5 day-of weather
stops a year across the whole coast (45 of the 105 countdowns stopped on the day,
2021-2026), that works out to a $4-8M/yr weather problem shared by every operator
(a16z_throughput_numbers.md, a16z_throughput_building_answer.txt). Those dollar figures
are the pitch's own estimates, and none of our datasets records their original source.
Read them as an order of magnitude, not a measured cost per scrub. We have no measured
market size. By the same record, weather alone is a small slice. The bigger costs are
turnaround and sequencing, and the 82% of misses that nobody explains, which means
nobody can plan around them.

**Where this project cuts the cost:**

- **Pick the window before committing the pad.** The hour-level weather-risk model scores
  2025-26 launches it never saw at ROC-AUC 0.735 and PR-AUC 0.42, against a 0.15 base rate
  (runs/weather-20260927-155215/metrics.json). The 7-day hourly risk view lets a scheduler
  choose a low-risk hour before the pad is tied up. Flexibility is the biggest lever we
  measured: a longer window plus a next-day recycle takes the weather-scrub risk from
  14.5% to 0.5% (a16z_throughput_numbers.md, from ~/Documents/scrubline/artifacts/mitigation.json).
- **Recover the cause.** 1,235 of 1,502 misses (82%) have no published cause (this
  README, section 3). Each one we explain with evidence gives operators, schedulers and
  insurers a case they can learn from, not just a blank.
- **Say what's missing before GO.** The rules check never marks an unmeasured rule as
  fine. It names the missing instrument (`UNAVAILABLE`), so a team knows before the count
  what it is flying blind on. Five of the ten launch commit criteria can't be evaluated
  from public data (a16z_throughput_numbers.md).

**Who it's for:** launch operators and range schedulers deciding which window to take;
satellite customers waiting on a ride, whose date depends on the launches ahead of them;
and insurers who need to know why a launch didn't go, not only that it didn't.

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
7. The user **corrects** it on the page; the correction is saved to GBrain as a sourced fact and the model retrains.

Visual: a CesiumJS globe (the engine GeoFS runs on; GeoFS itself can't be embedded) showing
the pad, the ascent corridor, real lightning flashes and cloud layers at their altitudes.

## 2. Stack and who does what

| Layer | Tool | Job | Status |
|---|---|---|---|
| Data | scrubline | 1,502 non-launch events, METAR / lightning / radar / upper-air per attempt, lightning-rule engine, launch windows | **have** |
| Space Force forecasts | 45th Weather Squadron | Past pre-launch forecasts (probability of violating weather constraints, primary concerns, shear/solar/recovery ratings) | **have**: 525 forecasts parsed, 2020-2026 (3 not retrievable from the archive, 1 image-only) |
| Next-week weather | Open-Meteo + NOAA SWPC | 7-day hourly forecast at CCSFS and KSC incl. 11 pressure levels (winds, temps, heights, shear), freezing level, CAPE, cloud cover; 3-day solar Kp | **have** |
| Prediction | Gradient-boosted trees | Chance of scrub/slip + top factors. Milliseconds, zero tokens | to build |
| Memory | **GBrain** (gbrain.io) | Every mission as a page with sourced events; lessons when a prediction was wrong; precedents for explanations | **loading**: 391 pages (all 389 missions incl. every unexplained event) |
| Front end | Web page + CesiumJS | Date / time / rocket input, the recommendation with citations, the correction box, the globe | to build |
| Scheduled jobs | **GBrain** (gbrain.io schedules) | Refresh the week's forecast, check for new Space Force forecasts, nightly retrain | to wire |
| Cause labeler | **River** | Post-train a small open model on evidence-verified causes, label the unknowns (stays "predicted" until confirmed) | stretch |

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
| FORECAST | Space Force forecast exists for that mission/date; its probability is attached | Yes: median 60% before known weather scrubs vs 20% before non-weather ones | 45 |
| ~~PAD~~ | Another launch from the same pad within 48 h | **Rejected**: fires 38% on known vehicle/weather events vs 40% on unknowns (no signal) | - |
| ~~RANGE~~ | Another scrub at the Cape in the previous 24 h | **Rejected**: 14% vs 13% (no signal) | - |

**Result: 200 of 1,235 unexplained events now have an evidence-backed candidate cause
(67 of 132 scrubs, 133 of 1,103 slips). 1,035 have no evidence in any data we hold.**
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
4. Recommendation step (one LLM call over model score + rule results + GBrain precedents) and the correction box, 60 min.
5. CesiumJS view, 30 min (parallel, second person).
6. Demo run: real upcoming launch, a teammate corrects it, rerun cites the lesson.
7. Stretch: River cause labeler.

## Ground rules

- **No proxies, no invented thresholds.** Missing measurement = `UNAVAILABLE`, named.
- **Forecasts are never graded as observations.** Predictions are never stored as labels.
- **Report what closes.** Measured numbers only, including the ones that don't flatter us.
