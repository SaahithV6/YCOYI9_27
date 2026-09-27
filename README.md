# Scrubline

**Why launches don't go, and which ones won't.**

Launch pads on the Space Coast have shown they can fly 143 times a year. They fly 73.
Every time a launch doesn't go, the industry learns almost nothing, because most of the
time nobody says why. Of **1,502** missed launch dates at Cape Canaveral and Kennedy since
2020 (1,216 slips, 286 scrubs, across 389 missions), **1,235 (82%) have no public cause**.
There isn't even an agreed definition of what counts as a scrub, a slip, or a launch that
went fine.

You can't schedule around, insure, or engineer against a failure you can't name. Scrubline
turns those silent misses into a record with evidence behind every cause, and a model that
predicts which launches will slip, and why, before the countdown starts.

Built at the YC *Own Your Intelligence* hackathon, 2026-09-27.

## 1. A definition of launch outcomes

Every event gets three fields. A prediction is never stored as a label; every label
carries the quote, video timestamp or instrument reading behind it.

| field | values |
|---|---|
| **outcome** | `FLEW_ON_TIME` · `SLIP` (moved before the count) · `SCRUB` (count stopped) |
| **cause** | `WX_LAUNCH_SITE` · `WX_RECOVERY` · `WX_UPPER_WINDS` · `VEHICLE` · `GROUND_SYSTEM` · `PAYLOAD` · `RANGE` · `TRAFFIC_ORBITAL` · `SCHEDULE` · `CASCADE` (knock-on from another launch) · `UNKNOWN` |
| **evidence** | `STATED` (operator) · `WEBCAST` (quote + offset) · `INSTRUMENT` (METAR, GLM, radar, sounding, sea state) · `PREDICTED` · `NONE` |

## 2. How the record gets filled in

```
             1,235 unexplained events
                        │
      ┌──────────── QM agent fleet ────────────┐
      │ webcast · operator · weather · recovery │   one agent per evidence source,
      │ range/orbital traffic                   │   in parallel
      └────────────────────┬───────────────────┘
                           │ proposed label + evidence
                           ▼
              Slack review channel (QM)          a teammate approves or rejects
                           │
                           ▼
          GBrain association graph               launches · boosters · pads · payloads · causes
          knock-on slips, repeat failures,       (A scrubs → B slips the next range day)
          graph features for the model
                           │
                           ▼
          River: small open cause-labeler         fine-tuned on evidence-verified labels;
                                                  output stays PREDICTED until confirmed
                           │
                           ▼
          Prediction model (gradient-boosted trees, not an LLM)
          P(slip / scrub) + likely cause for an upcoming launch
```

## 3. The prediction model

- **Features:** pad, vehicle, provider, booster flight count; hour of day and month;
  cloud ceiling, CAPE, freezing level, nearest lightning, radar echo tops; hours since the
  pad last flew; same-range traffic within 48 h; recovery-zone sea state.
- **Evaluation:** train 2021–2024, test 2025–2026. Launches that flew on time are the
  negative class. The update stream behind scrub reasons does not exist before 2021.
- **Output:** probability of a slip or scrub and the top drivers, for a real upcoming launch.

## 4. Ground rules

- **No proxies, no invented thresholds.** A rule with no real measurement returns
  `UNAVAILABLE` and names the missing instrument.
- **Report what closes.** Webcasts closed 19% of unknowns in earlier work, and routine
  Starlink slips barely close at all. We measure the close rate; we don't promise one.

## Sponsors used

- **QM:** the enrichment agent fleet and the human review loop in Slack.
- **GBrain:** the association graph over launches, boosters, pads and causes.
- **River:** the fine-tuned cause labeler.

## Status

Day-of plan. Data and code land in the following commits.
