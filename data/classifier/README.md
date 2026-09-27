# Cause classifier dataset (for River)

Task: given one time a Cape launch did not go when announced, output `{"cause": <label>}`.

| file | rows | what |
|---|---|---|
| `train.jsonl` | 396 | 198 labeled events before 2025, each twice (full + masked) |
| `eval.jsonl` | 178 | 89 labeled events from 2025 on, each twice |
| `unlabeled.jsonl` | 1,215 | events with no published cause; run the trained model on these |
| `research/` | | web research on 59 unexplained scrubs: `results_*.json` (every attempt, with source URL + quote where found) and `verified.json` (each quote checked against its source page) |
| `schema.json` | | label enum, input fields, split, system prompt |
| `build_classifier_dataset.py` | | how the files were built (runs inside the scrubline repo) |

**Format:** OpenAI-style chat JSONL: `messages` = system, user, assistant (assistant = the label as
JSON), plus `metadata` (event index, mission, date, kind, label source, variant). Convert to whatever
River's fine-tuning endpoint expects if it differs; the fields are all there.

**Labels (12):** WX_LAUNCH_SITE, WX_RECOVERY, WX_UPPER_WINDS, VEHICLE, GROUND_SYSTEM, PAYLOAD, RANGE,
TRAFFIC_ORBITAL, SCHEDULE, CASCADE, STATION_READINESS, UNKNOWN. STATION_READINESS (added v2.1) = the
destination station (ISS) was not ready, e.g. Axiom Mission 4 held for Zvezda repair evaluation.

**Where labels come from:** only confirmed evidence, meaning the operator's own statement, a webcast
quote, or press (267 events), plus **19 web-research labels** (v2). Candidate causes (instrument /
forecast / carry) are NOT labels.

Web research: agents searched 59 unexplained scrub attempts under a strict rule (a source must state
the cause for that mission on that date, quoted exactly). 21 came back with a cause, 38 stayed UNKNOWN.
Every quote was then fetched from its source: 17 exact, 3 near-exact (punctuation), 1 confirmed by hand.
Rejected 1: Starlink 10-4 (quote does not tie the grounding to that date). Axiom Mission 4 was held
out in v2 because no label fit; v2.1 adds STATION_READINESS and includes it (20 research labels, 287 total).

**full vs masked:** `full` includes the operator's text; `masked` removes it and leaves mission,
vehicle, pad, month, hour and observed weather (verdict, rules fired, ceiling, CAPE, freezing level,
nearest lightning, echo tops, Space Force forecast probability when one exists). The unexplained
events look like `masked`, so **report the masked eval score as the headline**. The full score is
inflated because the cause is often written in the text.

**Caveats:**
- Small and imbalanced. Train (full variant): WX_LAUNCH_SITE 73, VEHICLE 50, SCHEDULE 21, RANGE 20,
  PAYLOAD 13, WX_RECOVERY 10, GROUND_SYSTEM 5, WX_UPPER_WINDS 4, TRAFFIC_ORBITAL 1, CASCADE 1.
- STATION_READINESS has 1 example and it is in eval (2025); train has none, so the model cannot learn it yet.
- Eval has 1 RANGE and no WX_UPPER_WINDS / GROUND_SYSTEM / TRAFFIC_ORBITAL / CASCADE, so eval
  accuracy says little about those classes.
- Predictions on `unlabeled.jsonl` stay graded PREDICTED until a person or an instrument confirms them.

Sources: Launch Library 2 update stream (via scrubline), KXMR METAR, GOES GLM, NEXRAD, IGRA soundings,
archived 45th Weather Squadron Launch Mission Execution Forecasts.
