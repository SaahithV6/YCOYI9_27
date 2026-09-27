# Cause classifier dataset (for River)

Task: given one time a Cape launch did not go when announced, output `{"cause": <label>}`.

| file | rows | what |
|---|---|---|
| `train.jsonl` | 366 | 183 labeled events before 2025, each twice (full + masked) |
| `eval.jsonl` | 168 | 84 labeled events from 2025 on, each twice |
| `unlabeled.jsonl` | 1,235 | events with no published cause; run the trained model on these |
| `schema.json` | | label enum, input fields, split, system prompt |
| `build_classifier_dataset.py` | | how the files were built (runs inside the scrubline repo) |

**Format:** OpenAI-style chat JSONL: `messages` = system, user, assistant (assistant = the label as
JSON), plus `metadata` (event index, mission, date, kind, label source, variant). Convert to whatever
River's fine-tuning endpoint expects if it differs; the fields are all there.

**Labels (11):** WX_LAUNCH_SITE, WX_RECOVERY, WX_UPPER_WINDS, VEHICLE, GROUND_SYSTEM, PAYLOAD, RANGE,
TRAFFIC_ORBITAL, SCHEDULE, CASCADE, UNKNOWN.

**Where labels come from:** only confirmed evidence, meaning the operator's own statement, a webcast
quote, or press (267 events). Candidate causes (instrument / forecast / carry) are NOT labels.
Web-research labels (source URL + quote) will be added in a later version after their quotes are verified.

**full vs masked:** `full` includes the operator's text; `masked` removes it and leaves mission,
vehicle, pad, month, hour and observed weather (verdict, rules fired, ceiling, CAPE, freezing level,
nearest lightning, echo tops, Space Force forecast probability when one exists). The unexplained
events look like `masked`, so **report the masked eval score as the headline**. The full score is
inflated because the cause is often written in the text.

**Caveats:**
- Small and imbalanced. Train (full variant): WX_LAUNCH_SITE 72, VEHICLE 45, SCHEDULE 20, RANGE 19,
  PAYLOAD 11, WX_RECOVERY 9, WX_UPPER_WINDS 4, GROUND_SYSTEM 2, TRAFFIC_ORBITAL 1.
- Eval has no RANGE / WX_UPPER_WINDS / GROUND_SYSTEM / TRAFFIC_ORBITAL, so eval accuracy says
  nothing about those classes.
- Predictions on `unlabeled.jsonl` stay graded PREDICTED until a person or an instrument confirms them.

Sources: Launch Library 2 update stream (via scrubline), KXMR METAR, GOES GLM, NEXRAD, IGRA soundings,
archived 45th Weather Squadron Launch Mission Execution Forecasts.
