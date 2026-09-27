"""Build the cause-classifier dataset (for a small model post-trained on River).

Schema: artifacts/classifier/schema.json. Task: given one non-launch event, output
{"cause": <enum>}. Labels come only from confirmed evidence: the event's own stated /
webcast / press cause, plus web-research results with a source URL and quote
(artifacts/research/results_*.json). Candidate causes are NOT labels.

Every labeled event appears twice: "full" (with the operator's text) and "masked"
(text removed; only mission, timing and observed weather). The masked variant is what
the unknowns look like; a model trained only on "full" would learn to read the cause
out of the text and have nothing to go on for the 1,035 events with no text.

Split by time: train = events before 2025-01-01, eval = 2025 onward.
Unlabeled events go to unlabeled.jsonl (masked + full text where any) for inference;
predictions stay graded PREDICTED until a person or an instrument confirms them.

    python3 scripts/build_classifier_dataset.py  ->  artifacts/classifier/
"""
import csv
import glob
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
OUT = ART / "classifier"
CAUSES = ["WX_LAUNCH_SITE", "WX_RECOVERY", "WX_UPPER_WINDS", "VEHICLE", "GROUND_SYSTEM", "PAYLOAD",
          "RANGE", "TRAFFIC_ORBITAL", "SCHEDULE", "CASCADE", "UNKNOWN"]
SYSTEM = ("You classify why a rocket launch at Cape Canaveral or Kennedy did not go when announced. "
          f"Answer with JSON {{\"cause\": <one of {', '.join(CAUSES)}>}}. "
          "Use UNKNOWN when the input does not support a cause. Observed weather is from instruments at the pad; "
          "a forecast probability is a forecast, not an observation.")


def weather_line(e, pov):
    parts = [f"weather verdict {e.get('verdict')}"]
    if e.get("rules_fired"):
        parts.append("rules fired: " + ", ".join(e["rules_fired"]))
    for k, label in (("ceiling_ft", "ceiling_ft"), ("cape", "CAPE"), ("freezing_ft", "freezing_level_ft"),
                     ("glm_nearest_nm", "nearest_lightning_nm"), ("echo_top_10nm_kft", "echo_top_10nm_kft")):
        if e.get(k) is not None:
            parts.append(f"{label} {e[k]}")
    if pov is not None:
        parts.append(f"45WS forecast POV {pov}%")
    return "; ".join(parts)


def render(e, pov, masked):
    hour = (e.get("time") or "00:00")[:2]
    lines = [f"Mission: {e['mission']} | Vehicle: {e['vehicle']} | Pad: {e['pad']} | Provider: {e['provider']}",
             f"Event: {e['kind']} ({e['happened']}) | Month: {e['date'][5:7]} | Hour UTC: {hour}",
             f"Observed: {weather_line(e, pov)}"]
    if not masked:
        if e.get("stated"):
            lines.append(f"Stated: {e['stated'][:500]}")
        if e.get("quote"):
            lines.append(f"Quote: {e['quote'][:500]}")
    return "\n".join(lines)


def example(e, pov, masked, cause=None, meta=None):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(e, pov, masked)}]
    if cause is not None:
        msgs.append({"role": "assistant", "content": json.dumps({"cause": cause})})
    return {"messages": msgs, "metadata": {**(meta or {}), "variant": "masked" if masked else "full"}}


def main():
    events = json.loads((ART / "all_events.json").read_text())
    # 45 WS launch-day POV per event (same matching as enrich_causes)
    pov = {int(r["event_index"]): int(r["forecast_pov_pct"]) for r in csv.DictReader((ART / "cause_candidates.csv").open())
           if r["evidence_grade"] == "FORECAST" and r["forecast_pov_pct"]}
    web = {}
    for f in glob.glob(str(ART / "research" / "results_*.json")):
        for r in json.loads(Path(f).read_text()):
            if r.get("evidence_grade") == "WEB" and r.get("cause") in CAUSES and r["cause"] != "UNKNOWN" and r.get("quote") and r.get("source_url"):
                web[int(r["event_index"])] = r

    OUT.mkdir(exist_ok=True)
    splits = {"train": [], "eval": [], "unlabeled": []}
    label_src = Counter()
    for i, e in enumerate(events):
        p = pov.get(i)
        meta = {"event_index": i, "mission": e["mission"], "date": e["date"], "kind": e["kind"]}
        if e["category"] != "UNKNOWN" and e.get("evidence_kind") not in (None, "", "none"):
            cause, src = e["category"], e["evidence_kind"]
        elif i in web:
            cause, src = web[i]["cause"], f"web: {web[i]['source_url']}"
            e = {**e, "quote": web[i]["quote"]}
        else:
            splits["unlabeled"].append(example(e, p, masked=False, meta=meta))
            continue
        label_src["stated/webcast/press" if not src.startswith("web:") else "web research"] += 1
        split = "train" if e["date"] < "2025-01-01" else "eval"
        for masked in (False, True):
            splits[split].append(example(e, p, masked, cause, {**meta, "label_source": src}))

    for name, rows in splits.items():
        with (OUT / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    schema = {
        "task": "single-label classification of why a Cape launch did not go when announced",
        "output": {"type": "object", "properties": {"cause": {"type": "string", "enum": CAUSES}}, "required": ["cause"]},
        "input_fields": ["mission", "vehicle", "pad", "provider", "event kind", "what happened", "month", "hour UTC",
                         "observed weather (verdict, rules fired, ceiling, CAPE, freezing level, nearest lightning, echo tops)",
                         "45WS forecast POV (when one exists)", "stated text / quote (full variant only)"],
        "labels_from": "confirmed evidence only: operator statement, webcast quote, press, or web research with URL + quote",
        "split": "train < 2025-01-01 <= eval; each labeled event twice (full + masked)",
        "format": "OpenAI-style chat JSONL (system, user, assistant) with metadata",
        "system_prompt": SYSTEM,
    }
    (OUT / "schema.json").write_text(json.dumps(schema, indent=2))

    def dist(rows):
        return dict(Counter(json.loads(r["messages"][-1]["content"])["cause"] for r in rows if r["metadata"]["variant"] == "full"))
    print(f"labeled events: {sum(label_src.values())} ({dict(label_src)})")
    print(f"train: {len(splits['train'])} rows  {dist(splits['train'])}")
    print(f"eval:  {len(splits['eval'])} rows  {dist(splits['eval'])}")
    print(f"unlabeled: {len(splits['unlabeled'])} rows -> {OUT}")


if __name__ == "__main__":
    main()
