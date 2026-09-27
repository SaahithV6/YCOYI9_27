"""LoRA SFT of the launch-cause classifier on River, then evaluation on the held-out 2025+ events.

    set -a; . ~/.config/river/env; set +a
    .venv/bin/python train/train_river.py --smoke          # 1 step + 4 eval samples, checks the pipeline
    .venv/bin/python train/train_river.py                  # full run

Writes runs/<timestamp>/: config.json, train_log.jsonl, eval_predictions.jsonl, metrics.json.
Eval reports accuracy and macro-F1 separately for the "masked" rows (no operator text: what the
unexplained events look like, the headline number) and the "full" rows, next to a majority-class
baseline. Predictions on unexplained events stay graded PREDICTED.
"""
import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import river_client as river
from river_client.renderers import TrainOnWhat, get_renderer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "classifier"
CAUSES = json.loads((DATA / "schema.json").read_text())["output"]["properties"]["cause"]["enum"]


def load(name):
    return [json.loads(l) for l in (DATA / f"{name}.jsonl").open()]


def gold(row):
    return json.loads(row["messages"][-1]["content"])["cause"]


def parse(text):
    m = re.search(r'"cause"\s*:\s*"([A-Z_]+)"', text)
    if m and m.group(1) in CAUSES:
        return m.group(1)
    return next((c for c in CAUSES if c in text), "UNPARSEABLE")


def macro_f1(pairs):
    labels = sorted({g for g, _ in pairs})
    f1s = []
    for c in labels:
        tp = sum(g == c and p == c for g, p in pairs)
        fp = sum(g != c and p == c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate(model, renderer, rows, batch=16):
    preds = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = [renderer.build_sample_prompt(r["messages"][:-1]).to_kwargs()["prompt"] for r in chunk]
        groups = model.sample(prompts, max_tokens=32, temperature=0.0, stop=renderer.get_stop_strings())
        for r, g in zip(chunk, groups):
            text = g[0].text
            preds.append({"metadata": r["metadata"], "gold": gold(r), "pred": parse(text), "raw": text[:200]})
    return preds


def score(preds, majority):
    out = {}
    for variant in ("masked", "full"):
        pairs = [(p["gold"], p["pred"]) for p in preds if p["metadata"]["variant"] == variant]
        if not pairs:
            continue
        out[variant] = {
            "n": len(pairs),
            "accuracy": round(sum(g == p for g, p in pairs) / len(pairs), 3),
            "macro_f1": round(macro_f1(pairs), 3),
            "majority_baseline_accuracy": round(sum(g == majority for g, _ in pairs) / len(pairs), 3),
            "unparseable": sum(p == "UNPARSEABLE" for _, p in pairs),
            "confusions": Counter(f"{g}->{p}" for g, p in pairs if g != p).most_common(8),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    train, evalset = load("train"), load("eval")
    majority = Counter(gold(r) for r in train).most_common(1)[0][0]
    run = ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if a.smoke else ""))
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({**vars(a), "n_train": len(train), "n_eval": len(evalset),
                                                 "majority_class": majority, "labels": CAUSES}, indent=2))

    renderer = get_renderer(a.base, thinking=False)
    examples = [renderer.build_training_example(r["messages"], train_on=TrainOnWhat.LAST_ASSISTANT).to_dict() for r in train]
    client = river.Client(api_key=os.environ["RIVER_API_KEY"], endpoint="api.river.ai")
    rng = random.Random(a.seed)
    log = (run / "train_log.jsonl").open("w")

    with client.session(project="scrubline", run=run.name) as session:
        model = session.create_model(base_model=a.base, lora=river.LoraConfig(rank=a.rank, seed=a.seed))
        epochs = 1 if a.smoke else a.epochs
        for ep in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
            for bi, idx in enumerate(batches[:1] if a.smoke else batches):
                fb, opt = model.train_step([examples[i] for i in idx], lr=a.lr, loss_fn="cross_entropy", grad_clip_norm=1.0)
                rec = {"epoch": ep, "batch": bi, "step": model.step, "loss_mean": fb.metrics.get("loss_mean"),
                       "grad_norm": opt.metrics.get("grad_norm")}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
        ckpt = None if a.smoke else model.save_weights(f"scrubline-cause-{run.name}")
        preds = evaluate(model, renderer, evalset[:4] if a.smoke else evalset)

    with (run / "eval_predictions.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    metrics = {"eval": score(preds, majority), "checkpoint": getattr(ckpt, "path", None) if ckpt else None}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
