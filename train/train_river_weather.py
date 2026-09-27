"""River LoRA SFT: will weather stop this launch attempt? Compared head-to-head with LightGBM.

    set -a; . ~/.config/river/env; set +a
    .venv/bin/python train/train_river_weather.py [--smoke]

Same train/test launches as train_weather.py. Weather-stopped rows are repeated (--pos-repeat) so the
model does not learn to always answer "no" (1 in 6 rows is a weather stop). Reports precision / recall /
F1 on "yes" for the fine-tuned model, and LightGBM's precision / recall at the same number of "yes" calls
on the same launches, so the comparison is like for like.

Writes runs/river-weather-<ts>/: config.json, train_log.jsonl, test_predictions.jsonl, metrics.json.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import river_client as river
from river_client.renderers import TrainOnWhat, get_renderer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "weather"


def load(name):
    return [json.loads(l) for l in (DATA / f"{name}.jsonl").open()]


def prf(gold, pred):
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    tp, fp, fn = int(tp), int(fp), int(fn)
    return {"yes_calls": tp + fp, "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}


def lightgbm_same_budget(n_yes):
    """LightGBM (trained on the same pre-2025 rows) calling 'yes' on its n_yes riskiest test launches."""
    import sys
    sys.path.insert(0, str(ROOT / "train"))
    from train_weather import PARAMS, ROUNDS, frame
    df = pd.read_csv(DATA / "train_table.csv")
    tr, te = df[df.time_utc < "2025-01-01"], df[df.time_utc >= "2025-01-01"]
    Xtr = frame(tr)
    b = lgb.train({**PARAMS, "scale_pos_weight": (tr.label == 0).sum() / tr.label.sum()}, lgb.Dataset(Xtr, tr.label.values),
                  num_boost_round=ROUNDS)
    s = b.predict(frame(te, list(Xtr.columns)))
    top = set(pd.Series(s).nlargest(n_yes).index) if n_yes else set()
    return prf(list(te.label.values == 1), [i in top for i in range(len(te))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--pos-repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    train, test = load("river_train"), load("river_test")
    run = ROOT / "runs" / (f"river-weather-{time.strftime('%Y%m%d-%H%M%S')}" + ("-smoke" if a.smoke else ""))
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({**vars(a), "n_train": len(train), "n_test": len(test)}, indent=2))

    renderer = get_renderer(a.base, thinking=False)
    rows = train + [r for r in train if r["metadata"]["label"] == 1] * (a.pos_repeat - 1)
    examples = [renderer.build_training_example(r["messages"], train_on=TrainOnWhat.LAST_ASSISTANT).to_dict() for r in rows]
    client = river.Client(api_key=os.environ["RIVER_API_KEY"], endpoint="api.river.ai")
    rng = random.Random(a.seed)
    log = (run / "train_log.jsonl").open("w")

    with client.session(project="scrubline", run=run.name) as session:
        model = session.create_model(base_model=a.base, lora=river.LoraConfig(rank=a.rank, seed=a.seed))
        for ep in range(1 if a.smoke else a.epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
            for bi, idx in enumerate(batches[:1] if a.smoke else batches):
                fb, opt = model.train_step([examples[i] for i in idx], lr=a.lr, loss_fn="cross_entropy", grad_clip_norm=1.0)
                rec = {"epoch": ep, "batch": bi, "step": model.step, "loss_mean": fb.metrics.get("loss_mean"),
                       "grad_norm": opt.metrics.get("grad_norm")}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
        ckpt = None if a.smoke else model.save_weights(f"scrubline-weather-{run.name}")
        evalrows = test[:6] if a.smoke else test
        preds = []
        for i in range(0, len(evalrows), 16):
            chunk = evalrows[i:i + 16]
            prompts = [renderer.build_sample_prompt(r["messages"][:-1]).to_kwargs()["prompt"] for r in chunk]
            groups = model.sample(prompts, max_tokens=24, temperature=0.0, stop=renderer.get_stop_strings())
            for r, g in zip(chunk, groups):
                text = g[0].text
                preds.append({**r["metadata"], "pred": "yes" if '"yes"' in text else "no" if '"no"' in text else "unparseable",
                              "raw": text[:120]})

    with (run / "test_predictions.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    gold = [p["label"] == 1 for p in preds]
    pred = [p["pred"] == "yes" for p in preds]
    river_m = prf(gold, pred)
    metrics = {"n_test": len(preds), "positives": sum(gold), "unparseable": sum(p["pred"] == "unparseable" for p in preds),
               "river_llm": river_m, "checkpoint": getattr(ckpt, "path", None) if ckpt else None}
    if not a.smoke:
        metrics["lightgbm_same_number_of_yes_calls"] = lightgbm_same_budget(river_m["yes_calls"])
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
