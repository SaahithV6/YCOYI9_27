"""Train the weather-risk model: P(weather stops this attempt | forecast at T-0).

LightGBM on data/weather/train_table.csv. Time split: train < 2025-01-01 <= test. Reported next to
two baselines: the base rate, and logistic regression on the same features. Per-prediction factor
contributions come from LightGBM's own pred_contrib (no extra dependency).

    .venv/bin/python train/train_weather.py  ->  models/weather_model.txt, models/weather_model.json,
                                                 runs/weather-<ts>/metrics.json, data/weather/unexplained_scored.csv
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "weather"
META = ["event_index", "mission", "time_utc", "site", "vehicle", "kind", "cause", "label"]
CATS = ["site", "vehicle"]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
ROUNDS = 300


def frame(df, cats_levels=None):
    X = df.drop(columns=[c for c in META if c in df.columns])
    X = pd.concat([X, pd.get_dummies(df[CATS].astype(str), prefix=CATS, dtype=float)], axis=1)
    if cats_levels is not None:
        X = X.reindex(columns=cats_levels, fill_value=0.0)
    return X.astype(float)


def metrics(y, p):
    return {"n": int(len(y)), "positives": int(y.sum()), "roc_auc": round(roc_auc_score(y, p), 3),
            "pr_auc": round(average_precision_score(y, p), 3), "brier": round(brier_score_loss(y, p), 4)}


def main():
    df = pd.read_csv(DATA / "train_table.csv")
    train, test = df[df.time_utc < "2025-01-01"], df[df.time_utc >= "2025-01-01"]
    Xtr = frame(train)
    cols = list(Xtr.columns)
    Xte = frame(test, cols)
    ytr, yte = train.label.values, test.label.values
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())

    booster = lgb.train({**PARAMS, "scale_pos_weight": spw}, lgb.Dataset(Xtr, ytr), num_boost_round=ROUNDS)
    p_gbm = booster.predict(Xte)
    logit = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    logit.fit(Xtr, ytr)
    p_lr = logit.predict_proba(Xte)[:, 1]
    base = np.full(len(yte), ytr.mean())

    gain = booster.feature_importance("gain")
    top = sorted(zip(cols, gain), key=lambda x: -x[1])[:12]
    report = {
        "split": {"train": "< 2025-01-01", "test": ">= 2025-01-01"},
        "train": {"n": int(len(ytr)), "positives": int(ytr.sum())},
        "test": {"lightgbm": metrics(yte, p_gbm), "logistic_regression": metrics(yte, p_lr),
                 "base_rate": {**metrics(yte, base), "note": "constant = train positive rate; AUC 0.5 by construction"}},
        "top_features_by_gain": [[c, round(float(g), 1)] for c, g in top],
        "note": "scale_pos_weight balances classes, so raw scores are ranked risk, not calibrated probabilities.",
    }

    # Out-of-fold scores (5-fold) for every labeled row: the reference distributions the app compares a
    # new hour against. In-sample scores would make flown launches look safer and weather scrubs riskier.
    from sklearn.model_selection import StratifiedKFold
    Xall_cv, yall = frame(df, cols), df.label.values
    oof = np.zeros(len(df))
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(Xall_cv, yall):
        spw_f = (yall[tr_i] == 0).sum() / max(1, (yall[tr_i] == 1).sum())
        b = lgb.train({**PARAMS, "scale_pos_weight": spw_f}, lgb.Dataset(Xall_cv.iloc[tr_i], yall[tr_i]), num_boost_round=ROUNDS)
        oof[te_i] = b.predict(Xall_cv.iloc[te_i])
    report["oof_all"] = metrics(yall, oof)

    # Refit on all labeled data for serving, then score the unexplained events.
    Xall = frame(df, cols)
    final = lgb.train({**PARAMS, "scale_pos_weight": (df.label == 0).sum() / max(1, df.label.sum())},
                      lgb.Dataset(Xall, df.label.values), num_boost_round=ROUNDS)
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    final.save_model(str(models / "weather_model.txt"))
    (models / "weather_model.json").write_text(json.dumps({"features": cols, "categoricals": CATS, "params": PARAMS,
                                                           "rounds": ROUNDS, "trained_rows": int(len(df)),
                                                           "test_metrics": report["test"]}, indent=2))
    (models / "reference_scores.json").write_text(json.dumps({
        "note": "5-fold out-of-fold scores of the labeled rows; the app ranks a new hour against these",
        "flew": sorted(round(float(x), 5) for x in oof[yall == 0]),
        "weather_scrub": sorted(round(float(x), 5) for x in oof[yall == 1])}))
    unx = pd.read_csv(DATA / "unexplained_table.csv")
    unx["weather_risk_score"] = final.predict(frame(unx, cols)).round(4)
    unx[["event_index", "mission", "time_utc", "site", "kind", "weather_risk_score"]].sort_values(
        "weather_risk_score", ascending=False).to_csv(DATA / "unexplained_scored.csv", index=False)
    report["unexplained_scored"] = {"n": int(len(unx)), "score_ge_0.5": int((unx.weather_risk_score >= 0.5).sum())}

    run = ROOT / "runs" / f"weather-{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
