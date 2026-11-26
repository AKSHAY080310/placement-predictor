"""Honest evaluation of the placement model.

Run from the project root:   python -m ml.evaluate

What it reports
  1. Held-out test metrics of the SAVED model (trained only on the training split).
  2. 5-fold stratified cross-validation (more stable on a small dataset).
  3. Baselines: always predicting the majority class, and logistic regression.
  4. An ablation without Gender and Age (sensitive / weak features).

Results are also written to models/metrics.json so the README numbers can be
copied from one place.
"""
import json
import os

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from ml.data import (CAT_COLS, MODEL_DIR, PASSTHROUGH_COLS, RANDOM_STATE,
                     build_pipeline, load_data, split_data)

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}


def holdout_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def cv_summary(pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(clone(pipeline), X, y, cv=cv, scoring=SCORING)
    return {m: (float(res[f"test_{m}"].mean()), float(res[f"test_{m}"].std())) for m in SCORING}


def fmt(d):
    return " | ".join(f"{k} {v:.3f}" for k, v in d.items())


def main():
    X, y, info = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    report = {"dataset": info, "n_train": len(X_train), "n_test": len(X_test)}

    print(f"Dataset: {info['rows_in_file']} rows in file, {info['duplicates_removed']} exact duplicates "
          f"removed -> {info['rows_used']} unique rows")
    print(f"Split: {len(X_train)} train / {len(X_test)} test (stratified, random_state={RANDOM_STATE})\n")

    # 1. Held-out evaluation of the saved model
    model = joblib.load(os.path.join(MODEL_DIR, "placement_model.joblib"))
    pre = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    Xt = pre.transform(X_test)
    pred, prob = model.predict(Xt), model.predict_proba(Xt)[:, 1]
    hold = holdout_metrics(y_test, pred, prob)
    report["holdout"] = hold
    print("1) HELD-OUT TEST SET (saved model)")
    print("  ", fmt(hold))
    print("   confusion matrix [[TN FP] [FN TP]]:", confusion_matrix(y_test, pred).tolist(), "\n")

    # 2. Cross-validation
    cv = cv_summary(build_pipeline(), X, y)
    report["cross_validation_5fold"] = cv
    print("2) 5-FOLD CROSS-VALIDATION (mean +/- std)")
    print("  ", " | ".join(f"{k} {m:.3f}+/-{s:.3f}" for k, (m, s) in cv.items()), "\n")

    # 3. Baselines
    base = {}
    for name, est in {"majority_class": DummyClassifier(strategy="most_frequent"),
                      "logistic_regression": LogisticRegression(max_iter=1000)}.items():
        base[name] = cv_summary(build_pipeline(model=est), X, y)
    report["baselines_cv"] = base
    print("3) BASELINES (5-fold CV)")
    for name, s in base.items():
        print(f"   {name:20s} accuracy {s['accuracy'][0]:.3f} | f1 {s['f1'][0]:.3f} | roc_auc {s['roc_auc'][0]:.3f}")
    print()

    # 4. Ablation: without Gender and Age
    abl = cv_summary(build_pipeline(cat_cols=[c for c in CAT_COLS if c != "Gender"],
                                    passthrough_cols=[c for c in PASSTHROUGH_COLS if c != "Age"]), X, y)
    report["ablation_without_gender_age_cv"] = abl
    print("4) WITHOUT Gender and Age (5-fold CV)")
    print("  ", " | ".join(f"{k} {m:.3f}" for k, (m, s) in abl.items()), "\n")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("Saved models/metrics.json")


if __name__ == "__main__":
    main()
