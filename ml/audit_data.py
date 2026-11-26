"""Audit a tabular dataset BEFORE modelling it.

Usage (from the project root):
    python -m ml.audit_data data/raw/placement.csv --target PlacedOrNot
    python -m ml.audit_data placementdata.csv --target PlacementStatus --id-cols StudentID

It answers: are there duplicates, do identical rows have different labels,
does one feature explain almost everything, do values pile up at a min/max
(a sign of clipped, generated data), and how much can a model really learn?
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def encode_target(s):
    classes = sorted(s.dropna().unique().tolist(), key=str)
    if len(classes) != 2:
        raise SystemExit(f"Target must be binary, found {len(classes)} classes: {classes[:6]}")
    # positive class = the one that looks like "placed"/1/yes/true, otherwise the last one
    pos = next((c for c in classes if str(c).strip().lower() in {"1", "placed", "yes", "true"}), classes[-1])
    return (s == pos).astype(int), pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--target", required=True)
    ap.add_argument("--id-cols", nargs="*", default=[])
    a = ap.parse_args()

    df = pd.read_csv(a.csv).drop(columns=a.id_cols, errors="ignore")
    _, pos = encode_target(df[a.target])
    feats = [c for c in df.columns if c != a.target]
    flags = []

    print(f"File: {a.csv}\nRows: {len(df)} | features: {len(feats)} | positive class: {pos!r}")
    miss = df.isna().mean()
    print("Missing values:", "none" if miss.sum() == 0 else miss[miss > 0].round(3).to_dict())

    # 1. exact duplicates
    dup = int(df.duplicated().sum())
    print(f"\n1) Exact duplicate rows: {dup} ({dup / len(df):.1%})")
    if dup / len(df) > 0.02:
        flags.append(f"{dup / len(df):.0%} of rows are exact duplicates: they inflate test scores unless removed before splitting")
    u = df.drop_duplicates().reset_index(drop=True)
    y = encode_target(u[a.target])[0]
    print(f"   Unique rows: {len(u)}")

    # 2. label conflicts
    g = u.groupby(feats, dropna=False)[a.target].nunique()
    conf = int((g > 1).sum())
    print(f"\n2) Identical feature rows with DIFFERENT labels: {conf} of {len(g)} feature combinations ({conf / len(g):.1%})")
    if conf / len(g) > 0.05:
        flags.append(f"{conf / len(g):.0%} of feature combinations have conflicting labels, so accuracy has a hard ceiling")

    # 3. class balance
    bal = y.mean()
    print(f"\n3) Positive-class share: {bal:.1%}")
    if bal < 0.15 or bal > 0.85:
        flags.append("strongly imbalanced target: accuracy is misleading, use F1/ROC-AUC")

    # 4. clipping / few distinct values
    print("\n4) Numeric columns: distinct values and pile-up at the min/max")
    for c in feats:
        if pd.api.types.is_numeric_dtype(u[c]):
            n = u[c].nunique()
            lo, hi = (u[c] == u[c].min()).mean(), (u[c] == u[c].max()).mean()
            note = ""
            if n <= 6:
                note = "  <- very few distinct values"
                flags.append(f"'{c}' has only {n} distinct values")
            if n > 10 and max(lo, hi) > 0.10:
                note = "  <- values pile up at a bound (clipped / generated data?)"
                flags.append(f"'{c}': {max(lo, hi):.0%} of rows sit exactly at its min or max, which real data rarely does")
            print(f"   {c:28s} distinct={n:4d} at_min={lo:5.1%} at_max={hi:5.1%}{note}")

    # 5. single-feature power
    X = pd.get_dummies(u[feats], dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    maj = max(bal, 1 - bal)
    print(f"\n5) How much does ONE feature explain? (depth-3 tree, 5-fold CV accuracy; majority baseline {maj:.3f})")
    best = (None, 0)
    for c in feats:
        Xc = pd.get_dummies(u[[c]], dtype=float)
        acc = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), Xc, y, cv=cv).mean()
        if acc > best[1]:
            best = (c, acc)
    print(f"   strongest single feature: {best[0]} -> {best[1]:.3f}")
    if best[1] - maj > 0.15:
        flags.append(f"one feature ('{best[0]}') alone reaches {best[1]:.0%} accuracy: the label is mostly a threshold on it")

    # 6. realistic ceiling
    gb = cross_val_score(GradientBoostingClassifier(random_state=42), X, y, cv=cv).mean()
    print(f"\n6) Gradient Boosting, all features, 5-fold CV accuracy: {gb:.3f} (majority baseline {maj:.3f})")
    if gb - maj < 0.05:
        flags.append("a full model barely beats guessing the majority class: little learnable signal")

    print("\n" + "=" * 70 + "\nFLAGS")
    print("\n".join(f" - {f}" for f in flags) if flags else " none raised")


if __name__ == "__main__":
    main()
