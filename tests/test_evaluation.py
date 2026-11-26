"""Guards against the evaluation mistakes this project originally had:
duplicate rows and train/test overlap. Run: python -m pytest tests/test_evaluation.py"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data import load_data, split_data


def test_no_duplicate_rows_after_loading():
    X, y, _ = load_data()
    full = X.assign(PlacedOrNot=y)
    assert not full.duplicated().any()


def test_train_and_test_do_not_overlap():
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    train = set(map(tuple, X_train.assign(t=y_train).values.tolist()))
    test = set(map(tuple, X_test.assign(t=y_test).values.tolist()))
    assert train.isdisjoint(test), "Some test rows also appear in the training set (data leakage)"


def test_split_is_deterministic_and_stratified():
    X, y, _ = load_data()
    a = split_data(X, y)
    b = split_data(X, y)
    assert a[1].index.equals(b[1].index)
    assert abs(a[3].mean() - y.mean()) < 0.03
