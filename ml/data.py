"""Shared data loading, splitting and pipeline construction.

Every script (training, evaluation, tests) imports from here so they all use
exactly the same cleaning and the same train/test split. That is what keeps the
reported metrics honest.
"""
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "placement_predictor.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TARGET = "PlacedOrNot"
CAT_COLS = ["Gender", "Stream", "HistoryOfBacklogs"]
PASSTHROUGH_COLS = ["Age", "Internships"]
SCALED_COLS = ["CGPA"]

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(drop_duplicates=True):
    """Load the processed dataset.

    The raw data contains many exact duplicate rows. If duplicates are kept, a
    random split puts copies of the same student in both the training and the
    test set, and the model is then "tested" on rows it has already seen. So
    duplicates are dropped by default.

    Returns (X, y, info) where info describes the cleaning that was done.
    """
    df = pd.read_pickle(DATA_PATH)
    for col in CAT_COLS:
        df[col] = df[col].astype(str)
    df[TARGET] = df[TARGET].astype(int)

    n_rows = len(df)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    info = {
        "rows_in_file": n_rows,
        "rows_used": len(df),
        "duplicates_removed": n_rows - len(df),
    }
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return X, y, info


def split_data(X, y):
    """The single, fixed, stratified train/test split used everywhere."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor(cat_cols=None, passthrough_cols=None):
    cat_cols = CAT_COLS if cat_cols is None else cat_cols
    passthrough_cols = PASSTHROUGH_COLS if passthrough_cols is None else passthrough_cols
    transformers = [("cgpa", StandardScaler(), SCALED_COLS)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if passthrough_cols:
        transformers.append(("num", "passthrough", passthrough_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model():
    return GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    )


def build_pipeline(model=None, cat_cols=None, passthrough_cols=None):
    return Pipeline(
        [
            ("preprocessing", build_preprocessor(cat_cols, passthrough_cols)),
            ("model", model if model is not None else build_model()),
        ]
    )
