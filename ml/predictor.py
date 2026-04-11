# ml/predictor.py

import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "placement_model.joblib"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.joblib"))

def predict_placement_probability(user_features: dict) -> float:

    X_df = pd.DataFrame([user_features])

    # Force categorical columns to string
    cat_cols = ["Gender", "Stream", "HistoryOfBacklogs"]
    for col in cat_cols:
        X_df[col] = X_df[col].astype(str)

    X = preprocessor.transform(X_df)
    probability = model.predict_proba(X)[0][1]

    return round(probability, 2)
