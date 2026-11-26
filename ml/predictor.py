import joblib
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "placement_model.joblib"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.joblib"))

def predict_placement_probability(user_features: dict) -> float:

    X_df = pd.DataFrame([user_features])

    cat_cols = ["Gender", "Stream", "HistoryOfBacklogs"]
    for col in cat_cols:
        X_df[col] = X_df[col].astype(str)

    X = preprocessor.transform(X_df)
    probability = model.predict_proba(X)[0][1]

    # Never present a statistical estimate as certainty: keep it inside 2%-98%.
    return round(float(np.clip(probability, 0.02, 0.98)), 2)
