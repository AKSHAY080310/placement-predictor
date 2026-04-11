import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data","processed", "placement_predictor.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def train_and_save_model():
    print(DATA_PATH)
    df = pd.read_pickle(DATA_PATH)
    
    X = df.drop("PlacedOrNot", axis=1)
    y = df["PlacedOrNot"]

    num_scaled_cols = ["CGPA"]
    cat_cols = ["Gender", "Stream", "HistoryOfBacklogs"]
    passthrough_cols = ["Age", "Internships"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cgpa", StandardScaler(), num_scaled_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        pipeline.named_steps["model"],
        os.path.join(MODEL_DIR, "placement_model.joblib")
    )

    joblib.dump(
        pipeline.named_steps["preprocessing"],
        os.path.join(MODEL_DIR, "preprocessor.joblib")
    )

    print("Gradient Boosting model trained with CGPA scaling only")


if __name__ == "__main__":
    train_and_save_model()

