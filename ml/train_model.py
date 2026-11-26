"""Train the placement model and save it.

The model is fitted ONLY on the training split. The held-out test split is never
seen during training, so the metrics reported by ml/evaluate.py describe exactly
the model that gets saved here.

Run from the project root:   python -m ml.train_model
"""
import os

import joblib

from ml.data import MODEL_DIR, build_pipeline, load_data, split_data


def train_and_save_model():
    X, y, info = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    # Saved as two files so ml/predictor.py can keep loading them as before.
    joblib.dump(pipeline.named_steps["model"], os.path.join(MODEL_DIR, "placement_model.joblib"))
    joblib.dump(pipeline.named_steps["preprocessing"], os.path.join(MODEL_DIR, "preprocessor.joblib"))

    print(f"Rows in file: {info['rows_in_file']} | duplicates removed: {info['duplicates_removed']} "
          f"| unique rows used: {info['rows_used']}")
    print(f"Trained on {len(X_train)} rows, held out {len(X_test)} rows for testing.")
    print("Model saved to", MODEL_DIR)
    print("Now run:  python -m ml.evaluate")


if __name__ == "__main__":
    train_and_save_model()
