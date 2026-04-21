import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
import os
import joblib
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "placement_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data","processed", "placement_predictor.pkl")
model=joblib.load(MODEL_PATH)
preprocessor=joblib.load(PREPROCESSOR_PATH)

df=pd.read_pickle(DATA_PATH)
df = df.rename(columns={
    "gender": "Gender",
    "stream": "Stream",
    "history_of_backlogs": "HistoryOfBacklogs"
})
x= df.drop("PlacedOrNot", axis=1)
y= df["PlacedOrNot"].astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_test_transformed = preprocessor.transform(x_test)
y_pred = model.predict(x_test_transformed).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
