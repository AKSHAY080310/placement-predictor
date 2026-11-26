# Placement Predictor & Skill Gap Analyzer

A full-stack machine learning app that estimates a student's placement probability from their academic profile and shows which skills they are missing for a target role.

**Stack:** scikit-learn · FastAPI · React (Vite) · joblib

---

## What it does

Enter your academic details (CGPA, internships, stream, backlogs, ...), pick a target role and tick the skills you already have. The app returns:

- **Placement probability** from a Gradient Boosting model
- **Skill match score** against the role's must-have / good-to-have / bonus skills
- **Readiness score** (30% model probability + 70% skill match)
- **Missing skills**, ordered by importance

---

## Model results

Evaluated with `python -m ml.evaluate` on **1,003 unique students** (see "Dataset and evaluation" below).

| Metric | Held-out test (201 rows) | 5-fold cross-validation |
|---|---|---|
| Accuracy | 79.1% | 79.4% ± 1.6% |
| Precision | 89.0% | 87.4% ± 2.1% |
| Recall | 74.2% | 76.4% ± 2.7% |
| F1 | 0.81 | 0.82 ± 0.02 |
| ROC-AUC | 0.87 | 0.88 ± 0.01 |

Comparison with simpler models (5-fold CV):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Always predict "placed" (majority class) | 59.5% | 0.75 | 0.50 |
| Logistic regression | 72.4% | 0.77 | 0.83 |
| **Gradient Boosting (used)** | **79.4%** | **0.82** | **0.88** |

Without Gender and Age as inputs, cross-validated accuracy drops to 75.7%, so those columns carry some signal in this dataset (see Limitations).

---

## Dataset and evaluation

- The source file has 2,966 rows, but **1,963 of them are exact duplicates**. After the `Hostel` column is dropped, only **1,003 unique rows** remain.
- Duplicates are removed *before* splitting. If they are kept, copies of the same student land in both the training and test sets and the metrics look better than they are.
- The model is fitted **only on the training split** (802 rows) and tested on a stratified 20% held-out split (201 rows). The held-out rows are never used for training.
- `tests/test_evaluation.py` checks that no duplicates remain and that train and test do not overlap.
- An earlier version of this project trained on all rows and then evaluated on a subset of the same rows, which reported about 89% accuracy. That number was inflated and has been replaced by the results above.

## Limitations

- **The data is very simple.** CGPA takes only the values 5, 6, 7, 8 and 9. In this dataset every student with CGPA 8 or above is placed and only 7% of students with CGPA 5 are, so the model mostly learns a CGPA threshold. It does not capture real-world hiring. Treat the output as an illustration of an ML workflow, not as a prediction of a real outcome.
- **Label noise.** 124 of 879 distinct feature combinations appear with both outcomes, which limits the achievable accuracy.
- **Gender and Age are model inputs.** They improve accuracy in this dataset, but using them to judge a person is questionable. A production system should not do this.
- **Small dataset.** 1,003 unique rows is enough for a demo, not for a reliable estimator.
- Displayed probabilities are clipped to 2%-98% so the app never claims certainty.

---

## Project structure

```
placement-predictor/
├── backend/        FastAPI app (main.py, schemas.py)
├── frontend/       React + Vite UI
├── ml/             data.py (loading/splitting), train_model.py, evaluate.py,
│                   predictor.py, skill_match_role.py, aggregator.py
├── data/           raw and processed datasets
├── models/         saved model files and metrics.json
├── notebooks/      exploration, feature engineering, training
├── tests/          pytest tests
└── deployment/     Dockerfile, docker-compose, nginx, heroku config
```

---

## How to run

```bash
git clone https://github.com/AKSHAY080310/placement-predictor.git
cd placement-predictor
```

**Backend**

```bash
pip install -r requirements.txt
python -m ml.train_model      # train and save the model (recommended on your own machine)
python -m ml.evaluate         # print the metrics shown above
python -m uvicorn backend.main:app --reload
```

The saved `.joblib` files depend on the scikit-learn version, so retrain with `python -m ml.train_model` after installing the requirements.

**Frontend**

```bash
cd frontend
cp .env.example .env          # set VITE_API_URL if your backend is not on localhost:8000
npm install
npm run dev
```

Open http://localhost:5173

**Tests**

```bash
python -m pytest tests/test_ml.py tests/test_evaluation.py
```

**Deployment settings:** the backend reads `CORS_ORIGINS` (comma-separated allowed frontend origins) and the frontend reads `VITE_API_URL`. See `.env.example` files.

---

## Checking a dataset before using it

`ml/audit_data.py` reports duplicates, conflicting labels, features with very few distinct values or values piled up at a minimum/maximum (a sign of generated data), how much a single feature explains, and how far a full model beats a majority-class guess:

```bash
python -m ml.audit_data data/raw/placement.csv --target PlacedOrNot
```

Running it on this project's own data is what exposed the duplicate rows and the CGPA-threshold pattern described above.

---

## Possible next steps

- Audit and compare other public placement datasets, or use anonymised, permission-based data from a real placement cell
- Add model explainability (feature importance, SHAP)
- Probability calibration and uncertainty display
- Remove Gender/Age from the inputs and measure the effect
- User accounts and history

---

## Author

**Akshay Raj** · B.Tech, Computer & Data Science
