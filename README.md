# 🎓 Placement Predictor & Skill Gap Analyzer

A full-stack machine learning application that predicts placement probability and provides personalized skill recommendations based on a candidate’s academic profile and target role.

---

## 🚀 Overview

This project combines **machine learning, backend APIs, and an interactive frontend** to help students understand their placement readiness and identify missing skills required for specific roles.

The system takes user inputs such as CGPA, internships, stream, and skills, and outputs:

* 📊 Placement Probability
* 🎯 Skill Match Score
* 📈 Final Placement Readiness Score
* 🧠 Personalized Missing Skills

---

## 🧠 Key Features

* **Placement Prediction Model**

  * Built using Gradient Boosting Classifier
  * Predicts probability of placement based on academic features

* **Skill Gap Analyzer**

  * Matches user skills with role-specific requirements
  * Categorizes skills into *must-have*, *good-to-have*, and *bonus*
  * Recommends prioritized missing skills

* **Hybrid Scoring System**

  * Combines ML prediction + skill match score
  * Produces a final placement readiness score

* **Dynamic Frontend**

  * Interactive form with role-based skill selection
  * Real-time prediction results

* **REST API Backend**

  * Built with FastAPI
  * Endpoints for prediction, role options, and skill mapping

---

## 🏗️ Tech Stack

* **Machine Learning:** scikit-learn, pandas, numpy
* **Backend:** FastAPI, Uvicorn
* **Frontend:** React / Streamlit
* **Model Persistence:** joblib

---

## 📊 Model Performance

* **Accuracy:** ~89%
* **Precision:** ~96% (high reliability for predicted placements)
* **Recall:** ~82%
* **F1 Score:** ~0.89

> The model demonstrates strong predictive capability with balanced precision and recall.

---

## 📁 Project Structure

```id="tree1"
placement-predictor/
│
├── backend/          # FastAPI backend
├── frontend/         # React frontend
├── ml/               # ML models and logic
├── data/             # Processed datasets
├── models/           # Saved model files
└── README.md
```

---

## ⚙️ How to Run

### 1️⃣ Clone the repository

```id="clone"
git clone https://github.com/your-username/placement-predictor.git
cd placement-predictor
```

### 2️⃣ Run Backend

```id="backend"
python -m uvicorn backend.main:app --reload
```

### 3️⃣ Run Frontend

```id="frontend"
cd frontend
npm install
npm run dev
```

### 4️⃣ Open in Browser

```id="url"
http://localhost:5173
```

---

## 📌 Example Workflow

1. Enter academic details (CGPA, internships, stream, etc.)
2. Select target role
3. Choose known skills
4. Click **Evaluate**
5. View:

   * Placement probability
   * Final score
   * Missing skills

---

## 🔍 Future Improvements

* Model explainability (feature importance, SHAP)
* Advanced ML models (XGBoost, LightGBM)
* User authentication & history tracking
* Deployment (cloud hosting)
* Data expansion for improved accuracy

---

## 💡 Key Learnings

* End-to-end ML system design
* Feature engineering and preprocessing pipelines
* Model evaluation (accuracy, precision, recall, F1-score)
* Full-stack integration (ML + API + frontend)

---

## 👨‍💻 Author

**Akshay Raj**
B.Tech – Computer Science & Data Science

---

## ⭐ If you found this useful, consider giving a star!
