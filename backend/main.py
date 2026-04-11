from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import os

from ml.predictor import predict_placement_probability
from ml.skill_match_role import skill_match_score, get_ordered_missing_skills
from ml.aggregator import calculate_final_score

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

ROLE_SKILLS_PATH = os.path.join(
    BASE_DIR, "data", "processed", "role_skills.pkl"
)

PLACEMENT_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "placement_predictor.pkl"
)


class EvaluationInput(BaseModel):
    age: int
    gender: str
    stream: str
    internships: int
    cgpa: float
    history_of_backlogs: str
    role: str
    skills: List[str]



@app.get("/options/roles")
def get_roles():
    df = pd.read_pickle(ROLE_SKILLS_PATH)
    return {"roles": df["Role"].astype(str).tolist()}


@app.get("/options/placement")
def get_placement_options():
    df = pd.read_pickle(PLACEMENT_DATA_PATH)

    return {
        "gender": df["Gender"].astype(str).unique().tolist(),
        "stream": df["Stream"].astype(str).unique().tolist(),
        "history_of_backlogs": df["HistoryOfBacklogs"].astype(str).unique().tolist()
    }


@app.get("/options/skills")
def get_skills_for_role(role: str):
    df = pd.read_pickle(ROLE_SKILLS_PATH)
    row = df[df["Role"] == role]

    if row.empty:
        return {"skills": []}

    role_skill_map = row["role_skill_map"].iloc[0]

    skills = (
        role_skill_map["must"]
        | role_skill_map["good"]
        | role_skill_map["bonus"]
    )

    return {"skills": sorted(list(skills))}



@app.post("/evaluate")
def evaluate(data: EvaluationInput):

    academic_data = {
        "Age": data.age,
        "Gender": data.gender,
        "Stream": data.stream,
        "Internships": data.internships,
        "CGPA": data.cgpa,
        "HistoryOfBacklogs": data.history_of_backlogs
    }

    placement_prob = predict_placement_probability(academic_data)

    skill_score = skill_match_score(data.skills, data.role)
    missing_skills = get_ordered_missing_skills(data.skills, data.role)


    final_score = calculate_final_score(placement_prob, skill_score)

    return {
        "placement_probability": round(placement_prob, 2),
        "skill_match_score": round(skill_score, 2),
        "final_score": round(final_score, 2),
        "missing_skills": missing_skills
    }
