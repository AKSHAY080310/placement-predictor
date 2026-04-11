import os
import pandas as pd

SKILL_WEIGHTS = {
    "must": 3,
    "good": 2,
    "bonus": 1
}

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "role_skills.pkl")


def _normalize_skill(skill: str) -> str:
    """
    Normalizes a skill string for reliable matching
    """
    return skill.strip().lower()


def skill_match_score(user_skills, role):
    df = pd.read_pickle(DATA_PATH)

    row = df[df["Role"] == role]

    if row.empty:
        raise ValueError(f"Role '{role}' not found")

    role_skill_map = row["role_skill_map"].iloc[0]

    user_skills = {_normalize_skill(s) for s in user_skills}

    score = 0
    max_score = 0

    for category, skills in role_skill_map.items():
        weight = SKILL_WEIGHTS[category]
        skills = {_normalize_skill(s) for s in skills}
        max_score += len(skills) * weight
        score += len(user_skills & skills) * weight

    return round(score / max_score, 2) if max_score else 0.0


def get_ordered_missing_skills(user_skills, role):
    df = pd.read_pickle(DATA_PATH)
    row = df[df["Role"] == role]

    if row.empty:
        raise ValueError(f"Role '{role}' not found")

    role_skill_map = row["role_skill_map"].iloc[0]
    user_skills = {_normalize_skill(s) for s in user_skills}

    missing = []

    missing.extend(sorted({_normalize_skill(s) for s in role_skill_map["must"]} - user_skills))
    missing.extend(sorted({_normalize_skill(s) for s in role_skill_map["good"]} - user_skills))
    missing.extend(sorted({_normalize_skill(s) for s in role_skill_map["bonus"]} - user_skills))

    return missing
