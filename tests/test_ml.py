import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ml.predictor import predict_placement_probability
from ml.skill_match_role import skill_match_score, get_ordered_missing_skills
from ml.aggregator import calculate_final_score


user_academic_data = {
    "CGPA": 7.1,
    "Age": 21,
    "Gender": "Male",
    "Stream": "Computer Science",
    "Internships": 2,
    "HistoryOfBacklogs": 0
}

placement_prob = predict_placement_probability(user_academic_data)
print("Placement Probability:", placement_prob)

role = "Data Scientist"
user_skills = ["python", "sql", "statistics"]

"""role_skill_map = {
    "must": {
        "python",
        "sql",
        "statistics",
        "machine learning",
        "data cleaning"
    },
    "good": {
        "data visualization",
        "feature engineering",
        "model evaluation"
    },
    "bonus": {
        "nlp",
        "deep learning"
    }
}"""

skill_score = skill_match_score(user_skills, role)
missing_skills = get_ordered_missing_skills(user_skills, role)

print("Skill Match Score:", skill_score)
print("Missing Skills (ordered):", missing_skills)

final_score = calculate_final_score(placement_prob, skill_score)
print("Final Score:", final_score)
