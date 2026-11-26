"""Tests for the ML pipeline. Run from the project root:  python -m pytest tests/test_ml.py
(Train first if models/ is empty:  python -m ml.train_model)"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from ml.aggregator import calculate_final_score
from ml.predictor import predict_placement_probability
from ml.skill_match_role import get_ordered_missing_skills, skill_match_score

USER = {"CGPA": 7.1, "Age": 21, "Gender": "Male", "Stream": "Computer Science",
        "Internships": 2, "HistoryOfBacklogs": 0}


def test_probability_is_valid():
    p = predict_placement_probability(USER)
    assert 0.0 <= p <= 1.0


def test_better_profile_scores_at_least_as_high():
    strong = dict(USER, CGPA=9.0, Internships=2, HistoryOfBacklogs=0)
    weak = dict(USER, CGPA=5.0, Internships=0, HistoryOfBacklogs=1)
    assert predict_placement_probability(strong) >= predict_placement_probability(weak)


def test_skill_match_score_range_and_missing_order():
    score = skill_match_score(["python", "sql"], "Data Scientist")
    assert 0.0 <= score <= 1.0
    missing = get_ordered_missing_skills(["python", "sql"], "Data Scientist")
    assert "python" not in missing and "sql" not in missing


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        skill_match_score(["python"], "Not A Real Role")


def test_final_score_weighting():
    assert calculate_final_score(1.0, 0.0) == 0.3
    assert calculate_final_score(0.0, 1.0) == 0.7
