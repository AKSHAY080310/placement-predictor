# ml/aggregator.py

def calculate_final_score(placement_probability, skill_match_score):
    """
    Final score logic:
    30% placement probability
    70% skill match score
    """

    final = (0.3 * placement_probability) + (0.7 * skill_match_score)
    return round(final, 2)
