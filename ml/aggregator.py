def calculate_final_score(placement_probability, skill_match_score):

    final = (0.3 * placement_probability) + (0.7 * skill_match_score)
    return round(final, 2)
