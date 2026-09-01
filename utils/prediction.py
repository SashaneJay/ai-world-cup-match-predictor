def predict_match(team_a_stats, team_b_stats):
    team_a_strength = (
        team_a_stats["win_rate"] * 0.5
        + team_a_stats["avg_goals_for"] * 0.3
        - team_a_stats["avg_goals_against"] * 0.2
    )

    team_b_strength = (
        team_b_stats["win_rate"] * 0.5
        + team_b_stats["avg_goals_for"] * 0.3
        - team_b_stats["avg_goals_against"] * 0.2
    )

    total_strength = team_a_strength + team_b_strength

    if total_strength <= 0:
        team_a_prob = 0.33
        team_b_prob = 0.33
    else:
        team_a_prob = team_a_strength / total_strength
        team_b_prob = team_b_strength / total_strength

    draw_prob = 0.20

    team_a_prob = team_a_prob * (1 - draw_prob)
    team_b_prob = team_b_prob * (1 - draw_prob)

    return {
        "team_a_win": team_a_prob,
        "draw": draw_prob,
        "team_b_win": team_b_prob,
        "team_a_expected_goals": team_a_stats["avg_goals_for"],
        "team_b_expected_goals": team_b_stats["avg_goals_for"],
    }