import pandas as pd


def get_team_stats(df, team):

    matches = df[
        (df["home_team"] == team) |
        (df["away_team"] == team)
    ]

    wins = 0
    goals_for = 0
    goals_against = 0

    for _, row in matches.iterrows():

        if row["home_team"] == team:

            goals_for += row["home_score"]
            goals_against += row["away_score"]

            if row["home_score"] > row["away_score"]:
                wins += 1

        else:

            goals_for += row["away_score"]
            goals_against += row["home_score"]

            if row["away_score"] > row["home_score"]:
                wins += 1

    games = len(matches)

    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games if games > 0 else 0,
        "avg_goals_for": goals_for / games if games > 0 else 0,
        "avg_goals_against": goals_against / games if games > 0 else 0
    }