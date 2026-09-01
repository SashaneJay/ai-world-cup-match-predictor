import pandas as pd


def get_team_stats(df, team):
    matches = df[
        (df["home_team"] == team) |
        (df["away_team"] == team)
    ]

    games = len(matches)

    if games == 0:
        return {
            "games": 0,
            "wins": 0,
            "win_rate": 0,
            "avg_goals_for": 0,
            "avg_goals_against": 0,
        }

    home = matches[matches["home_team"] == team]
    away = matches[matches["away_team"] == team]

    goals_for = home["home_score"].sum() + away["away_score"].sum()
    goals_against = home["away_score"].sum() + away["home_score"].sum()

    wins = (
        (home["home_score"] > home["away_score"]).sum()
        + (away["away_score"] > away["home_score"]).sum()
    )

    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "avg_goals_for": goals_for / games,
        "avg_goals_against": goals_against / games,
    }