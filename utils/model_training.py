import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score


FEATURE_COLUMNS = [
    "home_win_rate",
    "away_win_rate",
    "home_goals_for",
    "away_goals_for",
    "home_goals_against",
    "away_goals_against",
    "win_rate_difference",
    "goal_difference_difference",
    "neutral_numeric",
]


def create_ml_dataset(df, form_window=10):
    data = df.copy()

    data = data.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    )

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values("date").reset_index(drop=True)

    # Store each team's historical results.
    history = {}

    training_rows = []

    for _, match in data.iterrows():

        home_team = match["home_team"]
        away_team = match["away_team"]

        home_history = history.get(home_team, [])
        away_history = history.get(away_team, [])

        # Require some previous history for both teams.
        if len(home_history) >= 5 and len(away_history) >= 5:

            home_recent = home_history[-form_window:]
            away_recent = away_history[-form_window:]

            home_games = len(home_recent)
            away_games = len(away_recent)

            home_wins = sum(
                game["result"] == "win"
                for game in home_recent
            )

            away_wins = sum(
                game["result"] == "win"
                for game in away_recent
            )

            home_win_rate = home_wins / home_games
            away_win_rate = away_wins / away_games

            home_goals_for = np.mean(
                [
                    game["goals_for"]
                    for game in home_recent
                ]
            )

            away_goals_for = np.mean(
                [
                    game["goals_for"]
                    for game in away_recent
                ]
            )

            home_goals_against = np.mean(
                [
                    game["goals_against"]
                    for game in home_recent
                ]
            )

            away_goals_against = np.mean(
                [
                    game["goals_against"]
                    for game in away_recent
                ]
            )

            home_goal_difference = (
                home_goals_for
                - home_goals_against
            )

            away_goal_difference = (
                away_goals_for
                - away_goals_against
            )

            if match["home_score"] > match["away_score"]:
                result = "home_win"

            elif match["home_score"] < match["away_score"]:
                result = "away_win"

            else:
                result = "draw"

            training_rows.append(
                {
                    "date": match["date"],

                    "home_win_rate":
                        home_win_rate,

                    "away_win_rate":
                        away_win_rate,

                    "home_goals_for":
                        home_goals_for,

                    "away_goals_for":
                        away_goals_for,

                    "home_goals_against":
                        home_goals_against,

                    "away_goals_against":
                        away_goals_against,

                    "win_rate_difference":
                        home_win_rate
                        - away_win_rate,

                    "goal_difference_difference":
                        home_goal_difference
                        - away_goal_difference,

                    "neutral_numeric":
                        int(match["neutral"]),

                    "result":
                        result,
                }
            )

        # Update histories only AFTER creating features.
        home_score = match["home_score"]
        away_score = match["away_score"]

        if home_score > away_score:
            home_result = "win"
            away_result = "loss"

        elif home_score < away_score:
            home_result = "loss"
            away_result = "win"

        else:
            home_result = "draw"
            away_result = "draw"

        history.setdefault(
            home_team,
            []
        ).append(
            {
                "result": home_result,
                "goals_for": home_score,
                "goals_against": away_score,
            }
        )

        history.setdefault(
            away_team,
            []
        ).append(
            {
                "result": away_result,
                "goals_for": away_score,
                "goals_against": home_score,
            }
        )

    return pd.DataFrame(training_rows)


def train_random_forest(df):

    ml_df = create_ml_dataset(df)

    X = ml_df[FEATURE_COLUMNS]
    y = ml_df["result"]

    # Time-based split instead of random split.
    split_index = int(
        len(ml_df) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return (
    model,
    accuracy,
    report,
    len(X_train),
    len(X_test)
    )

def compare_models(df):

    ml_df = create_ml_dataset(df)

    X = ml_df[FEATURE_COLUMNS]
    y = ml_df["result"]

    # Same chronological 80/20 split for every model
    split_index = int(len(ml_df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Macro F1": macro_f1,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Macro F1",
        ascending=False
    ).reset_index(drop=True)

    return results_df