import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_matches
from utils.model_training import (
    build_prediction_features,
    train_best_model,
)
from utils.world_cup import (
    WORLD_CUP_2026_TEAMS,
    get_dataset_team_name,
)


st.set_page_config(
    page_title="FIFA World Cup 2026 Match Predictor",
    page_icon="🏆",
    layout="wide",
)


st.title("🏆 FIFA World Cup 2026 Match Predictor")

st.write(
    "Use machine learning and historical international football data "
    "to estimate win, draw and loss probabilities for World Cup matchups."
)


df = load_matches()

all_teams = sorted(
    set(df["home_team"]).union(
        set(df["away_team"])
    )
)


world_cup_teams = []

for team in WORLD_CUP_2026_TEAMS:
    dataset_name = get_dataset_team_name(team)

    if dataset_name in all_teams:
        world_cup_teams.append(team)

missing_world_cup_teams = [
    team
    for team in WORLD_CUP_2026_TEAMS
    if get_dataset_team_name(team) not in all_teams
]


@st.cache_resource
def load_model():
    return train_best_model(df)


model = load_model()


st.divider()

st.subheader("🌎 Prediction Mode")

prediction_mode = st.radio(
    "Choose teams from:",
    [
        "2026 World Cup",
        "All International Teams",
    ],
    horizontal=True,
)


if prediction_mode == "2026 World Cup":

    selectable_teams = world_cup_teams

    st.caption(
        f"{len(world_cup_teams)} World Cup teams are currently "
        "recognised in the historical dataset."
    )

    if missing_world_cup_teams:
        with st.expander("Teams not matched to historical dataset"):
            st.write(missing_world_cup_teams)

else:

    selectable_teams = all_teams

col1, col2 = st.columns(2)


with col1:

    default_a = (
        selectable_teams.index("Brazil")
        if "Brazil" in selectable_teams
        else 0
    )

    team_a = st.selectbox(
        "Team A",
        selectable_teams,
        index=default_a,
    )


with col2:

    default_b = (
        selectable_teams.index("France")
        if "France" in selectable_teams
        else min(1, len(selectable_teams) - 1)
    )

    team_b = st.selectbox(
        "Team B",
        selectable_teams,
        index=default_b,
    )


st.divider()

if team_a == team_b:

    st.warning(
        "Please select two different teams."
    )


else:

    st.markdown(
        f"## {team_a} ⚔️ {team_b}"
    )

    if st.button(
        "Predict Match",
        type="primary",
    ):

        with st.spinner(
            "Analysing recent form and generating prediction..."
        ):

            dataset_team_a = get_dataset_team_name(team_a)
            dataset_team_b = get_dataset_team_name(team_b)

            features = build_prediction_features(
                df,
                dataset_team_a,
                dataset_team_b,
            )

            probabilities = model.predict_proba(
                features
            )[0]

            classes = model.classes_

            probability_map = dict(
                zip(
                    classes,
                    probabilities,
                )
            )

        home_probability = probability_map.get(
            "home_win",
            0,
        )

        draw_probability = probability_map.get(
            "draw",
            0,
        )

        away_probability = probability_map.get(
            "away_win",
            0,
        )

        predicted_class = model.predict(
            features
        )[0]


        result_labels = {
            "home_win": f"{team_a} Win",
            "draw": "Draw",
            "away_win": f"{team_b} Win",
        }

        st.subheader("🎯 Prediction")

        prediction_col1, prediction_col2, prediction_col3 = (
            st.columns(3)
        )


        prediction_col1.metric(
            f"{team_a} Win",
            f"{home_probability:.1%}",
        )

        prediction_col2.metric(
            "Draw",
            f"{draw_probability:.1%}",
        )

        prediction_col3.metric(
            f"{team_b} Win",
            f"{away_probability:.1%}",
        )


        st.success(
            f"Predicted Result: "
            f"{result_labels[predicted_class]}"
        )


        st.caption(
            "Prediction generated using Logistic Regression with "
            "recent team form, scoring statistics and Elo team-strength ratings."
        )

        st.divider()

        st.subheader(
            "📊 Prediction Probabilities"
        )


        probability_df = pd.DataFrame(
            {
                "Outcome": [
                    f"{team_a} Win",
                    "Draw",
                    f"{team_b} Win",
                ],
                "Probability": [
                    home_probability,
                    draw_probability,
                    away_probability,
                ],
            }
        )


        probability_chart = px.bar(
            probability_df,
            x="Outcome",
            y="Probability",
            text_auto=".1%",
            title=(
                f"{team_a} vs {team_b} "
                "Outcome Probabilities"
            ),
        )


        probability_chart.update_yaxes(
            tickformat=".0%",
            range=[0, 1],
            title="Probability",
        )


        probability_chart.update_layout(
            xaxis_title="Match Outcome",
            showlegend=False,
        )


        st.plotly_chart(
            probability_chart,
            use_container_width=True,
        )

        st.divider()

        st.subheader("⚡ Team Strength (Elo Rating)")

        home_elo = features.iloc[0]["home_elo"]
        away_elo = features.iloc[0]["away_elo"]
        elo_difference = features.iloc[0]["elo_difference"]

        elo_col1, elo_col2, elo_col3 = st.columns(3)

        elo_col1.metric(
            f"{team_a} Elo",
            f"{home_elo:.0f}"
        )

        elo_col2.metric(
            f"{team_b} Elo",
            f"{away_elo:.0f}"
        )

        if elo_difference > 0:
            stronger_team = team_a
            advantage = elo_difference

        elif elo_difference < 0:
            stronger_team = team_b
            advantage = abs(elo_difference)

        else:
            stronger_team = "Even"
            advantage = 0

        elo_col3.metric(
            "Elo Advantage",
            f"{advantage:.0f}",
            help=f"Current Elo advantage for {stronger_team}"
        )

        if stronger_team != "Even":
            st.caption(
                f"📈 {stronger_team} has a {advantage:.0f}-point "
                "Elo rating advantage based on historical international results."
            )
        else:
            st.caption(
                "Both teams currently have the same Elo rating."
            )

        st.subheader(
            "🔥 Recent Form Comparison"
        )


        home_form = {
            "Win Rate":
                features.iloc[0]["home_win_rate"],

            "Goals Scored / Game":
                features.iloc[0]["home_goals_for"],

            "Goals Conceded / Game":
                features.iloc[0]["home_goals_against"],
        }


        away_form = {
            "Win Rate":
                features.iloc[0]["away_win_rate"],

            "Goals Scored / Game":
                features.iloc[0]["away_goals_for"],

            "Goals Conceded / Game":
                features.iloc[0]["away_goals_against"],
        }


        form_col1, form_col2 = st.columns(2)


        with form_col1:

            st.markdown(
                f"### {team_a}"
            )

            st.metric(
                "Recent Win Rate",
                f"{home_form['Win Rate']:.1%}",
            )

            st.metric(
                "Goals Scored / Game",
                f"{home_form['Goals Scored / Game']:.2f}",
            )

            st.metric(
                "Goals Conceded / Game",
                f"{home_form['Goals Conceded / Game']:.2f}",
            )


        with form_col2:

            st.markdown(
                f"### {team_b}"
            )

            st.metric(
                "Recent Win Rate",
                f"{away_form['Win Rate']:.1%}",
            )

            st.metric(
                "Goals Scored / Game",
                f"{away_form['Goals Scored / Game']:.2f}",
            )

            st.metric(
                "Goals Conceded / Game",
                f"{away_form['Goals Conceded / Game']:.2f}",
            )


        st.subheader(
            "📈 Statistical Comparison"
        )


        comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "Win Rate",
                    "Goals Scored / Game",
                    "Goals Conceded / Game",
                ],

                team_a: [
                    home_form["Win Rate"],
                    home_form["Goals Scored / Game"],
                    home_form["Goals Conceded / Game"],
                ],

                team_b: [
                    away_form["Win Rate"],
                    away_form["Goals Scored / Game"],
                    away_form["Goals Conceded / Game"],
                ],
            }
        )


        comparison_long = comparison_df.melt(
            id_vars="Metric",
            var_name="Team",
            value_name="Value",
        )


        comparison_chart = px.bar(
            comparison_long,
            x="Metric",
            y="Value",
            color="Team",
            barmode="group",
            title="Recent Team Performance",
        )


        st.plotly_chart(
            comparison_chart,
            use_container_width=True,
        )

        st.divider()

        st.subheader(
            "🎯 Prediction Confidence"
        )


        highest_probability = max(
            home_probability,
            draw_probability,
            away_probability,
        )


        if highest_probability >= 0.60:

            confidence = "High"

        elif highest_probability >= 0.45:

            confidence = "Moderate"

        else:

            confidence = "Low"


        confidence_col1, confidence_col2 = (
            st.columns(2)
        )


        confidence_col1.metric(
            "Confidence Level",
            confidence,
        )


        confidence_col2.metric(
            "Highest Probability",
            f"{highest_probability:.1%}",
        )


        if confidence == "High":

            st.success(
                "The model shows a relatively strong "
                "preference for one match outcome."
            )

        elif confidence == "Moderate":

            st.info(
                "The model shows a moderate preference, "
                "but considerable uncertainty remains."
            )

        else:

            st.warning(
                "The model considers this matchup relatively "
                "uncertain. Multiple outcomes have meaningful "
                "probabilities."
            )

        st.divider()

        with st.expander(
            "🧠 How was this prediction generated?"
        ):

            st.markdown(
                """
                **Machine Learning Model:** Logistic Regression

                The model was selected after comparing:

                - Logistic Regression
                - Random Forest
                - Gradient Boosting

                Logistic Regression produced the strongest
                **Macro F1 score**, indicating the most balanced
                performance across home wins, draws and away wins.

                ### Prediction Features

                The model uses:

                - Recent win rate
                - Recent goals scored per game
                - Recent goals conceded per game
                - Difference in team win rates
                - Difference in recent goal difference
                - Neutral venue indicator

                Team form is calculated using each team's
                most recent international matches.

                ### Model Evaluation

                Training and testing are separated
                chronologically so the model is evaluated
                against newer matches rather than randomly
                selected historical games.
                """
            )


        st.caption(
            "⚠️ Predictions are statistical estimates for "
            "educational purposes and should not be interpreted "
            "as guaranteed sporting outcomes."
        )