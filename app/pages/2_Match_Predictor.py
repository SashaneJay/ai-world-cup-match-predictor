import streamlit as st

from utils.data_loader import load_matches
from utils.model_training import (
    build_prediction_features,
    train_best_model,
)


st.set_page_config(
    page_title="Match Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI World Cup Match Predictor")

st.write(
    "Predict international match outcomes using a "
    "machine learning model trained on historical football data."
)

df = load_matches()

teams = sorted(
    set(df["home_team"])
    .union(set(df["away_team"]))
)


@st.cache_resource
def load_model():
    return train_best_model(df)


model = load_model()

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox(
        "Team A",
        teams,
        index=teams.index("Brazil")
        if "Brazil" in teams
        else 0
    )

with col2:
    team_b = st.selectbox(
        "Team B",
        teams,
        index=teams.index("France")
        if "France" in teams
        else 1
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
        type="primary"
    ):

        features = build_prediction_features(
            df,
            team_a,
            team_b
        )

        probabilities = model.predict_proba(
            features
        )[0]

        classes = model.classes_

        probability_map = dict(
            zip(
                classes,
                probabilities
            )
        )

        home_probability = probability_map.get(
            "home_win",
            0
        )

        draw_probability = probability_map.get(
            "draw",
            0
        )

        away_probability = probability_map.get(
            "away_win",
            0
        )

        predicted_class = model.predict(
            features
        )[0]

        result_labels = {
            "home_win": f"{team_a} Win",
            "draw": "Draw",
            "away_win": f"{team_b} Win",
        }

        st.subheader("Prediction")

        p1, p2, p3 = st.columns(3)

        p1.metric(
            f"{team_a} Win",
            f"{home_probability:.1%}"
        )

        p2.metric(
            "Draw",
            f"{draw_probability:.1%}"
        )

        p3.metric(
            f"{team_b} Win",
            f"{away_probability:.1%}"
        )

        st.success(
            f"Predicted Result: "
            f"{result_labels[predicted_class]}"
        )

        import pandas as pd
        import plotly.express as px

        st.divider()

        st.subheader("📊 Prediction Probabilities")

        probability_df = pd.DataFrame(
            {
                "Outcome": [
                    f"{team_a} Win",
                    "Draw",
                    f"{team_b} Win"
                ],
                "Probability": [
                    home_probability,
                    draw_probability,
                    away_probability
                ]
            }
        )

        fig_probability = px.bar(
            probability_df,
            x="Outcome",
            y="Probability",
            text_auto=".1%",
            title="Predicted Match Outcome Probabilities"
        )

        fig_probability.update_yaxes(
            tickformat=".0%",
            range=[0, 1]
        )

        st.plotly_chart(
            fig_probability,
            use_container_width=True
        )
        st.subheader("🔥 Recent Form Comparison")

        home_form = {
            "Win Rate": features.iloc[0]["home_win_rate"],
            "Goals Scored / Game": features.iloc[0]["home_goals_for"],
            "Goals Conceded / Game": features.iloc[0]["home_goals_against"],
        }

        away_form = {
            "Win Rate": features.iloc[0]["away_win_rate"],
            "Goals Scored / Game": features.iloc[0]["away_goals_for"],
            "Goals Conceded / Game": features.iloc[0]["away_goals_against"],
        }

        form_col1, form_col2 = st.columns(2)

        with form_col1:
            st.markdown(f"### 🇧🇷 {team_a}" if team_a == "Brazil" else f"### {team_a}")

            st.metric(
                "Recent Win Rate",
                f"{home_form['Win Rate']:.1%}"
            )

            st.metric(
                "Goals Scored / Game",
                f"{home_form['Goals Scored / Game']:.2f}"
            )

            st.metric(
                "Goals Conceded / Game",
                f"{home_form['Goals Conceded / Game']:.2f}"
            )

        with form_col2:
            st.markdown(f"### 🇫🇷 {team_b}" if team_b == "France" else f"### {team_b}")

            st.metric(
                "Recent Win Rate",
                f"{away_form['Win Rate']:.1%}"
            )

            st.metric(
                "Goals Scored / Game",
                f"{away_form['Goals Scored / Game']:.2f}"
            )

            st.metric(
                "Goals Conceded / Game",
                f"{away_form['Goals Conceded / Game']:.2f}"
            )

            st.divider()

            highest_probability = max(
                home_probability,
                draw_probability,
                away_probability
            )

            if highest_probability >= 0.60:
                confidence = "High"
            elif highest_probability >= 0.45:
                confidence = "Moderate"
            else:
                confidence = "Low"

            st.subheader("🎯 Prediction Confidence")

            st.metric(
                "Confidence Level",
                confidence
            )

            st.caption(
                f"Highest predicted probability: {highest_probability:.1%}"
            )