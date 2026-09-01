import streamlit as st

from utils.data_loader import load_matches


st.set_page_config(
    page_title="Match Predictor",
    page_icon="🤖",
    layout="wide"
)

df = load_matches()

teams = sorted(
    set(df["home_team"])
    .union(set(df["away_team"]))
)

st.title("🤖 AI Match Predictor")

st.write(
    "Compare two national teams and predict "
    "the probability of each match outcome."
)

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
        "Select two different teams."
    )

else:
    st.markdown(
        f"## {team_a}  ⚔️  {team_b}"
    )

    if st.button(
        "Predict Match",
        type="primary"
    ):
        st.info(
            "The trained machine learning model "
            "will be connected here in the next phase."
        )