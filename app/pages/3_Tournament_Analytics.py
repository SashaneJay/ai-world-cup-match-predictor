import plotly.express as px
import streamlit as st

from utils.data_loader import load_matches


st.set_page_config(
    page_title="Tournament Analytics",
    page_icon="🏆",
    layout="wide"
)

df = load_matches()

st.title("🏆 Tournament Analytics")

tournament = st.selectbox(
    "Select tournament",
    sorted(df["tournament"].dropna().unique())
)

filtered = df[
    df["tournament"] == tournament
]

st.metric(
    "Matches",
    len(filtered)
)

teams = set(
    filtered["home_team"]
).union(
    set(filtered["away_team"])
)

st.metric(
    "Teams",
    len(teams)
)

goal_average = (
    filtered["home_score"]
    + filtered["away_score"]
).mean()

st.metric(
    "Average Goals",
    f"{goal_average:.2f}"
)

st.subheader("Matches by Year")

matches_by_year = (
    filtered.groupby("year")
    .size()
    .reset_index(name="matches")
)

fig = px.bar(
    matches_by_year,
    x="year",
    y="matches"
)

st.plotly_chart(
    fig,
    use_container_width=True
)