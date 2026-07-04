from pathlib import Path
from utils.feature_engineering import get_team_stats
from utils.data_loader import load_matches

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="AI World Cup Match Predictor",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "results.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["total_goals"] = df["home_score"] + df["away_score"]

    df["winner"] = df.apply(
        lambda row: row["home_team"]
        if row["home_score"] > row["away_score"]
        else row["away_team"]
        if row["away_score"] > row["home_score"]
        else "Draw",
        axis=1
    )

    return df


df = load_data()

st.title("🏆 AI FIFA World Cup 2026 Analytics")
st.write(
    "Explore international football history before building a machine learning match prediction model."
)

total_matches = len(df)
teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
num_teams = len(teams)
start_year = df["year"].min()
end_year = df["year"].max()
avg_goals = df["total_goals"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Matches", f"{total_matches:,}")
col2.metric("Countries / Teams", num_teams)
col3.metric("Year Range", f"{start_year} - {end_year}")
col4.metric("Avg Goals / Match", f"{avg_goals:.2f}")

st.divider()

st.subheader("📈 Matches Played Over Time")

matches_by_year = (
    df.groupby("year")
    .size()
    .reset_index(name="matches")
)

fig_year = px.line(
    matches_by_year,
    x="year",
    y="matches",
    title="International Matches by Year"
)

st.plotly_chart(fig_year, use_container_width=True)

st.subheader("🏆 Top Winning Teams")

wins = (
    df[df["winner"] != "Draw"]["winner"]
    .value_counts()
    .head(15)
    .reset_index()
)

wins.columns = ["team", "wins"]

fig_wins = px.bar(
    wins,
    x="team",
    y="wins",
    title="Top 15 Teams by Match Wins"
)

st.plotly_chart(fig_wins, use_container_width=True)

st.subheader("⚽ Goal Distribution")

fig_goals = px.histogram(
    df,
    x="total_goals",
    nbins=15,
    title="Total Goals per Match"
)

st.plotly_chart(fig_goals, use_container_width=True)

st.subheader("🌍 Tournament Analysis")

top_tournaments = (
    df["tournament"]
    .value_counts()
    .head(15)
    .reset_index()
)

top_tournaments.columns = ["tournament", "matches"]

fig_tournaments = px.bar(
    top_tournaments,
    x="tournament",
    y="matches",
    title="Top 15 Tournaments by Number of Matches"
)

st.plotly_chart(fig_tournaments, use_container_width=True)

st.subheader("🔎 Explore Matches")

selected_team = st.selectbox(
    "Select a team",
    teams
)

team_matches = df[
    (df["home_team"] == selected_team)
    | (df["away_team"] == selected_team)
].sort_values("date", ascending=False)

st.write(f"Showing recent matches for **{selected_team}**")

st.dataframe(
    team_matches[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "country",
            "neutral"
        ]
    ].head(20),
    use_container_width=True
)

st.divider()

st.subheader("🤖 Next Phase")

st.info(
    "Next, we will engineer team form features and train a machine learning model to predict match outcomes."
)

df = load_matches()

st.subheader("📊 Team Analytics")

team = st.selectbox(
    "Choose a Team",
    teams
)

stats = get_team_stats(df, team)

st.metric(
    "Games Played",
    stats["games"]
)

st.metric(
    "Win Rate",
    f"{stats['win_rate']:.2%}"
)

st.metric(
    "Goals Scored/Game",
    f"{stats['avg_goals_for']:.2f}"
)

st.metric(
    "Goals Conceded/Game",
    f"{stats['avg_goals_against']:.2f}"
)