from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="AI World Cup 2026 Analytics",
    page_icon="🏆",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "results.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=["home_score", "away_score"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    df["home_score"] = pd.to_numeric(df["home_score"])
    df["away_score"] = pd.to_numeric(df["away_score"])

    df["total_goals"] = (
        df["home_score"] + df["away_score"]
    )

    df["winner"] = df.apply(
        lambda row:
        row["home_team"]
        if row["home_score"] > row["away_score"]
        else row["away_team"]
        if row["away_score"] > row["home_score"]
        else "Draw",
        axis=1
    )

    return df


df = load_data()

teams = sorted(
    set(df["home_team"])
    .union(set(df["away_team"]))
)

st.title("🏆 AI FIFA World Cup 2026 Analytics")

st.markdown(
    """
    Explore international football history, analyse national teams,
    and predict match outcomes using machine learning.
    """
)

st.divider()

total_matches = len(df)
num_teams = len(teams)
start_year = int(df["year"].min())
end_year = int(df["year"].max())
avg_goals = df["total_goals"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "⚽ Matches",
    f"{total_matches:,}"
)

col2.metric(
    "🌍 Teams",
    f"{num_teams:,}"
)

col3.metric(
    "📅 Dataset",
    f"{start_year}–{end_year}"
)

col4.metric(
    "🥅 Goals / Match",
    f"{avg_goals:.2f}"
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("📈 International Football Over Time")

    matches_by_year = (
        df.groupby("year")
        .size()
        .reset_index(name="matches")
    )

    fig_year = px.line(
        matches_by_year,
        x="year",
        y="matches",
        labels={
            "year": "Year",
            "matches": "Matches"
        }
    )

    st.plotly_chart(
        fig_year,
        use_container_width=True
    )

with right:
    st.subheader("🏆 Most Successful Teams")

    wins = (
        df[df["winner"] != "Draw"]["winner"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    wins.columns = ["team", "wins"]

    fig_wins = px.bar(
        wins,
        x="wins",
        y="team",
        orientation="h",
        labels={
            "wins": "Wins",
            "team": "Team"
        }
    )

    fig_wins.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig_wins,
        use_container_width=True
    )

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("⚽ Goals per Match")

    fig_goals = px.histogram(
        df,
        x="total_goals",
        nbins=15,
        labels={
            "total_goals": "Goals"
        }
    )

    st.plotly_chart(
        fig_goals,
        use_container_width=True
    )

with right:
    st.subheader("🌍 Most Played Competitions")

    competitions = (
        df["tournament"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    competitions.columns = [
        "tournament",
        "matches"
    ]

    fig_comp = px.bar(
        competitions,
        x="matches",
        y="tournament",
        orientation="h"
    )

    fig_comp.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig_comp,
        use_container_width=True
    )

st.divider()

st.info(
    "Use the pages in the sidebar to explore teams, "
    "generate match predictions, analyse tournaments, "
    "and inspect machine learning model performance."
)