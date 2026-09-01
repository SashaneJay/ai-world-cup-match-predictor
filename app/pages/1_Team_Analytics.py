import plotly.express as px
import streamlit as st

from utils.data_loader import load_matches
from utils.feature_engineering import get_team_stats


st.set_page_config(
    page_title="Team Analytics",
    page_icon="📊",
    layout="wide"
)

df = load_matches()

teams = sorted(
    set(df["home_team"])
    .union(set(df["away_team"]))
)

st.title("📊 Team Analytics")

st.write(
    "Explore historical performance, recent form, "
    "scoring trends and competition history."
)

selected_team = st.selectbox(
    "Choose a national team",
    teams,
    index=teams.index("Brazil")
    if "Brazil" in teams
    else 0
)

stats = get_team_stats(
    df,
    selected_team
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Games Played",
    stats["games"]
)

col2.metric(
    "Win Rate",
    f"{stats['win_rate']:.1%}"
)

col3.metric(
    "Goals / Game",
    f"{stats['avg_goals_for']:.2f}"
)

col4.metric(
    "Conceded / Game",
    f"{stats['avg_goals_against']:.2f}"
)

team_matches = df[
    (df["home_team"] == selected_team)
    | (df["away_team"] == selected_team)
].sort_values(
    "date",
    ascending=False
)

recent = team_matches.head(10).copy()


def result_for_team(row):
    if row["home_team"] == selected_team:
        team_score = row["home_score"]
        opponent_score = row["away_score"]
    else:
        team_score = row["away_score"]
        opponent_score = row["home_score"]

    if team_score > opponent_score:
        return "W"

    if team_score < opponent_score:
        return "L"

    return "D"


recent["result"] = recent.apply(
    result_for_team,
    axis=1
)

st.subheader("🔥 Recent Form")

form = "   ".join(
    recent["result"].tolist()
)

st.markdown(
    f"### {form}"
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("⚽ Scoring Performance")

    scoring_df = {
        "Metric": [
            "Goals Scored",
            "Goals Conceded"
        ],
        "Average": [
            stats["avg_goals_for"],
            stats["avg_goals_against"]
        ]
    }

    fig = px.bar(
        scoring_df,
        x="Metric",
        y="Average"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:
    st.subheader("🏆 Competition Breakdown")

    tournament_counts = (
        team_matches["tournament"]
        .value_counts()
        .head(8)
        .reset_index()
    )

    tournament_counts.columns = [
        "Tournament",
        "Matches"
    ]

    fig = px.pie(
        tournament_counts,
        names="Tournament",
        values="Matches"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

st.subheader("📅 Recent Matches")

st.dataframe(
    recent[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "result"
        ]
    ],
    use_container_width=True,
    hide_index=True
)