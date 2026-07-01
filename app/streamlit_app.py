import streamlit as st

st.set_page_config(page_title="AI World Cup Match Predictor")

st.title("⚽ AI World Cup Match Predictor")

st.write(
    "Predict World Cup match outcomes using machine learning and AI explanations."
)

teams = [
    "Argentina",
    "Brazil",
    "England",
    "France",
    "Germany",
    "Portugal",
    "Spain",
    "Netherlands",
    "Belgium",
    "United States",
    "Mexico",
    "Canada",
    "Japan",
    "Morocco",
    "Croatia",
    "Senegal",
]

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Select Team A", teams)

with col2:
    team_b = st.selectbox("Select Team B", teams, index=1)

if team_a == team_b:
    st.warning("Please select two different teams.")
else:
    st.subheader("Match Preview")
    st.write(f"### {team_a} vs {team_b}")

    if st.button("Predict Match"):
        st.info("Machine learning prediction model coming next.")