import streamlit as st


st.set_page_config(
    page_title="Model Performance",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Machine Learning Model Performance")

st.info(
    "This page will display model accuracy, "
    "classification metrics, confusion matrices "
    "and model comparisons once training is complete."
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    "Coming Soon"
)

col2.metric(
    "F1 Score",
    "Coming Soon"
)

col3.metric(
    "Training Samples",
    "Coming Soon"
)