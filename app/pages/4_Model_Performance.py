import streamlit as st

from utils.data_loader import load_matches
from utils.model_training import train_random_forest


st.set_page_config(
    page_title="Model Performance",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Machine Learning Model Performance")

df = load_matches()

with st.spinner("Training Random Forest model..."):
    (
        model,
        accuracy,
        report,
        train_size,
        test_size
    ) = train_random_forest(df)

st.success("Model trained successfully.")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model",
    "Random Forest"
)

col2.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col3.metric(
    "Training Samples",
    f"{train_size:,}"
)

st.caption(
    f"Evaluation performed on {test_size:,} "
    "chronologically newer matches."
)

st.divider()

st.subheader("Classification Performance")

class_names = [
    "home_win",
    "draw",
    "away_win"
]

for class_name in class_names:
    if class_name in report:

        st.markdown(
            f"### {class_name.replace('_', ' ').title()}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Precision",
            f"{report[class_name]['precision']:.2%}"
        )

        c2.metric(
            "Recall",
            f"{report[class_name]['recall']:.2%}"
        )

        c3.metric(
            "F1 Score",
            f"{report[class_name]['f1-score']:.2%}"
        )