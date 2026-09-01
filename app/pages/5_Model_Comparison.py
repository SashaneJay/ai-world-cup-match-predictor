import plotly.express as px
import streamlit as st

from utils.data_loader import load_matches
from utils.model_training import compare_models


st.set_page_config(
    page_title="Model Comparison",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Machine Learning Model Comparison")

st.write(
    "Compare multiple machine learning algorithms using the "
    "same chronological training and testing data."
)

df = load_matches()

with st.spinner("Training and evaluating models..."):
    results = compare_models(df)

st.success("Model comparison completed.")

# Display results as percentages
display_results = results.copy()

display_results["Accuracy"] = (
    display_results["Accuracy"] * 100
).round(2)

display_results["Macro F1"] = (
    display_results["Macro F1"] * 100
).round(2)

st.subheader("📊 Model Results")

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)

best_model = results.iloc[0]

st.subheader("🏆 Best Model")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model",
    best_model["Model"]
)

col2.metric(
    "Accuracy",
    f"{best_model['Accuracy']:.2%}"
)

col3.metric(
    "Macro F1",
    f"{best_model['Macro F1']:.2%}"
)

st.divider()

st.subheader("📈 Performance Comparison")

chart_data = results.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Macro F1"],
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    chart_data,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="Machine Learning Model Performance"
)

fig.update_yaxes(
    tickformat=".0%"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info(
    "Models are evaluated on chronologically newer matches "
    "to better simulate predictions on future football matches."
)