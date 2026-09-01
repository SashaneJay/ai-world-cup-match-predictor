import streamlit as st


st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About This Project")

st.markdown(
    """
    ## AI FIFA World Cup 2026 Match Predictor

    This project combines historical international football
    data, machine learning and generative AI to analyse
    national teams and predict football match outcomes.

    ### Planned Machine Learning Pipeline

    Historical Match Data  
    ↓  
    Feature Engineering  
    ↓  
    Recent Team Form  
    ↓  
    Model Training  
    ↓  
    Win / Draw / Loss Probabilities  
    ↓  
    AI-Generated Match Analysis

    ### Technologies

    - Python
    - Pandas
    - NumPy
    - scikit-learn
    - Plotly
    - Streamlit
    - OpenAI API

    ### Project Goal

    Build an end-to-end sports analytics platform capable
    of producing data-driven match predictions for the
    2026 FIFA World Cup.
    """
)