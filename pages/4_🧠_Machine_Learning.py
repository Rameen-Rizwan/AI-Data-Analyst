import streamlit as st

from src.session_manager import (
    has_dataset,
    get_dataset,
    get_filename
)

from components.ml_dashboard import show_ml_dashboard

st.set_page_config(

    page_title="Machine Learning",

    page_icon="🧠",

    layout="wide"

)

st.title("🧠 Machine Learning Studio")

if not has_dataset():

    st.warning("Please upload a dataset from the Dashboard page.")

    st.stop()

df = get_dataset()

st.success(f"Dataset : {get_filename()}")

show_ml_dashboard(df)