import streamlit as st

from src.session_manager import (
    has_dataset,
    get_dataset,
    get_filename
)

from components.ai_report_dashboard import show_ai_report_dashboard

st.set_page_config(

    page_title="AI Report",

    page_icon="📄",

    layout="wide"

)

st.title("📄 AI Report Generator")

if not has_dataset():

    st.warning(

        "Please upload a dataset from the Dashboard page."

    )

    st.stop()

df = get_dataset()

st.success(

    f"Dataset : {get_filename()}"

)

show_ai_report_dashboard(df)