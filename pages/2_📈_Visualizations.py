import streamlit as st

from src.session_manager import (
    has_dataset,
    get_dataset,
    get_filename
)

from components.dashboard_header import show_dashboard_header
from components.recommendations import show_recommendations
from components.visualization_dashboard import show_visualization_dashboard

st.set_page_config(
    page_title="Visualization Studio",
    page_icon="📈",
    layout="wide"
)

if not has_dataset():

    st.warning("Upload a dataset from Dashboard first.")

    st.stop()

df = get_dataset()

show_dashboard_header(

    "📈 Visualization Studio",

    "Create beautiful interactive charts from your dataset."

)

col1, col2, col3 = st.columns(3)

col1.metric(

    "Dataset",

    get_filename()

)

col2.metric(

    "Rows",

    len(df)

)

col3.metric(

    "Columns",

    len(df.columns)

)

st.markdown("---")

show_recommendations(df)

st.markdown("---")

show_visualization_dashboard(df)