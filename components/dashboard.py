import streamlit as st


def show_dashboard(summary):

    st.header("📊 Dataset Dashboard")

    st.caption(
        "Explore your uploaded dataset before moving to "
        "visualization, AI analysis, and machine learning."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Rows", summary["Rows"])
    col2.metric("Columns", summary["Columns"])
    col3.metric("Missing Values", summary["Missing Values"])
    col4.metric("Duplicate Rows", summary["Duplicate Rows"])
    col5.metric("Memory (MB)", summary["Memory (MB)"])