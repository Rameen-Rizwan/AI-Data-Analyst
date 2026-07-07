import streamlit as st


def show_descriptive_statistics(df):

    st.subheader("📊 Descriptive Statistics")

    st.dataframe(

        df.describe(include="all"),

        use_container_width=True

    )