import streamlit as st


def show_dataset_preview(df):

    with st.expander("📂 Dataset Preview", expanded=True):

        st.dataframe(
            df,
            use_container_width=True
        )