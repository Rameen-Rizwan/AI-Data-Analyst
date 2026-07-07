import streamlit as st
import pandas as pd


def show_column_profile(df):

    st.subheader("📋 Column Profile")

    selected_column = st.selectbox(

        "Select a Column",

        df.columns

    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Data Type",

            str(df[selected_column].dtype)

        )

        st.metric(

            "Unique Values",

            df[selected_column].nunique()

        )

        st.metric(

            "Missing Values",

            int(df[selected_column].isnull().sum())

        )

    with col2:

        if pd.api.types.is_numeric_dtype(df[selected_column]):

            st.metric(

                "Minimum",

                round(df[selected_column].min(), 2)

            )

            st.metric(

                "Maximum",

                round(df[selected_column].max(), 2)

            )

            st.metric(

                "Mean",

                round(df[selected_column].mean(), 2)

            )

        else:

            mode = df[selected_column].mode()

            st.metric(

                "Most Frequent",

                mode.iloc[0] if not mode.empty else "-"

            )

            st.metric(

                "Categories",

                df[selected_column].nunique()

            )

    st.markdown("---")

    st.write("Top 10 Values")

    st.dataframe(

        df[selected_column]
        .value_counts(dropna=False)
        .head(10)
        .reset_index()
        .rename(
            columns={
                "index": selected_column,
                selected_column: "Count"
            }
        ),

        use_container_width=True
    )