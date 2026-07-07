import streamlit as st


def show_filter_panel(df):

    st.subheader("🔍 Dataset Filters")

    filtered_df = df.copy()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # ==========================================
    # Categorical Filters
    # ==========================================

    for column in categorical_columns:

        values = st.multiselect(

            column,

            options=sorted(df[column].dropna().unique())

        )

        if values:

            filtered_df = filtered_df[
                filtered_df[column].isin(values)
            ]

    # ==========================================
    # Numerical Filters
    # ==========================================

    for column in numerical_columns:

        minimum = float(df[column].min())
        maximum = float(df[column].max())

        selected = st.slider(

            column,

            min_value=minimum,

            max_value=maximum,

            value=(minimum, maximum)

        )

        filtered_df = filtered_df[
            (filtered_df[column] >= selected[0]) &
            (filtered_df[column] <= selected[1])
        ]

    return filtered_df