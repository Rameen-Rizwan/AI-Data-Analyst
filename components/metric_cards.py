import streamlit as st


def show_metric_cards(df):

    rows = len(df)

    columns = len(df.columns)

    missing = int(df.isnull().sum().sum())

    duplicates = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Rows",

        f"{rows:,}"

    )

    c2.metric(

        "Columns",

        columns

    )

    c3.metric(

        "Missing Values",

        missing

    )

    c4.metric(

        "Duplicate Rows",

        duplicates
    )