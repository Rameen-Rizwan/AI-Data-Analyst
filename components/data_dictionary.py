import streamlit as st
import pandas as pd


def show_data_dictionary(df):

    st.subheader("📚 Data Dictionary")

    dictionary = pd.DataFrame({

        "Column": df.columns,

        "Data Type": df.dtypes.astype(str).values,

        "Non-Null Values": df.count().values,

        "Missing Values": df.isnull().sum().values,

        "Unique Values": df.nunique().values,

        "Example Value": [

            str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "-"

            for col in df.columns

        ]

    })

    st.dataframe(

        dictionary,

        use_container_width=True,

        hide_index=True

    )