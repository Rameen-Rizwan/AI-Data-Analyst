import streamlit as st


def show_data_quality_report(df):

    st.subheader("📑 Data Quality Report")

    rows = len(df)
    columns = len(df.columns)

    total_cells = rows * columns

    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_columns = len(
        df.select_dtypes(include="number").columns
    )

    categorical_columns = len(
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    memory_usage = round(
        df.memory_usage(deep=True).sum() / 1024 / 1024,
        2
    )

    completeness = (
        (total_cells - missing_cells) / total_cells
    ) * 100 if total_cells else 0

    duplicate_score = (
        (rows - duplicate_rows) / rows
    ) * 100 if rows else 0

    readiness_score = (
        completeness * 0.7 +
        duplicate_score * 0.3
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "Completeness",

        f"{completeness:.2f}%"

    )

    c2.metric(

        "Duplicate Free",

        f"{duplicate_score:.2f}%"

    )

    c3.metric(

        "Readiness",

        f"{readiness_score:.2f}%"

    )

    st.progress(readiness_score / 100)

    st.markdown("---")

    report = {

        "Rows": rows,

        "Columns": columns,

        "Missing Cells": missing_cells,

        "Duplicate Rows": duplicate_rows,

        "Numeric Columns": numeric_columns,

        "Categorical Columns": categorical_columns,

        "Memory Usage (MB)": memory_usage,

        "Completeness (%)": round(completeness, 2),

        "Readiness Score (%)": round(readiness_score, 2)

    }

    st.dataframe(

        report,

        use_container_width=True

    )

    st.markdown("---")

    if readiness_score >= 95:

        st.success(
            "✅ Dataset is ready for Machine Learning."
        )

    elif readiness_score >= 80:

        st.info(
            "ℹ️ Dataset is suitable for analysis with minor preprocessing."
        )

    elif readiness_score >= 60:

        st.warning(
            "⚠️ Moderate preprocessing is recommended before modeling."
        )

    else:

        st.error(
            "❌ Dataset quality is poor. Clean the dataset before analysis."
        )