import streamlit as st


def show_dataset_health(df):

    total_cells = df.shape[0] * df.shape[1]

    missing = int(df.isnull().sum().sum())

    duplicates = int(df.duplicated().sum())

    missing_percentage = (missing / total_cells) * 100 if total_cells else 0

    duplicate_percentage = (duplicates / len(df)) * 100 if len(df) else 0

    health_score = max(
        0,
        100 - (missing_percentage + duplicate_percentage)
    )

    st.subheader("🩺 Dataset Health")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Health Score",
        f"{health_score:.1f}%"
    )

    c2.metric(
        "Missing Cells",
        missing
    )

    c3.metric(
        "Duplicate Rows",
        duplicates
    )

    st.progress(health_score / 100)

    if health_score >= 90:

        st.success("Excellent dataset quality.")

    elif health_score >= 70:

        st.warning("Dataset quality is good but can be improved.")

    else:

        st.error("Dataset requires preprocessing before analysis.")