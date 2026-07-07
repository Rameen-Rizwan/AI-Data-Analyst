import streamlit as st


def show_recommendations(df):

    st.subheader("💡 Smart Chart Recommendations")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    recommendations = []

    # ======================================================
    # Histogram
    # ======================================================

    for column in numeric_columns:

        recommendations.append({

            "Chart": "📊 Histogram",

            "Column": column,

            "Reason": "Visualize the distribution of numerical values."

        })

    # ======================================================
    # Box Plot
    # ======================================================

    for column in numeric_columns:

        recommendations.append({

            "Chart": "📦 Box Plot",

            "Column": column,

            "Reason": "Detect outliers and spread."

        })

    # ======================================================
    # Violin Plot
    # ======================================================

    for column in numeric_columns:

        recommendations.append({

            "Chart": "🎻 Violin Plot",

            "Column": column,

            "Reason": "View data distribution."

        })

    # ======================================================
    # Pie Chart
    # ======================================================

    for column in categorical_columns:

        recommendations.append({

            "Chart": "🥧 Pie Chart",

            "Column": column,

            "Reason": "See category proportions."

        })

    # ======================================================
    # Bar Chart
    # ======================================================

    for column in categorical_columns:

        recommendations.append({

            "Chart": "📈 Bar Chart",

            "Column": column,

            "Reason": "Compare category frequencies."

        })

    # ======================================================
    # Scatter Plot
    # ======================================================

    if len(numeric_columns) >= 2:

        recommendations.append({

            "Chart": "📍 Scatter Plot",

            "Column": f"{numeric_columns[0]} vs {numeric_columns[1]}",

            "Reason": "Explore relationship."

        })

    # ======================================================
    # Correlation Heatmap
    # ======================================================

    if len(numeric_columns) >= 2:

        recommendations.append({

            "Chart": "🔥 Correlation Heatmap",

            "Column": "All Numerical Columns",

            "Reason": "Check feature correlation."

        })

    # ======================================================
    # Display
    # ======================================================

    st.dataframe(

        recommendations,

        use_container_width=True,

        hide_index=True

    )