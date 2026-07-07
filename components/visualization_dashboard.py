import streamlit as st

from src.chart_style import COLOR_THEMES
from src.visualization import (
    create_bar_chart,
    create_line_chart,
    create_histogram,
    create_scatter_plot,
    create_pie_chart,
    create_box_plot,
    create_violin_plot,
    create_correlation_heatmap
)


def show_visualization_dashboard(df):

    st.header("📈 Visualization Studio")

    st.markdown("---")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    all_columns = df.columns.tolist()

    left, right = st.columns([1, 2])

    with left:

        st.subheader("⚙️ Visualization Controls")

        chart_type = st.selectbox(

            "Chart Type",

            [

                "Bar Chart",

                "Line Chart",

                "Histogram",

                "Scatter Plot",

                "Pie Chart",

                "Box Plot",

                "Violin Plot",

                "Correlation Heatmap"

            ]

        )

        theme = st.selectbox(

            "🎨 Color Theme",

            list(COLOR_THEMES.keys())

        )

        st.divider()

        x = None
        y = None
        column = None

        if chart_type == "Histogram":

            column = st.selectbox(

                "Select Column",

                numeric_columns

            )

        elif chart_type == "Pie Chart":

            column = st.selectbox(

                "Select Column",

                categorical_columns

            )

        elif chart_type == "Box Plot":

            column = st.selectbox(

                "Select Column",

                numeric_columns

            )

        elif chart_type == "Violin Plot":

            column = st.selectbox(

                "Select Column",

                numeric_columns

            )

        elif chart_type == "Correlation Heatmap":

            st.info("Heatmap will use all numerical columns.")

        elif chart_type == "Scatter Plot":

            x = st.selectbox(

                "X Axis",

                numeric_columns

            )

            y = st.selectbox(

                "Y Axis",

                numeric_columns,

                index=1 if len(numeric_columns) > 1 else 0

            )

        else:

            x = st.selectbox(

                "X Axis",

                all_columns

            )

            y = st.selectbox(

                "Y Axis",

                numeric_columns

            )

        generate = st.button(

            "🚀 Generate Chart",

            use_container_width=True

        )

    with right:

        st.subheader("📊 Chart Preview")

        if generate:

            if chart_type == "Bar Chart":

                fig = create_bar_chart(

                    df,

                    x,

                    y,

                    theme

                )

            elif chart_type == "Line Chart":

                fig = create_line_chart(

                    df,

                    x,

                    y,

                    theme

                )

            elif chart_type == "Histogram":

                fig = create_histogram(

                    df,

                    column,

                    theme

                )

            elif chart_type == "Scatter Plot":

                fig = create_scatter_plot(

                    df,

                    x,

                    y,

                    theme

                )

            elif chart_type == "Pie Chart":

                fig = create_pie_chart(

                    df,

                    column,

                    theme

                )

            elif chart_type == "Box Plot":

                fig = create_box_plot(

                    df,

                    column,

                    theme

                )

            elif chart_type == "Violin Plot":

                fig = create_violin_plot(

                    df,

                    column,

                    theme

                )

            elif chart_type == "Correlation Heatmap":

                fig = create_correlation_heatmap(

                    df

                )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

        else:

            st.info("Click **Generate Chart** to create a visualization.")

    st.markdown("---")

    st.subheader("💡 Suggested Charts")

    suggestions = []

    if len(numeric_columns) > 0:

        suggestions.append("📊 Histogram")

        suggestions.append("📦 Box Plot")

        suggestions.append("🎻 Violin Plot")

    if len(categorical_columns) > 0:

        suggestions.append("🥧 Pie Chart")

        suggestions.append("📊 Bar Chart")

    if len(numeric_columns) >= 2:

        suggestions.append("📈 Scatter Plot")

        suggestions.append("🔥 Correlation Heatmap")

    cols = st.columns(2)

    for index, suggestion in enumerate(suggestions):

        with cols[index % 2]:

            st.success(suggestion)