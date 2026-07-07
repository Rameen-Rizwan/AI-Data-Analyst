import streamlit as st
import plotly.express as px


def show_overview_charts(df):

    st.subheader("📊 Dataset Overview")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    col1, col2 = st.columns(2)

    # ======================================================
    # Histogram
    # ======================================================

    with col1:

        if len(numeric_columns) > 0:

            st.markdown("### 📈 Distribution")

            fig = px.histogram(

                df,

                x=numeric_columns[0],

                template="plotly_white",

                color_discrete_sequence=["#2563EB"]

            )

            fig.update_layout(

                height=400,

                title=numeric_columns[0],

                title_x=0.5

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

    # ======================================================
    # Box Plot
    # ======================================================

    with col2:

        if len(numeric_columns) > 0:

            st.markdown("### 📦 Outliers")

            fig = px.box(

                df,

                y=numeric_columns[0],

                template="plotly_white",

                color_discrete_sequence=["#10B981"]

            )

            fig.update_layout(

                height=400,

                title=numeric_columns[0],

                title_x=0.5

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

    st.markdown("---")

    col3, col4 = st.columns(2)

    # ======================================================
    # Bar Chart
    # ======================================================

    with col3:

        if len(categorical_columns) > 0:

            st.markdown("### 📊 Category Distribution")

            counts = (
                df[categorical_columns[0]]
                .value_counts()
                .head(10)
                .reset_index()
            )

            counts.columns = [

                categorical_columns[0],

                "Count"

            ]

            fig = px.bar(

                counts,

                x=categorical_columns[0],

                y="Count",

                color=categorical_columns[0],

                template="plotly_white"

            )

            fig.update_layout(

                height=400,

                showlegend=False,

                title=categorical_columns[0],

                title_x=0.5

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

    # ======================================================
    # Pie Chart
    # ======================================================

    with col4:

        if len(categorical_columns) > 0:

            st.markdown("### 🥧 Category Proportion")

            counts = (
                df[categorical_columns[0]]
                .value_counts()
                .head(8)
                .reset_index()
            )

            counts.columns = [

                categorical_columns[0],

                "Count"

            ]

            fig = px.pie(

                counts,

                names=categorical_columns[0],

                values="Count",

                hole=0.45,

                template="plotly_white"

            )

            fig.update_layout(

                height=400,

                title=categorical_columns[0],

                title_x=0.5

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )