import streamlit as st
import plotly.express as px


def show_correlation_analysis(df):

    st.subheader("🔥 Correlation Analysis")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:

        st.info("At least two numerical columns are required.")

        return

    correlation = numeric_df.corr()

    fig = px.imshow(

        correlation,

        text_auto=".2f",

        color_continuous_scale="RdBu_r",

        aspect="auto",

        title="Correlation Matrix"

    )

    fig.update_layout(

        template="plotly_white",

        height=650,

        title_x=0.5,

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("📊 Strongest Correlations")

    corr_pairs = (
        correlation.where(~correlation.eq(1))
        .stack()
        .reset_index()
    )

    corr_pairs.columns = [

        "Feature 1",

        "Feature 2",

        "Correlation"

    ]

    corr_pairs["Absolute"] = corr_pairs["Correlation"].abs()

    corr_pairs = corr_pairs.sort_values(

        by="Absolute",

        ascending=False

    )

    corr_pairs = corr_pairs.drop_duplicates(

        subset=["Absolute"]

    )

    st.dataframe(

        corr_pairs.head(10)[

            [

                "Feature 1",

                "Feature 2",

                "Correlation"

            ]

        ],

        use_container_width=True,

        hide_index=True

    )