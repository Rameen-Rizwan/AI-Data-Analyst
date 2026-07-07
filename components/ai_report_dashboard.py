import streamlit as st

from src.report_generator import (
    dataset_report,
    data_quality_report,
    executive_summary,
    generate_pdf
)


def show_ai_report_dashboard(df):

    st.header("📄 AI Report Generator")

    st.caption(
        "Generate a professional AI-powered report for your dataset."
    )

    tab1, tab2, tab3, tab4 = st.tabs(

        [

            "📊 Dataset",

            "🧹 Quality",

            "🤖 AI Summary",

            "📄 Export"

        ]

    )

    # ==========================================================
    # DATASET
    # ==========================================================

    with tab1:

        report = dataset_report(df)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(

            "Rows",

            report["Rows"]

        )

        c2.metric(

            "Columns",

            report["Columns"]

        )

        c3.metric(

            "Numeric",

            report["Numeric"]

        )

        c4.metric(

            "Categorical",

            report["Categorical"]

        )

        st.markdown("---")

        st.dataframe(

            report["Table"],

            use_container_width=True,

            hide_index=True

        )

    # ==========================================================
    # QUALITY
    # ==========================================================

    with tab2:

        quality = data_quality_report(df)

        st.dataframe(

            quality,

            use_container_width=True,

            hide_index=True

        )

    # ==========================================================
    # AI SUMMARY
    # ==========================================================

    with tab3:

        summary = executive_summary(df)

        st.success(summary)

    # ==========================================================
    # PDF
    # ==========================================================

    with tab4:

        st.subheader("Download Professional Report")

        pdf = generate_pdf(df)

        st.download_button(

            "📥 Download PDF Report",

            pdf,

            file_name="AI_Data_Report.pdf",

            mime="application/pdf",

            use_container_width=True

        )