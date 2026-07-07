import streamlit as st

from src.ai_engine import (
    dataset_summary,
    analyze_missing_values,
    analyze_duplicates,
    detect_outliers,
    preprocessing_recommendations,
    recommend_ml_problem
)

from src.gemini_engine import ask_gemini


def show_ai_dashboard(df):

    st.header("🤖 AI Data Copilot")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(

        [

            "📋 Summary",

            "🧹 Data Quality",

            "💡 AI Recommendations",

            "🧠 ML Advisor",

            "💬 Gemini AI"

        ]

    )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    with tab1:

        summary = dataset_summary(df)

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Rows", summary["Rows"])

        c2.metric("Columns", summary["Columns"])

        c3.metric("Numeric", summary["Numerical Columns"])

        c4.metric("Categorical", summary["Categorical Columns"])

        c5.metric("Memory (MB)", summary["Memory Usage"])

    # ==========================================================
    # DATA QUALITY
    # ==========================================================

    with tab2:

        missing = analyze_missing_values(df)

        if missing["status"]:

            st.success(missing["message"])

        else:

            st.dataframe(

                missing["data"],

                use_container_width=True

            )

        st.markdown("---")

        duplicates = analyze_duplicates(df)

        st.metric(

            "Duplicate Rows",

            duplicates

        )

        st.markdown("---")

        outliers = detect_outliers(df)

        st.subheader("Outlier Summary")

        st.dataframe(

            outliers,

            use_container_width=True

        )

    # ==========================================================
    # RECOMMENDATIONS
    # ==========================================================

    with tab3:

        recommendations = preprocessing_recommendations(df)

        for recommendation in recommendations:

            st.info(recommendation)

    # ==========================================================
    # ML ADVISOR
    # ==========================================================

    with tab4:

        target = st.selectbox(

            "Target Column",

            df.columns

        )

        result = recommend_ml_problem(

            df,

            target

        )

        st.success(

            f"Problem Type : {result['Problem']}"

        )

        st.subheader("Recommended Models")

        for model in result["Models"]:

            st.write(f"✅ {model}")

    # ==========================================================
    # GEMINI AI
    # ==========================================================

    from src.smart_questions import SMART_QUESTIONS

    # ==========================================================
    # GEMINI AI
    # ==========================================================

    with tab5:

        st.subheader("💬 AI Dataset Assistant")

        api_key = st.text_input(

            "Gemini API Key",

            type="password"

        )

        st.markdown("### ⚡ Smart Questions")

        selected = st.selectbox(

            "Choose a predefined AI task",

            ["Custom Question"] + list(SMART_QUESTIONS.keys())

        )

        if selected == "Custom Question":

            question = st.text_area(

                "Ask your own question",

                height=150

            )

        else:

            question = SMART_QUESTIONS[selected]

            st.info(question)

        if st.button(

            "🚀 Analyze with AI",

            type="primary",

            use_container_width=True

        ):

            if api_key == "":

                st.error("Please enter your Gemini API Key.")

            elif question.strip() == "":

                st.warning("Please enter a question.")

            else:

                with st.spinner("Analyzing dataset..."):

                    try:

                        answer = ask_gemini(

                            api_key,

                            df,

                            question

                        )

                        st.success("Analysis Complete")

                        st.markdown(answer)

                    except Exception as e:

                        st.error(str(e))