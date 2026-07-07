import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.ml_engine import (
    train_model,
    compare_models,
    predictions_to_csv,
    model_to_bytes,
    model_information,
    suggest_target_column,
    detect_dataset_type,
    clean_dataset,
    generate_business_insights
)


def show_ml_dashboard(df):

    st.header("🧠 Machine Learning Studio")

    st.caption(
        "Train Machine Learning models without writing code."
    )

    dataset_type = detect_dataset_type(df)

    st.success(
        f"📂 Detected Dataset Type: {dataset_type}"
    )

    st.markdown("---")

    # ==========================================================
    # Configuration
    # ==========================================================

    col1, col2 = st.columns(2)

    with col1:

        suggested_target, reason = suggest_target_column(df)

        if suggested_target is not None:

            st.info(
                f"🎯 Suggested Target: **{suggested_target}**\n\nReason: {reason}"
            )

            default_index = list(df.columns).index(suggested_target)

        else:

            default_index = 0

        target = st.selectbox(

            "🎯 Target Column",

            df.columns,

            index=default_index

        )

    with col2:

        target_dtype = df[target].dtype

        if pd.api.types.is_numeric_dtype(target_dtype):

            unique_values = df[target].nunique()

            default_problem = (
                "Classification"
                if unique_values <= 20
                else "Regression"
            )

        else:

            default_problem = "Classification"

        problem_type = st.radio(

            "Problem Type",

            [

                "Classification",

                "Regression"

            ],

            index=0 if default_problem == "Classification" else 1,

            horizontal=True

        )

        st.caption(

            f"💡 Recommended: {default_problem}"

        )

    classification_models = [

        "Logistic Regression",

        "Decision Tree",

        "Random Forest"

    ]

    regression_models = [

        "Linear Regression",

        "Decision Tree",

        "Random Forest"

    ]

    # Validate target compatibility

    if problem_type == "Classification":

        if df[target].nunique() > 50:

            st.warning(
                "This target has many unique values. Regression may be more appropriate."
            )

        model_name = st.selectbox(

            "Model",

            classification_models

        )

    else:

        if not pd.api.types.is_numeric_dtype(df[target]):

            st.error(
                "Regression requires a numeric target column."
            )

            st.stop()

        model_name = st.selectbox(

            "Model",

            regression_models

        )

    test_size = st.slider(

        "Test Size",

        min_value=0.10,

        max_value=0.50,

        value=0.20,

        step=0.05

    )

    with st.expander(

        "⚙️ Advanced Hyperparameters",

        expanded=False

    ):

        auto_tune = st.checkbox(

            "🤖 Automatically Find Best Hyperparameters (Grid Search)",

            value=False

        )

        if model_name == "Random Forest":

            n_estimators = st.slider(

                "Number of Trees",

                50,

                500,

                200,

                50

            )

            max_depth = st.slider(

                "Maximum Depth",

                2,

                50,

                10

            )

        elif model_name == "Decision Tree":

            max_depth = st.slider(

                "Maximum Depth",

                2,

                50,

                10

            )

            min_samples_split = st.slider(

                "Minimum Samples Split",

                2,

                20,

                2

            )

        elif model_name == "Logistic Regression":

            C = st.slider(

                "Regularization (C)",

                0.01,

                10.0,

                1.0

            )

        elif model_name == "Linear Regression":

            fit_intercept = st.checkbox(

                "Fit Intercept",

                value=True

            )

    st.markdown("---")

    st.markdown("---")

    st.subheader("🧹 One-Click Data Cleaning")

    if st.button(

        "Clean Dataset",

        use_container_width=True

    ):

        cleaned_df = clean_dataset(df)

        st.session_state["cleaned_dataset"] = cleaned_df

        st.success(

            "Dataset cleaned successfully."

        )

        st.write(

            f"Rows: {len(df)} → {len(cleaned_df)}"

        )

        st.write(

            f"Columns: {df.shape[1]} → {cleaned_df.shape[1]}"

        )

    if "cleaned_dataset" in st.session_state:

        st.subheader("✨ Cleaned Dataset Preview")

        st.dataframe(

            st.session_state["cleaned_dataset"].head(),

            use_container_width=True

        )

    # ==========================================================
    # Model Information
    # ==========================================================

    info = model_information(

        model_name,

        problem_type

    )

    with st.expander("ℹ Model Information"):

        st.write(

            f"**Problem Type:** {info['Problem Type']}"

        )

        st.write(

            f"**Model:** {info['Model']}"

        )

        st.write(

            info["Description"]

        )

    st.markdown("---")

    # ==========================================================
    # Train
    # ==========================================================

    # Validation before training

    can_train = True

    if target is None:

        st.warning(
            "Please select a target column."
        )
        can_train = False

    elif len(df) < 20:

        st.warning(
            "Dataset must contain at least 20 rows."
        )
        can_train = False

    elif len(df.columns) < 2:

        st.warning(
            "Dataset must contain at least one feature and one target column."
        )
        can_train = False

    train = st.button(

        "🚀 Train Model",

        use_container_width=True,

        type="primary",

        disabled=not can_train

    )

    compare = st.button(

        "📊 Compare All Models",

        use_container_width=True

    )

    if compare:

        comparison = compare_models(

            df,

            target,

            problem_type,

            test_size

        )

        st.markdown("---")

        st.subheader("🏆 Model Comparison")

        st.dataframe(

            comparison,

            use_container_width=True,

            hide_index=True

        )

        successful_models = comparison.dropna(subset=["Score"])

        fig = px.bar(

            successful_models,

            x="Model",

            y="Score",

            color="Score",

            text="Score",

            template="plotly_white"

        )

        fig.update_layout(

            height=450,

            title="Model Comparison",

            title_x=0.5

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        if successful_models.empty:

            st.error("❌ No models could be trained on this dataset.")

            st.dataframe(comparison, use_container_width=True)

            st.stop()

        best_model = successful_models.iloc[0]["Model"]

        best_score = successful_models.iloc[0]["Score"]

        st.success(

            f"🏆 Best Model: **{best_model}** (Score: {best_score:.4f})"

        )

        st.stop()

    if not train:

        return

    try:

        with st.spinner("Training model..."):

            results = train_model(

                df,

                target,

                model_name,

                problem_type,

                test_size,

                hyperparameters=locals()

            )

    except Exception as e:

        st.error(f"❌ {e}")

        st.stop()

    st.success("✅ Model trained successfully and is ready for evaluation.")

    st.markdown("---")

    # ==========================================================
    # Metrics
    # ==========================================================

    if problem_type == "Classification":

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(

            "Accuracy",

            f"{results['accuracy']:.4f}"

        )

        c2.metric(

            "Precision",

            f"{results['precision']:.4f}"

        )

        c3.metric(

            "Recall",

            f"{results['recall']:.4f}"

        )

        c4.metric(

            "F1 Score",

            f"{results['f1']:.4f}"

        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(

            "R²",

            f"{results['r2']:.4f}"

        )

        c2.metric(

            "MAE",

            f"{results['mae']:.4f}"

        )

        c3.metric(

            "MSE",

            f"{results['mse']:.4f}"

        )

        c4.metric(

            "RMSE",

            f"{results['rmse']:.4f}"

        )

    st.markdown("---")

    # ==========================================================
    # Prediction Preview
    # ==========================================================

    st.subheader("📄 Prediction Results")

    st.dataframe(

        results["predictions"],

        use_container_width=True,

        hide_index=True

    )

    # Results are intentionally kept alive for Part 2–4
    st.session_state["ml_results"] = results
    st.session_state["ml_problem_type"] = problem_type
    st.session_state["ml_model_name"] = model_name

    from datetime import datetime

    if "model_history" not in st.session_state:

        st.session_state["model_history"] = []

    history = st.session_state["model_history"]

    score = (

        results["accuracy"]

        if problem_type == "Classification"

        else results["r2"]

    )

    new_record = {

        "Time": datetime.now().strftime("%H:%M:%S"),

        "Model": model_name,

        "Problem": problem_type,

        "Target": target,

        "Score": round(score, 4)

    }

    if not history or history[-1] != new_record:

        history.append(new_record)

    # ==========================================================
    # Visual Results
    # ==========================================================

    # Continue using the current training results

    st.markdown("---")

    st.header("📊 Model Performance")

    # ==========================================================
    # Classification
    # ==========================================================

    if problem_type == "Classification":

        st.subheader("🔥 Confusion Matrix")

        matrix = results["confusion_matrix"]

        fig = ff.create_annotated_heatmap(

            z=matrix,

            x=[f"Pred {i}" for i in range(matrix.shape[1])],

            y=[f"Actual {i}" for i in range(matrix.shape[0])],

            colorscale="Blues",

            showscale=True

        )

        fig.update_layout(

            height=500,

            template="plotly_white"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.markdown("---")

        st.subheader("🎯 Prediction Distribution")

        counts = (

            results["predictions"]["Predicted"]

            .value_counts()

            .reset_index()

        )

        counts.columns = [

            "Class",

            "Count"

        ]

        fig = px.bar(

            counts,

            x="Class",

            y="Count",

            color="Class",

            template="plotly_white"

        )

        fig.update_layout(

            height=450,

            showlegend=False

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ==========================================================
    # Regression
    # ==========================================================

    else:

        st.subheader("📈 Actual vs Predicted")

        predictions = results["predictions"].copy()

        fig = px.scatter(

            predictions,

            x="Actual",

            y="Predicted",

            template="plotly_white"

        )

        fig.update_layout(

            height=550

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.markdown("---")

        st.subheader("📉 Prediction Error")

        predictions["Error"] = (

            predictions["Actual"]

            -

            predictions["Predicted"]

        )

        fig = px.histogram(

            predictions,

            x="Error",

            nbins=30,

            template="plotly_white",

            color_discrete_sequence=["#EF4444"]

        )

        fig.update_layout(

            height=450

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.markdown("---")

    # ==========================================================
    # Downloads & Reports
    # ==========================================================

    st.header("📁 Export & Reports")

    col1, col2 = st.columns(2)

    # ==========================================================
    # Download Predictions
    # ==========================================================

    with col1:

        csv = predictions_to_csv(

            results["predictions"]

        )

        st.download_button(

            label="📥 Download Predictions (.csv)",

            data=csv,

            file_name="predictions.csv",

            mime="text/csv",

            use_container_width=True

        )

    # ==========================================================
    # Download Model
    # ==========================================================

    with col2:

        model = model_to_bytes(

            results["model_file"]

        )

        st.download_button(

            label="💾 Download Trained Model (.pkl)",

            data=model,

            file_name="trained_model.pkl",

            mime="application/octet-stream",

            use_container_width=True

        )

    # ==========================================================
    # Download Cleaned Dataset
    # ==========================================================

    if "cleaned_dataset" in st.session_state:

        cleaned_csv = (

            st.session_state["cleaned_dataset"]

            .to_csv(index=False)

            .encode("utf-8")

        )

        st.download_button(

            label="⬇️ Download Cleaned Dataset (.csv)",

            data=cleaned_csv,

            file_name="cleaned_dataset.csv",

            mime="text/csv",

            use_container_width=True

        )

    st.markdown("---")

    # ==========================================================
    # Classification Report
    # ==========================================================

    if problem_type == "Classification":

        st.subheader("📋 Performance Summary")

        report = pd.DataFrame({

            "Metric": [

                "Accuracy",

                "Precision",

                "Recall",

                "F1 Score"

            ],

            "Value": [

                round(results["accuracy"], 4),

                round(results["precision"], 4),

                round(results["recall"], 4),

                round(results["f1"], 4)

            ]

        })

        st.dataframe(

            report,

            use_container_width=True,

            hide_index=True

        )

    # ==========================================================
    # Regression Report
    # ==========================================================

    else:

        st.subheader("📋 Performance Summary")

        report = pd.DataFrame({

            "Metric": [

                "R²",

                "MAE",

                "MSE",

                "RMSE"

            ],

            "Value": [

                round(results["r2"], 4),

                round(results["mae"], 4),

                round(results["mse"], 4),

                round(results["rmse"], 4)

            ]

        })

        st.dataframe(

            report,

            use_container_width=True,

            hide_index=True

        )

    st.markdown("---")

    # ==========================================================
    # Feature Importance
    # ==========================================================

    model = results["pipeline"].named_steps["model"]

    preprocessor = results["pipeline"].named_steps["preprocessor"]

    if hasattr(model, "feature_importances_"):

        st.subheader("🌳 Feature Importance")

        try:

            feature_names = preprocessor.get_feature_names_out()

        except Exception:

            feature_names = [

                f"Feature {i+1}"

                for i in range(len(model.feature_importances_))

            ]

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": model.feature_importances_

        })

        importance = importance.sort_values(

            by="Importance",

            ascending=False

        ).head(20)

        fig = px.bar(

            importance,

            x="Importance",

            y="Feature",

            orientation="h",

            color="Importance",

            template="plotly_white"

        )

        fig.update_layout(

            height=650,

            yaxis={"categoryorder": "total ascending"}

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.markdown("---")

    # ==========================================================
    # Advanced Evaluation
    # ==========================================================

    st.header("⚙️ Advanced Evaluation")

    pipeline = results["pipeline"]

    target_column = target

    X = df.drop(columns=[target_column])

    y = df[target_column]

    try:

        if problem_type == "Classification":

            scores = cross_val_score(

                pipeline,

                X,

                y,

                cv=5,

                scoring="accuracy"

            )

        else:

            scores = cross_val_score(

                pipeline,

                X,

                y,

                cv=5,

                scoring="r2"

            )

        st.subheader("5-Fold Cross Validation")

        c1, c2, c3 = st.columns(3)

        c1.metric(

            "Average Score",

            f"{scores.mean():.4f}"

        )

        c2.metric(

            "Best Fold",

            f"{scores.max():.4f}"

        )

        c3.metric(

            "Worst Fold",

            f"{scores.min():.4f}"

        )

        cv_df = pd.DataFrame({

            "Fold": [

                "Fold 1",

                "Fold 2",

                "Fold 3",

                "Fold 4",

                "Fold 5"

            ],

            "Score": scores

        })

        fig = px.line(

            cv_df,

            x="Fold",

            y="Score",

            markers=True,

            template="plotly_white"

        )

        fig.update_layout(

            height=450,

            title="Cross Validation Performance",

            title_x=0.5

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    except Exception as e:

        st.info(

            f"Cross-validation could not be performed: {e}"

        )

    st.markdown("---")

    # ==========================================================
    # AI Model Assessment
    # ==========================================================

    st.header("🤖 AI Model Assessment")

    if problem_type == "Classification":

        score = results["accuracy"]

    else:

        score = results["r2"]

    if score >= 0.95:

        st.success(

            "🏆 Excellent model performance. The model generalizes very well."

        )

    elif score >= 0.85:

        st.success(

            "✅ Very good model performance."

        )

    elif score >= 0.70:

        st.warning(

            "⚠️ Acceptable performance. Feature engineering may improve results."

        )

    elif score >= 0.50:

        st.warning(

            "⚠️ Moderate performance. Try more preprocessing or different models."

        )

    else:

        st.error(

            "❌ Poor performance. Consider cleaning the dataset, engineering features, or selecting another algorithm."

        )

    st.markdown("---")

    # ==========================================================
    # Training Summary
    # ==========================================================

    st.header("📑 Training Summary")

    summary = pd.DataFrame({

        "Property": [

            "Dataset Rows",

            "Dataset Columns",

            "Target",

            "Problem Type",

            "Model",

            "Test Size"

        ],

        "Value": [

            len(df),

            len(df.columns),

            target,

            problem_type,

            model_name,

            f"{int(test_size*100)}%"

        ]

    })

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )

    st.markdown("---")

    st.header("📚 Model Training History")

    history = st.session_state.get(

        "model_history",

        []

    )

    if len(history) > 0:

        history_df = pd.DataFrame(history)

        st.dataframe(

            history_df,

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info(

            "No models have been trained yet."

        )

    st.markdown("---")

    st.header("💼 Business Insights")

    insights = generate_business_insights(

        df,

        results,

        problem_type,

        target

    )

    for insight in insights:

        st.success(insight)

    st.success("🎉 Machine Learning pipeline completed successfully.")