import pandas as pd


# ==========================================================
# Dataset Summary
# ==========================================================

def dataset_summary(df):

    rows, columns = df.shape

    numerical = len(
        df.select_dtypes(include="number").columns
    )

    categorical = len(
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    memory = round(
        df.memory_usage(deep=True).sum() / 1024 / 1024,
        2
    )

    return {

        "Rows": rows,

        "Columns": columns,

        "Numerical Columns": numerical,

        "Categorical Columns": categorical,

        "Memory Usage": memory

    }


# ==========================================================
# Missing Values
# ==========================================================

def analyze_missing_values(df):

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if missing.empty:

        return {

            "status": True,

            "message": "No missing values found."

        }

    return {

        "status": False,

        "data": missing.sort_values(
            ascending=False
        )

    }


# ==========================================================
# Duplicate Rows
# ==========================================================

def analyze_duplicates(df):

    duplicates = int(df.duplicated().sum())

    return duplicates


# ==========================================================
# Outliers
# ==========================================================

def detect_outliers(df):

    numerical = df.select_dtypes(
        include="number"
    )

    results = {}

    for column in numerical.columns:

        q1 = numerical[column].quantile(0.25)

        q3 = numerical[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        outliers = numerical[
            (numerical[column] < lower) |
            (numerical[column] > upper)
        ]

        results[column] = len(outliers)

    return results


# ==========================================================
# Feature Types
# ==========================================================

def feature_information(df):

    return {

        "Numerical":

        df.select_dtypes(

            include="number"

        ).columns.tolist(),

        "Categorical":

        df.select_dtypes(

            include=["object", "category"]

        ).columns.tolist()

    }


# ==========================================================
# AI Recommendations
# ==========================================================

def preprocessing_recommendations(df):

    recommendations = []

    if df.isnull().sum().sum() > 0:

        recommendations.append(

            "Fill missing values."

        )

    if df.duplicated().sum() > 0:

        recommendations.append(

            "Remove duplicate rows."

        )

    numerical = df.select_dtypes(
        include="number"
    )

    if len(numerical.columns) > 0:

        recommendations.append(

            "Scale numerical features."

        )

    categorical = df.select_dtypes(

        include=["object", "category"]

    )

    if len(categorical.columns) > 0:

        recommendations.append(

            "Encode categorical variables."

        )

    recommendations.append(

        "Split data into training and testing sets."

    )

    return recommendations


# ==========================================================
# ML Recommendation
# ==========================================================

def recommend_ml_problem(df, target):

    if pd.api.types.is_numeric_dtype(

        df[target]

    ):

        return {

            "Problem": "Regression",

            "Models": [

                "Linear Regression",

                "Random Forest Regressor",

                "XGBoost",

                "LightGBM"

            ]

        }

    return {

        "Problem": "Classification",

        "Models": [

            "Logistic Regression",

            "Random Forest",

            "XGBoost",

            "CatBoost"

        ]

    }