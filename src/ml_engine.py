import io
import time
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import (

    train_test_split,

    GridSearchCV

)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# Data Preparation
# ==========================================================

def prepare_data(df, target):

    data = df.copy()

    X = data.drop(columns=[target])
    y = data[target]

    # Remove rows where target is missing
    valid_rows = ~y.isna()

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    if len(X) < 10:
        raise ValueError(
            "Dataset is too small after removing missing target values."
        )

    numerical_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns)
        ],
        remainder="drop"
    )

    return X, y, preprocessor


# ==========================================================
# Classification Models
# ==========================================================

def get_classification_model(

    model_name,

    hyperparameters=None

):

    hyperparameters = hyperparameters or {}

    if model_name == "Logistic Regression":

        return LogisticRegression(

            C=hyperparameters.get("C", 1.0),

            max_iter=1000,

            random_state=42

        )

    elif model_name == "Decision Tree":

        return DecisionTreeClassifier(

            max_depth=hyperparameters.get("max_depth"),

            min_samples_split=hyperparameters.get("min_samples_split", 2),

            random_state=42

        )

    elif model_name == "Random Forest":

        return RandomForestClassifier(

            n_estimators=hyperparameters.get("n_estimators", 200),

            max_depth=hyperparameters.get("max_depth"),

            random_state=42,

            n_jobs=-1

        )

    else:

        raise ValueError(

            "Unsupported Classification Model"

        )


# ==========================================================
# Regression Models
# ==========================================================

def get_regression_model(

    model_name,

    hyperparameters=None

):

    hyperparameters = hyperparameters or {}

    if model_name == "Linear Regression":

        return LinearRegression(

            fit_intercept=hyperparameters.get(

                "fit_intercept",

                True

            )

        )

    elif model_name == "Decision Tree":

        return DecisionTreeRegressor(

            max_depth=hyperparameters.get("max_depth"),

            min_samples_split=hyperparameters.get("min_samples_split", 2),

            random_state=42

        )

    elif model_name == "Random Forest":

        return RandomForestRegressor(

            n_estimators=hyperparameters.get("n_estimators", 200),

            max_depth=hyperparameters.get("max_depth"),

            random_state=42,

            n_jobs=-1

        )

    else:

        raise ValueError(

            "Unsupported Regression Model"

        )


# ==========================================================
# Classification Training
# ==========================================================

def train_classification(df, target, model_name, test_size, hyperparameters=None):

    if hyperparameters is None:
        hyperparameters = {}

    X, y, preprocessor = prepare_data(df, target)

    # Handle missing target values
    y = y.fillna("Missing").astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Ensure each class has enough samples
    class_counts = pd.Series(y).value_counts()
    valid_classes = class_counts[class_counts >= 2].index

    mask = pd.Series(y).isin(valid_classes)

    X = X.loc[mask].reset_index(drop=True)
    y = pd.Series(y)[mask].reset_index(drop=True).values

    if len(np.unique(y)) < 2:
        raise ValueError(
            "At least two classes with two or more samples are required."
        )

    # Use stratify only when every class has at least 2 samples
    unique, counts = np.unique(y, return_counts=True)

    if np.min(counts) >= 2:
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify
    )

    model = get_classification_model(

        model_name,

        hyperparameters

    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    if hyperparameters.get("auto_tune", False):

        if model_name == "Random Forest":

            param_grid = {

                "model__n_estimators": [100, 200, 300],

                "model__max_depth": [5, 10, 20, None]

            }

        elif model_name == "Decision Tree":

            param_grid = {

                "model__max_depth": [3, 5, 10, 20],

                "model__min_samples_split": [2, 5, 10]

            }

        elif model_name == "Logistic Regression":

            param_grid = {

                "model__C": [0.01, 0.1, 1, 10]

            }

        else:

            param_grid = {}

        if param_grid:

            search = GridSearchCV(

                pipeline,

                param_grid,

                cv=5,

                scoring="accuracy",

                n_jobs=-1

            )

            search.fit(

                X_train,

                y_train

            )

            pipeline = search.best_estimator_

        else:

            pipeline.fit(

                X_train,

                y_train

            )

    else:

        pipeline.fit(

            X_train,

            y_train

        )

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    matrix = confusion_matrix(y_test, predictions)

    prediction_df = pd.DataFrame(
        {
            "Actual": label_encoder.inverse_transform(y_test),
            "Predicted": label_encoder.inverse_transform(predictions)
        }
    )

    model_bytes = io.BytesIO()
    pickle.dump(pipeline, model_bytes)
    model_bytes.seek(0)

    return {
        "pipeline": pipeline,
        "score": accuracy,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
        "predictions": prediction_df,
        "model_file": model_bytes
    }


# ==========================================================
# Regression Training
# ==========================================================

def train_regression(df, target, model_name, test_size, hyperparameters=None):

    if hyperparameters is None:
        hyperparameters = {}

    X, y, preprocessor = prepare_data(df, target)

    # Target validation
    if y.isnull().all():
        raise ValueError(
            "Target column contains only missing values."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Regression requires at least two unique target values."
        )

    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError(
            "Regression target must contain numeric values."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    model = get_regression_model(

        model_name,

        hyperparameters

    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    if hyperparameters.get("auto_tune", False):

        if model_name == "Random Forest":

            param_grid = {

                "model__n_estimators": [100, 200, 300],

                "model__max_depth": [5, 10, 20, None]

            }

        elif model_name == "Decision Tree":

            param_grid = {

                "model__max_depth": [3, 5, 10, 20],

                "model__min_samples_split": [2, 5, 10]

            }

        else:

            param_grid = {}

        if param_grid:

            search = GridSearchCV(

                pipeline,

                param_grid,

                cv=5,

                scoring="r2",

                n_jobs=-1

            )

            search.fit(

                X_train,

                y_train

            )

            pipeline = search.best_estimator_

        else:

            pipeline.fit(

                X_train,

                y_train

            )

    else:

        pipeline.fit(

            X_train,

            y_train

        )

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    prediction_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": predictions
        }
    )

    model_bytes = io.BytesIO()
    pickle.dump(pipeline, model_bytes)
    model_bytes.seek(0)

    return {
        "pipeline": pipeline,
        "score": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "predictions": prediction_df,
        "model_file": model_bytes
    }


# ==========================================================
# Automatic Target Detection
# ==========================================================

def suggest_target_column(df):

    candidates = []

    for column in df.columns:

        unique = df[column].nunique(dropna=True)

        dtype = df[column].dtype

        # Binary categorical columns
        if unique == 2:
            candidates.append(
                (
                    column,
                    "Binary classification target"
                )
            )

        # Small categorical columns
        elif (
            df[column].dtype == "object"
            and
            3 <= unique <= 15
        ):
            candidates.append(
                (
                    column,
                    "Multi-class classification target"
                )
            )

        # Numeric regression candidates
        elif (
            pd.api.types.is_numeric_dtype(dtype)
            and
            unique > 20
        ):
            candidates.append(
                (
                    column,
                    "Regression target"
                )
            )

    if len(candidates) == 0:

        return None, "No suitable target detected."

    return candidates[0]


# ==========================================================
# Dataset Type Detection
# ==========================================================

def detect_dataset_type(df):

    columns = [column.lower() for column in df.columns]

    keywords = {

        "Customer Churn": [
            "churn",
            "customer",
            "tenure",
            "contract"
        ],

        "Sales": [
            "sales",
            "revenue",
            "profit",
            "quantity",
            "order"
        ],

        "Healthcare": [
            "patient",
            "diagnosis",
            "disease",
            "hospital",
            "age"
        ],

        "Finance": [
            "loan",
            "credit",
            "balance",
            "income",
            "payment"
        ],

        "Education": [
            "student",
            "grade",
            "marks",
            "attendance",
            "course"
        ],

        "HR": [
            "employee",
            "salary",
            "department",
            "attrition"
        ],

        "Retail": [
            "product",
            "category",
            "price",
            "store"
        ]

    }

    scores = {}

    for dataset_type, words in keywords.items():

        score = 0

        for word in words:

            if any(word in column for column in columns):

                score += 1

        scores[dataset_type] = score

    best_match = max(scores, key=scores.get)

    if scores[best_match] == 0:

        return "Generic Dataset"

    return f"{best_match} Dataset"


# ==========================================================
# One-Click Data Cleaning
# ==========================================================

def clean_dataset(df):

    cleaned = df.copy()

    # ------------------------------------------
    # Remove duplicate rows
    # ------------------------------------------

    cleaned = cleaned.drop_duplicates()

    # ------------------------------------------
    # Remove constant columns
    # ------------------------------------------

    constant_columns = [

        column

        for column in cleaned.columns

        if cleaned[column].nunique(dropna=False) <= 1

    ]

    cleaned = cleaned.drop(

        columns=constant_columns,

        errors="ignore"

    )

    # ------------------------------------------
    # Fill missing values
    # ------------------------------------------

    for column in cleaned.columns:

        if pd.api.types.is_numeric_dtype(cleaned[column]):

            cleaned[column] = cleaned[column].fillna(

                cleaned[column].median()

            )

        else:

            cleaned[column] = cleaned[column].fillna(

                cleaned[column].mode()[0]

            )

    # ------------------------------------------
    # Remove extra spaces
    # ------------------------------------------

    for column in cleaned.select_dtypes(

        include="object"

    ):

        cleaned[column] = cleaned[column].astype(str).str.strip()

    return cleaned


# ==========================================================
# Main Training Function
# ==========================================================

def train_model(df, target, model_name, problem_type, test_size, hyperparameters=None):

    if target not in df.columns:
        raise ValueError(
            "Selected target column does not exist."
        )

    if df.shape[0] < 20:
        raise ValueError(
            "Dataset must contain at least 20 rows for training."
        )

    if df.shape[1] < 2:
        raise ValueError(
            "Dataset must contain at least one feature and one target column."
        )

    try:
        if problem_type == "Classification":
            return train_classification(df, target, model_name, test_size, hyperparameters)

        elif problem_type == "Regression":
            return train_regression(df, target, model_name, test_size, hyperparameters)

        else:
            raise ValueError(
                "Unsupported Problem Type."
            )

    except Exception as e:
        raise ValueError(
            f"Training failed: {str(e)}"
        )


# ==========================================================
# Download Predictions
# ==========================================================

def predictions_to_csv(predictions):
    return predictions.to_csv(index=False).encode("utf-8")


# ==========================================================
# Download Trained Model
# ==========================================================

def model_to_bytes(model_bytes):
    model_bytes.seek(0)
    return model_bytes


# ==========================================================
# Business Insights
# ==========================================================

def generate_business_insights(

    df,

    results,

    problem_type,

    target

):

    insights = []

    # Dataset size

    insights.append(

        f"Dataset contains {len(df)} rows and {len(df.columns)} columns."

    )

    # Missing values

    missing = df.isna().sum().sum()

    if missing == 0:

        insights.append(

            "No missing values were detected."

        )

    else:

        insights.append(

            f"The dataset contains {missing} missing values."

        )

    # Duplicate rows

    duplicates = df.duplicated().sum()

    if duplicates == 0:

        insights.append(

            "No duplicate rows were found."

        )

    else:

        insights.append(

            f"{duplicates} duplicate rows were detected."

        )

    # Target analysis

    insights.append(

        f"Target column selected: '{target}'."

    )

    # Model performance

    if problem_type == "Classification":

        score = results["accuracy"]

        insights.append(

            f"The trained model achieved an accuracy of {score:.2%}."

        )

    else:

        score = results["r2"]

        insights.append(

            f"The trained model achieved an R² score of {score:.2%}."

        )

    # Performance interpretation

    if score >= 0.90:

        insights.append(

            "The model demonstrates excellent predictive performance."

        )

    elif score >= 0.75:

        insights.append(

            "The model shows good predictive performance."

        )

    elif score >= 0.60:

        insights.append(

            "The model performance is acceptable but can be improved."

        )

    else:

        insights.append(

            "The model performance is relatively low. Consider improving data quality or feature engineering."

        )

    return insights


# ==========================================================
# Model Information
# ==========================================================

def model_information(model_name, problem_type):

    information = {
        "Problem Type": problem_type,
        "Model": model_name
    }

    if model_name == "Logistic Regression":
        information["Description"] = (
            "A linear classification algorithm suitable "
            "for binary and multiclass classification."
        )

    elif model_name == "Linear Regression":
        information["Description"] = (
            "A linear regression algorithm for predicting "
            "continuous values."
        )

    elif model_name == "Decision Tree":
        information["Description"] = (
            "A tree-based algorithm capable of capturing "
            "non-linear relationships."
        )

    elif model_name == "Random Forest":
        information["Description"] = (
            "An ensemble learning algorithm using multiple "
            "decision trees for improved performance."
        )

    else:
        information["Description"] = "Machine learning model."

    return information
def compare_models(

    df,

    target,

    problem_type,

    test_size

):

    if problem_type == "Classification":

        models = [

            "Logistic Regression",

            "Decision Tree",

            "Random Forest"

        ]

    else:

        models = [

            "Linear Regression",

            "Decision Tree",

            "Random Forest"

        ]

    results = []

    for model_name in models:

        try:
            start_time = time.perf_counter()
            output = train_model(

                df,

                target,

                model_name,

                problem_type,

                test_size,

                hyperparameters={}

            )
            training_time = round(

                time.perf_counter() - start_time,

                3

            )

            if problem_type == "Classification":

                score = output["accuracy"]

            else:

                score = output["r2"]

            results.append({

                "Model": model_name,

                "Score": round(score, 4),

                "Training Time (s)": training_time,

                "Status": "✅ Success"

            })

        except Exception as e:

            results.append({

                "Model": model_name,

                "Score": None,
                
                "Training Time (s)": None,

                "Status": "❌ Failed",

                "Error": str(e)

            })

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(

    by="Score",

    ascending=False,

    na_position="last"

    ).reset_index(drop=True)

    comparison.insert(

    0,

    "Rank",

    range(1, len(comparison) + 1)

    )

    return comparison