import streamlit as st


# ==========================================================
# Dataset
# ==========================================================

def set_dataset(df):
    st.session_state["dataset"] = df


def get_dataset():
    return st.session_state.get("dataset", None)


def has_dataset():
    return "dataset" in st.session_state


def clear_dataset():
    if "dataset" in st.session_state:
        del st.session_state["dataset"]


# ==========================================================
# File Name
# ==========================================================

def set_filename(filename):
    st.session_state["filename"] = filename


def get_filename():
    return st.session_state.get("filename", "No Dataset")


# ==========================================================
# Dataset Information
# ==========================================================

def dataset_shape():

    df = get_dataset()

    if df is None:
        return (0, 0)

    return df.shape


def numerical_columns():

    df = get_dataset()

    if df is None:
        return []

    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns():

    df = get_dataset()

    if df is None:
        return []

    return df.select_dtypes(exclude="number").columns.tolist()