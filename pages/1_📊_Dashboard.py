import streamlit as st

from src.data_loader import load_data
from src.session_manager import (
    set_dataset,
    get_dataset,
    has_dataset,
    set_filename,
    get_filename
)

from src.statistics import (
    dataset_summary,
    numerical_columns,
    categorical_columns
)

from src.preprocessing import (
    missing_values,
    duplicate_rows
)

from components.dashboard import show_dashboard
from components.dataset_preview import show_dataset_preview
from components.dataset_information import show_dataset_information
from components.data_quality import show_data_quality
from components.descriptive_statistics import show_descriptive_statistics
from components.filter_panel import show_filter_panel
from components.dataset_health import show_dataset_health
from components.column_profile import show_column_profile
from components.correlation_analysis import show_correlation_analysis
from components.overview_charts import show_overview_charts
from components.data_quality_report import show_data_quality_report
from components.data_dictionary import show_data_dictionary
from components.download_dataset import show_download_dataset

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Dashboard")

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose CSV or Excel File",
    type=["csv", "xlsx"]
)

# ==========================================================
# Load Dataset
# ==========================================================

if uploaded_file is not None:

    df = load_data(uploaded_file)

    set_dataset(df)

    set_filename(uploaded_file.name)

# ==========================================================
# Check Dataset
# ==========================================================

if not has_dataset():

    st.info("👈 Please upload a dataset from the sidebar.")

    st.stop()

df = get_dataset()
with st.sidebar:

    st.markdown("---")

    filtered_df = show_filter_panel(df)

df = filtered_df

# ==========================================================
# Dataset Information
# ==========================================================

summary = dataset_summary(df)

numerical = numerical_columns(df)

categorical = categorical_columns(df)

missing = missing_values(df)

duplicates = duplicate_rows(df)

# ==========================================================
# Header
# ==========================================================

st.success(f"✅ Loaded Dataset : {get_filename()}")
from components.metric_cards import show_metric_cards

show_metric_cards(df)

st.markdown("---")
show_dataset_health(df)

st.markdown("---")

show_column_profile(df)

st.markdown("---")

show_correlation_analysis(df)

st.markdown("---")
show_overview_charts(df)

st.markdown("---")
show_data_quality_report(df)

st.markdown("---")
show_data_dictionary(df)

st.markdown("---")
show_download_dataset(df)

# ==========================================================
# Components
# ==========================================================

show_dashboard(summary)

show_dataset_preview(df)

show_dataset_information(
    df,
    numerical,
    categorical
)

show_data_quality(
    missing,
    duplicates
)

show_descriptive_statistics(df)