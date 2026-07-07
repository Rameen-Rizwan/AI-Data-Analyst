import io
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ==========================================================
# Dataset Report
# ==========================================================

def dataset_report(df):

    rows, columns = df.shape

    numeric = len(
        df.select_dtypes(include="number").columns
    )

    categorical = len(
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    table = pd.DataFrame({

        "Property": [

            "Rows",

            "Columns",

            "Numeric Columns",

            "Categorical Columns",

            "Memory Usage (MB)"

        ],

        "Value": [

            rows,

            columns,

            numeric,

            categorical,

            round(
                df.memory_usage(deep=True).sum()
                / 1024 / 1024,
                2
            )

        ]

    })

    return {

        "Rows": rows,

        "Columns": columns,

        "Numeric": numeric,

        "Categorical": categorical,

        "Table": table

    }


# ==========================================================
# Data Quality
# ==========================================================

def data_quality_report(df):

    report = pd.DataFrame({

        "Metric": [

            "Missing Cells",

            "Duplicate Rows",

            "Unique Columns",

            "Total Cells"

        ],

        "Value": [

            int(df.isnull().sum().sum()),

            int(df.duplicated().sum()),

            len(df.columns),

            df.shape[0] * df.shape[1]

        ]

    })

    return report


# ==========================================================
# Executive Summary
# ==========================================================

def executive_summary(df):

    rows, cols = df.shape

    missing = int(df.isnull().sum().sum())

    duplicates = int(df.duplicated().sum())

    summary = f"""
Dataset contains {rows:,} rows and {cols} columns.

Missing values detected: {missing}

Duplicate rows detected: {duplicates}

The dataset is suitable for exploratory data analysis and
machine learning after basic preprocessing.
"""

    return summary
# ==========================================================
# PDF Generator
# ==========================================================

def generate_pdf(df):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(

        buffer,

        pagesize=(8.27 * inch, 11.69 * inch)

    )

    styles = getSampleStyleSheet()

    elements = []

    # ======================================================
    # Title
    # ======================================================

    title = Paragraph(

        "<b><font size=20>AI Data Analysis Report</font></b>",

        styles["Title"]

    )

    elements.append(title)

    elements.append(Spacer(1, 0.3 * inch))

    # ======================================================
    # Executive Summary
    # ======================================================

    elements.append(

        Paragraph(

            "<b>Executive Summary</b>",

            styles["Heading1"]

        )

    )

    elements.append(

        Paragraph(

            executive_summary(df),

            styles["BodyText"]

        )

    )

    elements.append(

        Spacer(1, 0.25 * inch)

    )

    # ======================================================
    # Dataset Overview
    # ======================================================

    elements.append(

        Paragraph(

            "<b>Dataset Overview</b>",

            styles["Heading1"]

        )

    )

    overview = dataset_report(df)["Table"]

    overview_table = [

        overview.columns.tolist()

    ] + overview.values.tolist()

    table = Table(

        overview_table,

        colWidths=[3 * inch, 2.5 * inch]

    )

    table.setStyle(

        TableStyle(

            [

                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),

                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                ("GRID", (0, 0), (-1, -1), 1, colors.grey),

                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),

                ("BACKGROUND", (0, 1), (-1, -1), colors.beige)

            ]

        )

    )

    elements.append(table)

    elements.append(

        Spacer(1, 0.3 * inch)

    )

    # ======================================================
    # Data Quality
    # ======================================================

    elements.append(

        Paragraph(

            "<b>Data Quality Report</b>",

            styles["Heading1"]

        )

    )

    quality = data_quality_report(df)

    quality_table = [

        quality.columns.tolist()

    ] + quality.values.tolist()

    table = Table(

        quality_table,

        colWidths=[3 * inch, 2.5 * inch]

    )

    table.setStyle(

        TableStyle(

            [

                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16A34A")),

                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                ("GRID", (0, 0), (-1, -1), 1, colors.grey),

                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),

                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke)

            ]

        )

    )

    elements.append(table)

    elements.append(

        Spacer(1, 0.3 * inch)

    )
    # ==========================================================
# Descriptive Statistics
# ==========================================================

    elements.append(

        Paragraph(

            "<b>Descriptive Statistics</b>",

            styles["Heading1"]

        )

    )

    stats = df.describe(include="all").fillna("-")

    stats_table = [

        ["Column"] + list(stats.columns)

    ]

    for index in stats.index:

        row = [str(index)] + [

            str(value)

            for value in stats.loc[index]

        ]

        stats_table.append(row)

    table = Table(stats_table)

    table.setStyle(

        TableStyle(

            [

                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F59E0B")),

                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                ("FONTSIZE", (0, 0), (-1, -1), 7),

                ("BOTTOMPADDING", (0, 0), (-1, 0), 8)

            ]

        )

    )

    elements.append(table)

    elements.append(

        Spacer(1, 0.3 * inch)

    )

    # ======================================================
    # AI Recommendations
    # ======================================================

    elements.append(

        Paragraph(

            "<b>AI Recommendations</b>",

            styles["Heading1"]

        )

    )

    recommendations = []

    if df.isnull().sum().sum() > 0:

        recommendations.append(
            "• Handle missing values before training machine learning models."
        )

    if df.duplicated().sum() > 0:

        recommendations.append(
            "• Remove duplicate records."
        )

    if len(df.select_dtypes(include="number").columns) > 0:

        recommendations.append(
            "• Scale numerical features for better model performance."
        )

    if len(
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    ) > 0:

        recommendations.append(
            "• Encode categorical variables before model training."
        )

    recommendations.append(
        "• Perform feature selection to improve efficiency."
    )

    recommendations.append(
        "• Evaluate multiple machine learning algorithms."
    )

    recommendations.append(
        "• Use cross-validation before finalizing the model."
    )

    for recommendation in recommendations:

        elements.append(

            Paragraph(

                recommendation,

                styles["BodyText"]

            )

        )

    elements.append(

        Spacer(1, 0.3 * inch)

    )

    # ======================================================
    # Footer
    # ======================================================

    elements.append(

        Paragraph(

            "<b>Report Generated by AI Data Analyst</b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            "Developed using Python, Streamlit, Scikit-learn and ReportLab.",

            styles["BodyText"]

        )

    )

    elements.append(

        Paragraph(

            "This report is automatically generated based on the uploaded dataset.",

            styles["Italic"]

        )

    )

    # ======================================================
    # Build PDF
    # ======================================================

    document.build(elements)

    buffer.seek(0)

    return buffer