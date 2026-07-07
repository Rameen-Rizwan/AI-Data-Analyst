"""
Visualization Module
--------------------
Contains all chart generation functions for the AI Data Analyst project.
"""

import plotly.express as px

from src.chart_style import (
    COLOR_THEMES,
    apply_chart_theme
)


# ==========================================================
# Bar Chart
# ==========================================================

def create_bar_chart(df, x, y, theme):

    fig = px.bar(

        df,

        x=x,

        y=y,

        color=x,

        color_discrete_sequence=COLOR_THEMES[theme],

        title=f"📊 {y} by {x}"

    )

    fig.update_traces(

        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"

    )

    return apply_chart_theme(fig)


# ==========================================================
# Line Chart
# ==========================================================

def create_line_chart(df, x, y, theme):

    fig = px.line(

        df,

        x=x,

        y=y,

        markers=True,

        title=f"📈 {y} over {x}"

    )

    fig.update_traces(

        line=dict(width=4),

        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"

    )

    return apply_chart_theme(fig)


# ==========================================================
# Histogram
# ==========================================================

def create_histogram(df, column, theme):

    fig = px.histogram(

        df,

        x=column,

        color_discrete_sequence=COLOR_THEMES[theme],

        title=f"📉 Distribution of {column}"

    )

    fig.update_traces(

        hovertemplate="<b>%{x}</b><br>Count : %{y}<extra></extra>"

    )

    return apply_chart_theme(fig)


# ==========================================================
# Scatter Plot
# ==========================================================

def create_scatter_plot(df, x, y, theme):

    fig = px.scatter(

        df,

        x=x,

        y=y,

        color=y,

        color_continuous_scale="Viridis",

        title=f"📍 {x} vs {y}"

    )

    fig.update_traces(

        marker=dict(
            size=10,
            line=dict(
                width=1,
                color="white"
            )
        ),

        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"

    )

    return apply_chart_theme(fig)


# ==========================================================
# Pie Chart
# ==========================================================

def create_pie_chart(df, column, theme):

    counts = df[column].value_counts().reset_index()

    counts.columns = [column, "Count"]

    fig = px.pie(

        counts,

        names=column,

        values="Count",

        hole=0.45,

        color_discrete_sequence=COLOR_THEMES[theme],

        title=f"🥧 Distribution of {column}"

    )

    fig.update_traces(

        textposition="inside",

        textinfo="percent+label"

    )

    return apply_chart_theme(fig)


# ==========================================================
# Box Plot
# ==========================================================

def create_box_plot(df, column, theme):

    fig = px.box(

        df,

        y=column,

        color_discrete_sequence=COLOR_THEMES[theme],

        title=f"📦 Box Plot of {column}"

    )

    return apply_chart_theme(fig)


# ==========================================================
# Violin Plot
# ==========================================================

def create_violin_plot(df, column, theme):

    fig = px.violin(

        df,

        y=column,

        box=True,

        points="all",

        color_discrete_sequence=COLOR_THEMES[theme],

        title=f"🎻 Violin Plot of {column}"

    )

    return apply_chart_theme(fig)


# ==========================================================
# Correlation Heatmap
# ==========================================================

def create_correlation_heatmap(df):

    corr = df.select_dtypes(include="number").corr()

    fig = px.imshow(

        corr,

        text_auto=".2f",

        aspect="auto",

        color_continuous_scale="RdBu_r",

        title="🔥 Correlation Heatmap"

    )

    return apply_chart_theme(fig)