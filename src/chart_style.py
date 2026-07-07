"""
Professional Chart Styling
--------------------------
Provides a consistent theme and appearance for all Plotly charts.
"""

import plotly.express as px


# ==========================================================
# Professional Color Themes
# ==========================================================

COLOR_THEMES = {

    "Professional": px.colors.qualitative.Safe,

    "Modern": px.colors.qualitative.Set2,

    "Pastel": px.colors.qualitative.Pastel,

    "Bold": px.colors.qualitative.Bold,

    "Vivid": px.colors.qualitative.Vivid,

    "Dark": px.colors.qualitative.Dark24

}


# ==========================================================
# Apply Theme
# ==========================================================

def apply_chart_theme(fig):

    fig.update_layout(

        template="plotly_white",

        height=650,

        title=dict(

            x=0.5,

            xanchor="center",

            font=dict(

                family="Segoe UI",

                size=24,

                color="#1F2937"

            )

        ),

        font=dict(

            family="Segoe UI",

            size=13,

            color="#374151"

        ),

        paper_bgcolor="white",

        plot_bgcolor="white",

        margin=dict(

            l=30,

            r=30,

            t=80,

            b=30

        ),

        hoverlabel=dict(

            bgcolor="white",

            bordercolor="#D1D5DB",

            font=dict(

                family="Segoe UI",

                size=13,

                color="black"

            )

        ),

        legend=dict(

            orientation="h",

            yanchor="bottom",

            y=1.02,

            xanchor="right",

            x=1

        )

    )

    fig.update_xaxes(

        showgrid=False,

        showline=True,

        linewidth=1,

        linecolor="#D1D5DB",

        mirror=True,

        ticks="outside"

    )

    fig.update_yaxes(

        showgrid=True,

        gridcolor="#ECECEC",

        zeroline=False,

        showline=True,

        linewidth=1,

        linecolor="#D1D5DB",

        mirror=True,

        ticks="outside"

    )

    return fig