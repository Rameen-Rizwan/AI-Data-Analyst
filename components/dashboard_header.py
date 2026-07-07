import streamlit as st


def show_dashboard_header(title, subtitle):

    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg,#2563EB,#1D4ED8);
        padding:25px;
        border-radius:15px;
        color:white;
        margin-bottom:20px;
    ">
        <h1 style="margin:0;">{title}</h1>
        <p style="margin-top:10px;font-size:18px;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)