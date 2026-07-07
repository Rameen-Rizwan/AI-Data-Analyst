import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Home Page
# ==========================================================

st.title("🤖 AI Data Analyst")

st.markdown("""
Welcome to the **AI Data Analyst Platform**.

This application allows you to:

- 📊 Explore datasets
- 📈 Create professional visualizations
- 🤖 Generate AI-powered insights
- 🧠 Train Machine Learning models
- 📄 Export reports

---

### 🚀 Navigation

Use the **left sidebar** to navigate between pages.
""")