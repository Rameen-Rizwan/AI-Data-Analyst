import streamlit as st


def show_download_dataset(df):

    st.subheader("⬇️ Export Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download CSV",

        data=csv,

        file_name="processed_dataset.csv",

        mime="text/csv",

        use_container_width=True

    )