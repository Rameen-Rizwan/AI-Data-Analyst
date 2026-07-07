import streamlit as st


def show_data_quality(missing, duplicates):

    with st.expander("🧹 Data Quality"):

        st.subheader("❗ Missing Values")

        if len(missing) == 0:
            st.success("✅ No missing values found.")
        else:
            st.dataframe(
                missing,
                use_container_width=True
            )

        st.divider()

        st.subheader("📑 Duplicate Rows")

        if duplicates.empty:
            st.success("✅ No duplicate rows found.")
        else:
            st.dataframe(
                duplicates,
                use_container_width=True
            )