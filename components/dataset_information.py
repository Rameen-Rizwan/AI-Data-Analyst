import streamlit as st


def show_dataset_information(df, numerical, categorical):

    with st.expander("📑 Dataset Information"):

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🔢 Numerical Columns")

            if numerical:
                st.write(numerical)
            else:
                st.info("No numerical columns found.")

        with col2:

            st.subheader("🔤 Categorical Columns")

            if categorical:
                st.write(categorical)
            else:
                st.info("No categorical columns found.")

        st.divider()

        st.subheader("🏷️ Data Types")

        data_types = df.dtypes.astype(str).reset_index()
        data_types.columns = ["Column", "Data Type"]

        st.dataframe(
            data_types,
            use_container_width=True
        )