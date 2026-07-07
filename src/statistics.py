def dataset_summary(df):

    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum(),
        "Memory (MB)": round(df.memory_usage(deep=True).sum()/1024**2,2)
    }

    return summary
def numerical_columns(df):

    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df):

    return df.select_dtypes(exclude="number").columns.tolist()