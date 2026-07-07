def missing_values(df):

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    return missing.sort_values(ascending=False)
def duplicate_rows(df):

    return df[df.duplicated()]