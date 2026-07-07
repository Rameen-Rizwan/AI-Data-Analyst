def get_chart_recommendations(df):
    """
    Analyze the dataset and recommend suitable visualizations.
    """

    recommendations = []

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Histograms for numerical columns
    for column in numeric_columns:
        recommendations.append({
            "chart": "Histogram",
            "description": f"View the distribution of '{column}'."
        })

    # Bar charts for categorical columns
    for column in categorical_columns:
        recommendations.append({
            "chart": "Bar Chart",
            "description": f"Compare the frequency of '{column}'."
        })

    # Pie charts for categorical columns
    for column in categorical_columns:
        recommendations.append({
            "chart": "Pie Chart",
            "description": f"View the proportion of '{column}'."
        })

    # Scatter plot recommendation
    if len(numeric_columns) >= 2:
        recommendations.append({
            "chart": "Scatter Plot",
            "description": f"Explore the relationship between '{numeric_columns[0]}' and '{numeric_columns[1]}'."
        })

    # Heatmap recommendation
    if len(numeric_columns) >= 2:
        recommendations.append({
            "chart": "Correlation Heatmap",
            "description": "Analyze relationships among numerical features."
        })

    return recommendations