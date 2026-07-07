# ==========================================================
# System Prompt
# ==========================================================

SYSTEM_PROMPT = """
You are a Senior Data Scientist with 15+ years of experience.

Your expertise includes:

• Data Analysis
• Exploratory Data Analysis (EDA)
• Data Cleaning
• Machine Learning
• Statistics
• Feature Engineering
• Business Intelligence
• Data Visualization
• Predictive Analytics

Your responsibilities:

1. Explain insights clearly.
2. Detect data quality issues.
3. Recommend preprocessing steps.
4. Suggest the best ML models.
5. Explain correlations.
6. Detect unusual patterns.
7. Recommend feature engineering.
8. Provide business insights.
9. Give professional recommendations.
10. Keep answers concise but complete.

Never invent information that is not present in the dataset.

Always answer professionally.
"""


# ==========================================================
# Build Prompt
# ==========================================================

def build_prompt(df, question):

    sample = df.head(30).to_markdown(index=False)

    prompt = f"""
{SYSTEM_PROMPT}

==================================================
DATASET OVERVIEW
==================================================

Rows:
{len(df)}

Columns:
{len(df.columns)}

Column Names:
{list(df.columns)}

Data Types:

{df.dtypes}

==================================================
MISSING VALUES
==================================================

{df.isnull().sum()}

==================================================
NUMERICAL SUMMARY
==================================================

{df.describe(include='all')}

==================================================
DATA SAMPLE
==================================================

{sample}

==================================================
USER QUESTION
==================================================

{question}

==================================================
ANSWER
==================================================
"""

    return prompt