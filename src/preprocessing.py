
import pandas as pd
from sklearn.preprocessing import StandardScaler


NUMERICAL_COLUMNS = [
    "Age", "Years_At_Company", "Monthly_Income",
    "Job_Satisfaction", "Weekly_Hours", "Projects_Handled"
]

CATEGORICAL_COLUMNS = [
    "Gender", "Department", "Education",
    "Job_Level", "Remote_Work", "Performance_Rating"
]


def preprocess_employee_data(path):
    df = pd.read_csv(path)

    # Employee ID is an identifier, not a predictive feature.
    df = df.drop(columns=["Employee_ID"])

    # Numerical missing values -> median.
    for col in NUMERICAL_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    # Categorical missing values -> mode.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna(df[col].mode()[0])

    # One-hot encode categorical features.
    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int
    )

    # IQR outlier removal.
    mask = pd.Series(True, index=df.index)
    for col in ["Monthly_Income", "Weekly_Hours"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)

    df = df.loc[mask].copy()

    # Standardize numerical columns.
    scaler = StandardScaler()
    df[NUMERICAL_COLUMNS] = scaler.fit_transform(df[NUMERICAL_COLUMNS])

    return df, scaler
