import pandas as pd

# Read the feature dataset
df = pd.read_csv("features_data.csv")

# Check the dataset size
print("\nDataset shape:")
print(df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check numerical columns
print("\nNumerical summary:")
print(
    df[
        [
            "orig_cost_cr",
            "approval_year",
            "approval_month",
            "planned_duration_months"
        ]
    ].describe()
)

# Check target values
print("\nTarget values:")
print(df["is_delayed"].value_counts())

# Check for invalid planned durations
print("\nProjects with negative planned duration:")
print(
    df[df["planned_duration_months"] < 0].shape[0]
)
