import pandas as pd

# Read the cleaned dataset
df = pd.read_csv("model_data.csv")

# Convert dates into proper date format
df["approval_date"] = pd.to_datetime(
    df["approval_date"],
    errors="coerce"
)

df["orig_commissioning_date"] = pd.to_datetime(
    df["orig_commissioning_date"],
    errors="coerce"
)

# Get the year when the project was approved
df["approval_year"] = df["approval_date"].dt.year

# Get the month when the project was approved
df["approval_month"] = df["approval_date"].dt.month

# Calculate the planned project duration in months
df["planned_duration_months"] = (
    (df["orig_commissioning_date"] - df["approval_date"])
    .dt.days / 30.44
)

# Remove the original date columns
df = df.drop(
    columns=[
        "approval_date",
        "orig_commissioning_date"
    ]
)

# Show the new columns
print("\nNew dataset columns:")
print(df.columns.tolist())

# Show a few rows
print("\nSample data:")
print(df.head())

# Save the feature dataset
df.to_csv("features_data.csv", index=False)

print("\nFeature dataset saved as features_data.csv")
print("Shape:", df.shape)
