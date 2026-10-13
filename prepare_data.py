import pandas as pd

# Read the original dataset
df = pd.read_csv("SIH26017_MoSPI_1477_projects.csv")

# Keep only the columns needed for the model
features = [
    "sector",
    "agency",
    "state",
    "approval_date",
    "orig_commissioning_date",
    "orig_cost_cr"
]

# Add the target column
target = "is_delayed"

# Create the model dataset
model_df = df[features + [target]].copy()

# Convert approval date into a date
model_df["approval_date"] = pd.to_datetime(
    model_df["approval_date"],
    errors="coerce"
)

# Convert commissioning date into a date
model_df["orig_commissioning_date"] = pd.to_datetime(
    model_df["orig_commissioning_date"],
    errors="coerce"
)

# Show missing values
print("\nMissing values:")
print(model_df.isnull().sum())

# Show the target distribution
print("\nTarget distribution:")
print(model_df["is_delayed"].value_counts())

# Save the cleaned dataset
model_df.to_csv("model_data.csv", index=False)

print("\nCleaned dataset saved as model_data.csv")
print("Shape:", model_df.shape)