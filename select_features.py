import pandas as pd

# Read the project data
df = pd.read_csv("SIH26017_MoSPI_1477_projects.csv")

# Show all available columns
print("\nAll columns:")
for col in df.columns:
    print("-", col)

# Columns that describe the project before completion
safe_features = [
    "sector",
    "agency",
    "state",
    "approval_date",
    "orig_commissioning_date",
    "orig_cost_cr"
]

# Column we want to predict
target = "is_delayed"

# Columns that contain information from the project after it started
leakage_columns = [
    "rev_commissioning_date",
    "ant_commissioning_date",
    "rev_cost_cr",
    "ant_cost_cr",
    "cumulative_exp_cr",
    "physical_progress_pct",
    "delay_months",
    "cost_overrun_pct",
    "expenditure_ratio"
]

print("\nTarget column:")
print(target)

print("\nSafe feature columns:")
for col in safe_features:
    print("-", col)

print("\nColumns NOT used for prediction:")
for col in leakage_columns:
    print("-", col)