# Import pandas for reading the project dataset
import pandas as pd

# Import the prediction function
from predict_project import predict_project


# Read the original MoSPI dataset
df = pd.read_csv("SIH26017_MoSPI_1477_projects.csv")

# Select the first real project from the dataset
project = df.iloc[0]

# Convert the approval date into a date
approval_date = pd.to_datetime(
    project["approval_date"],
    errors="coerce"
)

# Convert the commissioning date into a date
commissioning_date = pd.to_datetime(
    project["orig_commissioning_date"],
    errors="coerce"
)

# Extract the approval year
approval_year = approval_date.year

# Extract the approval month
approval_month = approval_date.month

# Calculate the planned project duration in months
planned_duration_months = (
    (commissioning_date - approval_date).days / 30.44
)

# Send the real project information to the prediction model
result = predict_project(
    sector=project["sector"],
    agency=project["agency"],
    state=project["state"],
    orig_cost_cr=project["orig_cost_cr"],
    approval_year=approval_year,
    approval_month=approval_month,
    planned_duration_months=planned_duration_months
)

# Display the original project name
print("Project:", project["project_name"])

# Display the actual delay status from the dataset
print("Actual Status:", project["is_delayed"])

# Display the model prediction
print("Predicted Status:", result["prediction"])

# Display the predicted probability of delay
print("Delay Probability:", result["delay_probability"], "%")

# Display the risk score
print("Risk Score:", result["risk_score"])

# Display the risk category
print("Risk Level:", result["risk_level"])