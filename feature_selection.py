import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# Read the feature dataset
df = pd.read_csv("features_data.csv")


# Separate the input features
X = df.drop(columns=["is_delayed"])


# Separate the target
y = df["is_delayed"]


# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Split validation and test data
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# Define categorical columns
categorical_features = [
    "sector",
    "agency",
    "state"
]


# Define numerical columns
numerical_features = [
    "orig_cost_cr",
    "approval_year",
    "approval_month",
    "planned_duration_months"
]


# Prepare numerical data
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Prepare categorical data
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# Process only the training data
X_train_processed = preprocessor.fit_transform(X_train)


# Get the names of the processed features
feature_names = preprocessor.get_feature_names_out()


# Create Logistic Regression for feature selection
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train the model
model.fit(X_train_processed, y_train)


# Get the coefficient values
coefficients = model.coef_[0]


# Create a table of features and coefficients
feature_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})


# Calculate the absolute coefficient
feature_table["importance"] = (
    feature_table["coefficient"].abs()
)


# Sort features by importance
feature_table = feature_table.sort_values(
    "importance",
    ascending=False
)


# Display the top 20 features
print("\n===== TOP 20 FEATURES =====")

print(
    feature_table[
        ["feature", "coefficient", "importance"]
    ].head(20).to_string(index=False)
)


# Save the feature ranking
feature_table.to_csv(
    "feature_importance.csv",
    index=False
)


print("\nFeature ranking saved as feature_importance.csv")