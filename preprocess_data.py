import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Read the feature dataset
df = pd.read_csv("features_data.csv")


# Separate the input features
X = df.drop(columns=["is_delayed"])


# Separate the target
y = df["is_delayed"]


# Split the data into training and temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Split the temporary data into validation and test data
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# List the categorical columns
categorical_features = [
    "sector",
    "agency",
    "state"
]


# List the numerical columns
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
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Combine both preprocessing pipelines
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


# Learn preprocessing only from training data
X_train_processed = preprocessor.fit_transform(X_train)


# Apply the same preprocessing to validation data
X_val_processed = preprocessor.transform(X_val)


# Apply the same preprocessing to test data
X_test_processed = preprocessor.transform(X_test)


# Print the original feature count
print("\nOriginal number of features:")
print(X_train.shape[1])


# Print the processed feature count
print("\nProcessed number of features:")
print(X_train_processed.shape[1])


# Print the final dataset sizes
print("\nProcessed dataset sizes:")
print("Train:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Test:", X_test_processed.shape)


print("\nPreprocessing completed successfully!")