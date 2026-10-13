import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


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


# Combine preprocessing
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


# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform validation data
X_val_processed = preprocessor.transform(X_val)

# Transform test data
X_test_processed = preprocessor.transform(X_test)


# Select the 30 most useful features
selector = SelectKBest(
    score_func=f_classif,
    k=30
)


# Learn feature selection from training data
X_train_selected = selector.fit_transform(
    X_train_processed,
    y_train
)


# Apply selection to validation data
X_val_selected = selector.transform(
    X_val_processed
)


# Apply selection to test data
X_test_selected = selector.transform(
    X_test_processed
)


# Show feature counts
print("\n===== FEATURE SELECTION =====")

print(
    "Features before selection:",
    X_train_processed.shape[1]
)

print(
    "Features after selection:",
    X_train_selected.shape[1]
)


# Get feature names
feature_names = preprocessor.get_feature_names_out()


# Get selected feature positions
selected_indices = selector.get_support()


# Get selected feature names
selected_features = feature_names[selected_indices]


# Display selected features
print("\n===== SELECTED FEATURES =====")

for feature in selected_features:
    print("-", feature)


# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train the model
model.fit(
    X_train_selected,
    y_train
)


# Predict validation data
y_val_pred = model.predict(X_val_selected)

# Predict validation probabilities
y_val_prob = model.predict_proba(
    X_val_selected
)[:, 1]


# Calculate validation metrics
accuracy = accuracy_score(
    y_val,
    y_val_pred
)

precision = precision_score(
    y_val,
    y_val_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_val_pred,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_val_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_val,
    y_val_prob
)


# Display results
print("\n===== SELECTED FEATURE MODEL =====")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# Save selected feature names
pd.DataFrame({
    "selected_feature": selected_features
}).to_csv(
    "selected_features.csv",
    index=False
)


print("\nSelected features saved as selected_features.csv")