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
    roc_auc_score,
    confusion_matrix
)


# Read the feature dataset
df = pd.read_csv("features_data.csv")


# Separate the input features
X = df.drop(columns=["is_delayed"])


# Separate the target
y = df["is_delayed"]


# Create the same 70/15/15 split
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


# Define categorical features
categorical_features = [
    "sector",
    "agency",
    "state"
]


# Define numerical features
numerical_features = [
    "orig_cost_cr",
    "approval_year",
    "approval_month",
    "planned_duration_months"
]


# Prepare numerical features
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


# Prepare categorical features
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
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


# Learn preprocessing using training data only
X_train_processed = preprocessor.fit_transform(X_train)


# Transform validation data
X_val_processed = preprocessor.transform(X_val)


# Transform test data
X_test_processed = preprocessor.transform(X_test)


# Select the 30 best features
selector = SelectKBest(
    score_func=f_classif,
    k=30
)


# Learn feature selection from training data
X_train_selected = selector.fit_transform(
    X_train_processed,
    y_train
)


# Apply feature selection to validation data
X_val_selected = selector.transform(
    X_val_processed
)


# Apply feature selection to test data
X_test_selected = selector.transform(
    X_test_processed
)


# Remove constant columns
constant_columns = (
    X_train_selected.std(axis=0) == 0
)


X_train_selected = X_train_selected[
    :, ~constant_columns
]

X_val_selected = X_val_selected[
    :, ~constant_columns
]

X_test_selected = X_test_selected[
    :, ~constant_columns
]


# Create the final Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train the final model
print("\nTraining final Logistic Regression...")

model.fit(
    X_train_selected,
    y_train
)

print("Training completed!")


# Predict the test classes
y_pred = model.predict(
    X_test_selected
)


# Predict probability of delay
y_prob = model.predict_proba(
    X_test_selected
)[:, 1]


# Calculate final metrics
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# Calculate confusion matrix
matrix = confusion_matrix(
    y_test,
    y_pred
)


# Display final results
print("\n================================")
print("FINAL MODEL TEST RESULTS")
print("================================")

print(
    "Accuracy :",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1 Score :",
    round(f1, 4)
)

print(
    "ROC-AUC  :",
    round(roc_auc, 4)
)


# Display confusion matrix
print("\nConfusion Matrix:")
print(matrix)


print("\nFinal model evaluation completed!")