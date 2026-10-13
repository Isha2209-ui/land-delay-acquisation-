import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
                handle_unknown="ignore",
                sparse_output=False
            )
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


# Fit preprocessing only on the training data
X_train_processed = preprocessor.fit_transform(X_train)


# Transform the validation data
X_val_processed = preprocessor.transform(X_val)


# Select the 30 best features
selector = SelectKBest(
    score_func=f_classif,
    k=30
)


# Select features using only training data
X_train_selected = selector.fit_transform(
    X_train_processed,
    y_train
)


# Apply the same feature selection to validation data
X_val_selected = selector.transform(
    X_val_processed
)


# Remove any columns that are constant in the training data
constant_columns = (
    X_train_selected.std(axis=0) == 0
)


# Keep only non-constant columns
X_train_selected = X_train_selected[
    :, ~constant_columns
]

X_val_selected = X_val_selected[
    :, ~constant_columns
]


# Display the number of features
print("\n===== LDA FEATURE SET =====")

print(
    "Features after selection:",
    X_train_selected.shape[1]
)


# Create the LDA model
model = LinearDiscriminantAnalysis()


# Train the LDA model
print("\nTraining LDA...")

model.fit(
    X_train_selected,
    y_train
)

print("Training completed!")


# Predict validation classes
y_pred = model.predict(
    X_val_selected
)


# Predict validation probabilities
y_prob = model.predict_proba(
    X_val_selected
)[:, 1]


# Calculate accuracy
accuracy = accuracy_score(
    y_val,
    y_pred
)


# Calculate precision
precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)


# Calculate recall
recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)


# Calculate F1 score
f1 = f1_score(
    y_val,
    y_pred,
    zero_division=0
)


# Calculate ROC-AUC
roc_auc = roc_auc_score(
    y_val,
    y_prob
)


# Display the results
print("\n===== LDA VALIDATION RESULTS =====")

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