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


# Separate input features
X = df.drop(columns=["is_delayed"])


# Separate target
y = df["is_delayed"]


# Create the same train, validation and test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

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


# Process training data
X_train_processed = preprocessor.fit_transform(X_train)

# Process validation data
X_val_processed = preprocessor.transform(X_val)


# -------------------------------
# MODEL 1: ALL FEATURES
# -------------------------------

model_all = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train using all features
model_all.fit(
    X_train_processed,
    y_train
)

# Predict validation data
pred_all = model_all.predict(X_val_processed)

# Predict probabilities
prob_all = model_all.predict_proba(
    X_val_processed
)[:, 1]


# -------------------------------
# MODEL 2: SELECTED FEATURES
# -------------------------------

selector = SelectKBest(
    score_func=f_classif,
    k=30
)

# Select features using training data
X_train_selected = selector.fit_transform(
    X_train_processed,
    y_train
)

# Apply selection to validation data
X_val_selected = selector.transform(
    X_val_processed
)


model_selected = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train selected-feature model
model_selected.fit(
    X_train_selected,
    y_train
)

# Predict validation data
pred_selected = model_selected.predict(
    X_val_selected
)

# Predict probabilities
prob_selected = model_selected.predict_proba(
    X_val_selected
)[:, 1]


# Function to calculate results
def get_results(y_true, predictions, probabilities):

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities
        )
    }


# Calculate results
results_all = get_results(
    y_val,
    pred_all,
    prob_all
)

results_selected = get_results(
    y_val,
    pred_selected,
    prob_selected
)


# Display results
print("\n===== VALIDATION COMPARISON =====")

print("\nAll 119 features:")

for metric, value in results_all.items():
    print(
        metric + ":",
        round(value, 4)
    )


print("\nSelected 30 features:")

for metric, value in results_selected.items():
    print(
        metric + ":",
        round(value, 4)
    )