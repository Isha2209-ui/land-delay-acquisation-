from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "SIH26017_MoSPI_1477_projects.csv"
FRONTEND_DIR = BASE_DIR / "frontend"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SIH26017 Land Acquisition Delay Prediction System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)
print(
    df.loc[
        df["sector"].str.contains("coal", case=False, na=False)
        & df["agency"].str.contains("AAI", case=False, na=False),
        ["project_code", "project_name", "sector", "agency"]
    ].to_string(index=False)
)


# ============================================================
# DATE PROCESSING
# ============================================================

df["approval_date"] = pd.to_datetime(
    df["approval_date"],
    errors="coerce"
)

df["orig_commissioning_date"] = pd.to_datetime(
    df["orig_commissioning_date"],
    errors="coerce"
)

df["approval_year"] = (
    df["approval_date"].dt.year
)

df["approval_month"] = (
    df["approval_date"].dt.month
)

df["planned_duration_months"] = (
    (
        df["orig_commissioning_date"]
        - df["approval_date"]
    ).dt.days / 30.44
)


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "sector",
    "agency",
    "state",
    "orig_cost_cr",
    "approval_year",
    "approval_month",
    "planned_duration_months"
]

CATEGORICAL_FEATURES = [
    "sector",
    "agency",
    "state"
]

NUMERICAL_FEATURES = [
    "orig_cost_cr",
    "approval_year",
    "approval_month",
    "planned_duration_months"
]


# ============================================================
# TARGET
# ============================================================

df["is_delayed"] = pd.to_numeric(
    df["is_delayed"],
    errors="coerce"
)

df = df[
    df["is_delayed"].isin([0, 1])
].copy()

X = df[MODEL_FEATURES].copy()
y = df["is_delayed"].astype(int)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

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


# ============================================================
# PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
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
])

preprocessor = ColumnTransformer([
    (
        "numerical",
        numerical_pipeline,
        NUMERICAL_FEATURES
    ),
    (
        "categorical",
        categorical_pipeline,
        CATEGORICAL_FEATURES
    )
])


# ============================================================
# FIT PREPROCESSOR
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)


# ============================================================
# FEATURE SELECTION
# ============================================================

K_FEATURES = min(
    30,
    X_train_processed.shape[1]
)

selector = SelectKBest(
    score_func=f_classif,
    k=K_FEATURES
)

X_train_selected = selector.fit_transform(
    X_train_processed,
    y_train
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

base_model = LogisticRegression(
    C=0.1,
    max_iter=2000,
    random_state=42,
    class_weight=None
)

model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=5
)

model.fit(
    X_train_selected,
    y_train
)
# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_rows(input_df):

    processed = preprocessor.transform(
        input_df
    )

    selected = selector.transform(
        processed
    )

    predictions = model.predict(
        selected
    )

    probabilities = model.predict_proba(
        selected
    )[:, 1]

    return predictions, probabilities


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):

    score = float(probability) * 100

    if score <= 30:
        return "LOW"

    elif score <= 60:
        return "MEDIUM"

    elif score <= 80:
        return "HIGH"

    else:
        return "CRITICAL"


# ============================================================
# DESCRIPTION
# ============================================================

def get_prediction_description(probability):

    score = float(probability) * 100

    if score <= 30:

        return (
            f"The project has a low estimated probability "
            f"of delay ({score:.1f}%). No immediate "
            f"intervention is indicated."
        )

    elif score <= 60:

        return (
            f"The project has a moderate estimated "
            f"probability of delay ({score:.1f}%). "
            f"Routine monitoring is recommended."
        )

    elif score <= 80:

        return (
            f"The project has a high estimated probability "
            f"of delay ({score:.1f}%). Closer monitoring "
            f"and early intervention are recommended."
        )

    else:

        return (
            f"The project has a very high estimated "
            f"probability of delay ({score:.1f}%). "
            f"Immediate monitoring and intervention "
            f"are recommended."
        )


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    sector: str
    agency: str
    state: str

    orig_cost_cr: float = Field(gt=0)

    approval_year: int

    approval_month: int = Field(
        ge=1,
        le=12
    )

    planned_duration_months: float = Field(
        gt=0
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "dataset_rows": int(len(df)),
        "model": "Logistic Regression",
        "features": MODEL_FEATURES,
        "test_rows": int(len(X_test))
    }


# ============================================================
# OPTIONS
# ============================================================


@app.get("/options")
def get_options():
    sector_agency = (
        df[["sector", "agency"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(["sector", "agency"])
    )

    agencies_by_sector = {
        sector: sorted(group["agency"].unique().tolist())
        for sector, group in sector_agency.groupby("sector")
    }

    return {
        "sector": sorted(df["sector"].dropna().astype(str).unique().tolist()),
        "agency": sorted(df["agency"].dropna().astype(str).unique().tolist()),
        "agencies_by_sector": agencies_by_sector,
        "state": sorted(df["state"].dropna().astype(str).unique().tolist())
    }



# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    input_df = pd.DataFrame([{

        "sector": request.sector,

        "agency": request.agency,

        "state": request.state,

        "orig_cost_cr":
            request.orig_cost_cr,

        "approval_year":
            request.approval_year,

        "approval_month":
            request.approval_month,

        "planned_duration_months":
            request.planned_duration_months

    }])


    try:

        predictions, probabilities = predict_rows(
            input_df
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


    probability = float(
        probabilities[0]
    )

    prediction = int(
        predictions[0]
    )

    risk_score = probability * 100

    risk_level = get_risk_level(
        probability
    )


    if prediction == 1:

        prediction_text = (
            "Project is predicted to face a delay."
        )

    else:

        prediction_text = (
            "Project is predicted to remain on schedule."
        )


    return {

        "prediction":
            prediction_text,

        "predicted_class":
            prediction,

        "delay_probability":
            round(
                risk_score,
                1
            ),

        "risk_score":
            round(
                risk_score,
                1
            ),

        "risk_level":
            risk_level,

        "description":
            get_prediction_description(
                probability
            )

    }


# ============================================================
# HIGH-RIS@app.get("/high-risk-projects")K PROJECTS
# ============================================================


def high_risk_projects():

    prediction_input = df[
        MODEL_FEATURES
    ].copy()


    try:

        predictions, probabilities = (
            predict_rows(
                prediction_input
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


    result = df[
        [
            "project_code",
            "project_name",
            "state",
            "sector",
            "agency",
            "is_delayed"
        ]
    ].copy()


    result["predicted_class"] = predictions

    result["delay_probability"] = probabilities

    result["risk_score"] = (
        result["delay_probability"] * 100
    )

    result["risk_level"] = (
        result["delay_probability"]
        .apply(get_risk_level)
    )
    result = result.sort_values(
        by="risk_score",
        ascending=False
    )

    return result.replace(
        {float("nan"): None}
    ).to_dict(orient="records")
    

    result["actual_status"] = (
        result["is_delayed"].astype(int)
    )


    # Sort by actual model probability
    result = result.sort_values(
        by="delay_probability",
        ascending=False
    )


    # Top 20 highest-risk projects
    result = result.head(20)


    output = []


    for _, row in result.iterrows():

        probability = float(
            row["delay_probability"]
        )

        score = float(
            row["risk_score"]
        )


        output.append({

            "project_code":
                str(row["project_code"]),

            "project_name":
                str(row["project_name"]),

            "state":
                str(row["state"]),

            "sector":
                str(row["sector"]),

            "agency":
                str(row["agency"]),

            # IMPORTANT:
            # 3 decimal places so values like
            # 99.9975 don't all become 100.0
            "delay_probability":
                round(
                    probability * 100,
                    3
                ),

            "risk_score":
                round(
                    score,
                    3
                ),

            "risk_level":
                str(row["risk_level"]),

            "actual_status":
                int(row["actual_status"])

        })


    return output


# ============================================================
# ANALYTICS - SECTOR
# ============================================================

@app.get("/analytics/sector")
def analytics_sector():

    stats = (
        df.groupby("sector")
        .agg(
            project_count=(
                "project_code",
                "count"
            ),

            delayed_projects=(
                "is_delayed",
                "sum"
            )
        )
        .reset_index()
    )


    stats["delay_rate"] = (
        stats["delayed_projects"]
        /
        stats["project_count"]
        * 100
    )


    return stats[
        [
            "sector",
            "project_count",
            "delayed_projects",
            "delay_rate"
        ]
    ].to_dict(
        orient="records"
    )


# ============================================================
# ANALYTICS - YEAR
# ============================================================

@app.get("/analytics/year")
def analytics_year():

    year_df = df[
        df["approval_year"].notna()
    ].copy()


    stats = (
        year_df.groupby("approval_year")
        .agg(
            project_count=(
                "project_code",
                "count"
            ),

            delayed_projects=(
                "is_delayed",
                "sum"
            )
        )
        .reset_index()
    )


    stats["year"] = (
        stats["approval_year"]
        .astype(int)
    )


    stats["delay_rate"] = (
        stats["delayed_projects"]
        /
        stats["project_count"]
        * 100
    )


    return stats[
        [
            "year",
            "project_count",
            "delayed_projects",
            "delay_rate"
        ]
    ].to_dict(
        orient="records"
    )
# ============================================================
# ANALYTICS - DELAY DISTRIBUTION
# ============================================================

@app.get("/analytics/delay-distribution")
def analytics_delay_distribution():
    delayed = int((df["is_delayed"] == 1).sum())
    not_delayed = int((df["is_delayed"] == 0).sum())

    return {
        "labels": ["Delayed", "Not Delayed"],
        "counts": [delayed, not_delayed]
    }


# ============================================================
# ANALYTICS - PROJECT COST HISTOGRAM
# ============================================================

@app.get("/analytics/cost-distribution")
def analytics_cost_distribution():
    import numpy as np

    costs = pd.to_numeric(
        df["orig_cost_cr"], errors="coerce"
    ).dropna()

    costs = costs[costs >= 0]

    if costs.empty:
        return {"labels": [], "counts": []}

    counts, edges = np.histogram(costs, bins=8)

    labels = [
        f"{edges[i]:,.0f}–{edges[i + 1]:,.0f}"
        for i in range(len(edges) - 1)
    ]

    return {
        "labels": labels,
        "counts": counts.tolist()
    }


# ============================================================
# ANALYTICS - COST VS. PLANNED DURATION
# ============================================================

@app.get("/analytics/cost-duration")
def analytics_cost_duration():
    data = df[
        ["orig_cost_cr", "planned_duration_months", "is_delayed"]
    ].copy()

    data["orig_cost_cr"] = pd.to_numeric(
        data["orig_cost_cr"], errors="coerce"
    )
    data["planned_duration_months"] = pd.to_numeric(
        data["planned_duration_months"], errors="coerce"
    )

    data = data.dropna()
    data = data[
        (data["orig_cost_cr"] >= 0)
        & (data["planned_duration_months"] > 0)
    ]

    # Limit the response size for a responsive dashboard.
    if len(data) > 1000:
        data = data.sample(n=1000, random_state=42)

    return {
        "points": [
            {
                "x": float(row["orig_cost_cr"]),
                "y": float(row["planned_duration_months"]),
                "delayed": int(row["is_delayed"])
            }
            for _, row in data.iterrows()
        ]
    }



# ============================================================
# ANALYTICS SUMMARY
# ============================================================

@app.get("/analytics/summary")
def analytics_summary():

    total_projects = len(df)

    delayed_projects = int(
        df["is_delayed"].sum()
    )

    on_time_projects = (
        total_projects -
        delayed_projects
    )

    delay_rate = (
        delayed_projects /
        total_projects *
        100
    ) if total_projects else 0


    return {

        "total_projects":
            int(total_projects),

        "delayed_projects":
            delayed_projects,

        "on_time_projects":
            int(on_time_projects),

        "delay_rate":
            round(
                delay_rate,
                2
            )

    }
@app.get("/high-risk-projects")
def high_risk_projects():
    prediction_input = df[MODEL_FEATURES].copy()

    try:
        predictions, probabilities = predict_rows(prediction_input)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    result = df[
        [
            "project_code",
            "project_name",
            "state",
            "sector",
            "agency",
            "is_delayed"
        ]
    ].copy()

    result["predicted_class"] = predictions
    result["delay_probability"] = probabilities
    result["risk_score"] = result["delay_probability"] * 100
    result["risk_level"] = result["delay_probability"].apply(get_risk_level)

    result = result.sort_values("risk_score", ascending=False)

    return result.replace({float("nan"): None}).to_dict(orient="records")


# ============================================================
# FRONTEND
# ============================================================

if FRONTEND_DIR.exists():

    app.mount(
        "/",
        StaticFiles(
            directory=str(FRONTEND_DIR),
            html=True
        ),
        name="frontend"
    )