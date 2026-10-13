
import joblib
import pandas as pd
from pathlib import Path

from risk_engine import calculate_risk_score, get_risk_level


# ============================================================
# 1. LOAD MODEL AND DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "land_delay_model.pkl"
DATA_PATH = BASE_DIR / "model_data.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

model_package = joblib.load(MODEL_PATH)
options_df = pd.read_csv(DATA_PATH)

preprocessor = model_package["preprocessor"]
selector = model_package["selector"]
model = model_package["model"]
features = model_package["features"]


# Validate required dataset columns
required_columns = {"sector", "agency", "state"}
missing_columns = required_columns - set(options_df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {sorted(missing_columns)}"
    )


# Clean option values without changing the saved model
for column in ["sector", "agency", "state"]:
    options_df[column] = (
        options_df[column]
        .astype("string")
        .str.strip()
    )

options_df = options_df.dropna(subset=["sector", "agency", "state"])

SECTORS = sorted(options_df["sector"].unique().tolist())
STATES = sorted(options_df["state"].unique().tolist())

# Only agencies present in the selected sector
AGENCIES_BY_SECTOR = {
    sector: sorted(
        options_df.loc[
            options_df["sector"] == sector, "agency"
        ].unique().tolist()
    )
    for sector in SECTORS
}


# ============================================================
# 2. RISK EXPLANATIONS
# ============================================================

def get_risk_details(risk_level):
    details = {
        "LOW": {
            "meaning": (
                "The model estimates a relatively low probability of delay."
            ),
            "reasons": [
                "Estimated delay probability is 30% or lower.",
                "The model assigns the project a relatively low delay-risk score.",
                "Continue checking progress and keeping project records updated."
            ],
            "action": (
                "Continue routine monitoring and maintain documentation."
            )
        },

        "MEDIUM": {
            "meaning": (
                "The model estimates a moderate probability of delay."
            ),
            "reasons": [
                "Estimated delay probability is above 30% and up to 60%.",
                "The prediction warrants attention to project progress.",
                "Review pending approvals, land records and acquisition milestones."
            ],
            "action": (
                "Monitor progress more frequently and review pending tasks."
            )
        },

        "HIGH": {
            "meaning": (
                "The model estimates a high probability of delay."
            ),
            "reasons": [
                "Estimated delay probability is above 60% and up to 80%.",
                "The predicted risk warrants a detailed project review.",
                "Investigate possible approval, compensation or land-dispute bottlenecks."
            ],
            "action": (
                "Review potential bottlenecks and prepare a mitigation plan."
            )
        },

        "CRITICAL": {
            "meaning": (
                "The model estimates a very high probability of delay."
            ),
            "reasons": [
                "Estimated delay probability is above 80%.",
                "The project warrants priority review.",
                "Check land availability, pending approvals, disputes and "
                "rehabilitation issues where applicable."
            ],
            "action": (
                "Prioritize a detailed review, verify actual bottlenecks "
                "and assign corrective actions."
            )
        }
    }

    return details.get(risk_level, {
        "meaning": "Risk explanation is unavailable.",
        "reasons": ["Verify the prediction and project data."],
        "action": "Review the project manually."
    })


# ============================================================
# 3. PREDICTION FUNCTION
# ============================================================

def predict_project(
    sector,
    agency,
    state,
    orig_cost_cr,
    approval_year,
    approval_month,
    planned_duration_months
):
    project = pd.DataFrame([{
        "sector": sector,
        "agency": agency,
        "state": state,
        "orig_cost_cr": float(orig_cost_cr),
        "approval_year": int(approval_year),
        "approval_month": int(approval_month),
        "planned_duration_months": float(planned_duration_months)
    }])

    # Match the feature order used during training
    project = project[features]

    project_processed = preprocessor.transform(project)
    project_selected = selector.transform(project_processed)

    prediction = int(model.predict(project_selected)[0])

    # Probability of class 1 (delayed)
    classes = list(model.classes_)
    delayed_class_index = classes.index(1)

    probability = float(
        model.predict_proba(project_selected)[0][delayed_class_index]
    )

    risk_score = calculate_risk_score(probability)
    risk_level = get_risk_level(risk_score)
    details = get_risk_details(risk_level)

    return {
        "prediction": prediction,
        "delay_probability": round(probability * 100, 2),
        "risk_score": round(float(risk_score), 2),
        "risk_level": risk_level,
        "risk_meaning": details["meaning"],
        "risk_reasons": details["reasons"],
        "recommended_action": details["action"]
    }


# ============================================================
# 4. INTERACTIVE INPUT HELPERS
# ============================================================

def ask_choice(label, options):
    if not options:
        raise ValueError(f"No options are available for {label}.")

    print(f"\nAvailable {label} options:")

    for index, option in enumerate(options, start=1):
        print(f"{index:>3}. {option}")

    while True:
        value = input(
            f"\nSelect {label} by number (1-{len(options)}): "
        ).strip()

        try:
            choice = int(value)

            if 1 <= choice <= len(options):
                return options[choice - 1]

        except ValueError:
            pass

        print(f"Please enter a number from 1 to {len(options)}.")


def ask_number(label, default, converter):
    while True:
        value = input(f"{label} [{default}]: ").strip()

        try:
            return converter(value if value else default)
        except ValueError:
            print("Please enter a valid number.")


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("       LAND ACQUISITION DELAY PREDICTION SYSTEM")
    print("=" * 65)
    print("Choose options by number. Press Enter for numeric defaults.")

    # Choose sector
    sector = ask_choice("sector", SECTORS)

    # Filter agency options using the selected sector
    agencies = AGENCIES_BY_SECTOR.get(sector, [])

    if not agencies:
        print(f"No agencies found for sector: {sector}")
        return

    agency = ask_choice(f"agency for {sector}", agencies)

    # Choose state
    state = ask_choice("state", STATES)

    # Enter numeric project details
    print("\nEnter project details:")

    cost = ask_number("Project cost (₹ crore)", "12", float)
    year = ask_number("Approval year", "2020", int)
    month = ask_number("Approval month (1-12)", "5", int)
    duration = ask_number("Planned duration (months)", "12", float)

    if cost < 0:
        print("Project cost cannot be negative.")
        return

    if duration <= 0:
        print("Planned duration must be greater than zero.")
        return

    if not 1 <= month <= 12:
        print("Approval month must be between 1 and 12.")
        return

    try:
        result = predict_project(
            sector=sector,
            agency=agency,
            state=state,
            orig_cost_cr=cost,
            approval_year=year,
            approval_month=month,
            planned_duration_months=duration
        )

        status = (
            "DELAYED"
            if result["prediction"] == 1
            else "NOT DELAYED"
        )

        print("\n" + "=" * 65)
        print("                     PREDICTION RESULT")
        print("=" * 65)

        print(f"Sector                 : {sector}")
        print(f"Agency                 : {agency}")
        print(f"State                  : {state}")
        print(f"Project cost           : ₹{cost:.2f} crore")
        print(f"Approval year/month    : {year}/{month}")
        print(f"Planned duration       : {duration:.1f} months")

        print("-" * 65)
        print(f"Predicted status       : {status}")
        print(
            f"Delay probability      : "
            f"{result['delay_probability']:.2f}%"
        )
        print(
            f"Risk score             : "
            f"{result['risk_score']:.2f}/100"
        )
        print(f"Risk category          : {result['risk_level']}")

        print("\n" + "-" * 65)
        print("RISK EXPLANATION")
        print("-" * 65)
        print(result["risk_meaning"])

        print("\nREASONS / POINTS TO INVESTIGATE")

        for number, reason in enumerate(
            result["risk_reasons"], start=1
        ):
            print(f"{number}. {reason}")

        print("\nRECOMMENDED ACTION")
        print(result["recommended_action"])

        print("\n" + "=" * 65)
        print("Note: This is a model estimate, not a guarantee.")
        print(
            "The listed reasons are category-based guidance, "
            "not confirmed causes identified by the model."
        )
        print("=" * 65)

    except Exception as error:
        print(f"\nPrediction failed: {error}")
        print(
            "Check that the selected values and input data match "
            "the trained model's expected features."
        )


if __name__ == "__main__":
    main()
