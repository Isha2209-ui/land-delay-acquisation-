
def calculate_risk_score(probability):
    """
    Convert delay probability (0-1) to a risk score (0-100).
    """
    probability = float(probability)

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    return round(probability * 100, 2)


def get_risk_level(score):
    """
    Classify risk using the project's existing thresholds.
    """
    score = float(score)

    if score <= 30:
        return "LOW"
    elif score <= 60:
        return "MEDIUM"
    elif score <= 80:
        return "HIGH"
    else:
        return "CRITICAL"


if __name__ == "__main__":
    score = calculate_risk_score(0.9564)
    level = get_risk_level(score)

    print(f"Risk score: {score}/100")
    print(f"Risk category: {level}")
