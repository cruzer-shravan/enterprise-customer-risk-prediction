import pandas as pd

from src.models.business_impact import build_business_impact_report
from src.models.reporting import confusion_matrix_report


def test_business_impact_report_contains_deciles_and_campaign_scenarios():
    X = pd.DataFrame({"CLTV": [5000, 3000, 2000, 1000]})
    y_true = ["Yes", "No", "Yes", "No"]
    probabilities = [0.9, 0.7, 0.4, 0.1]

    report = build_business_impact_report(X, y_true, probabilities)

    assert report["assumptions"]["customer_value_field"] == "CLTV"
    assert report["decile_capture"]
    assert report["campaign_scenarios"]
    assert report["campaign_scenarios"][0]["targeted_customers"] >= 1


def test_confusion_matrix_report_uses_churn_labels():
    report = confusion_matrix_report(["Yes", "No", "Yes", "No"], ["Yes", "Yes", "No", "No"])

    assert report == {
        "labels": ["No", "Yes"],
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
