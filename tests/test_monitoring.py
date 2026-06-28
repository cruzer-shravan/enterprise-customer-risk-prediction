from pathlib import Path

import pandas as pd

from src.data.load_data import load_data
from src.monitoring.drift_detection import build_reference_profile, summarize_batch


def test_build_reference_profile_handles_numeric_strings():
    df = load_data(Path("tests/fixtures/sample_churn.csv"))
    df["Total Charges"] = df["Total Charges"].astype(str)
    df.loc[0, "Total Charges"] = " "

    profile = build_reference_profile(df)

    assert profile["numeric"]["Total Charges"]["missing_rate"] == 0.25
    assert profile["numeric"]["Total Charges"]["mean"] > 0


def test_summarize_batch_handles_raw_csv_numeric_strings():
    df = load_data(Path("tests/fixtures/sample_churn.csv"))
    df["Total Charges"] = df["Total Charges"].astype(str)
    predictions = pd.DataFrame(
        [
            {"churn_probability": 0.8, "risk_band": "High"},
            {"churn_probability": 0.2, "risk_band": "Low"},
        ]
    )

    summary = summarize_batch(df.head(2), predictions)

    assert summary["prediction"]["row_count"] == 2
    assert summary["prediction"]["average_churn_probability"] == 0.5
