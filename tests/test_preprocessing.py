from pathlib import Path

from src.data.load_data import load_data
from src.data.validate_data import validate_inference_data, validate_training_data
from src.features.feature_builder import build_features
from src.features.preprocessing import preprocess
from src.pipelines.inference_pipeline import prepare_inference_features


def test_preprocess_preserves_both_target_classes():
    df = load_data(Path("tests/fixtures/sample_churn.csv"))

    processed = preprocess(df)

    assert set(processed["Churn Label"]) == {"Yes", "No"}
    assert processed.shape[0] == df.shape[0]


def test_preprocess_converts_total_charges_to_numeric():
    df = load_data(Path("tests/fixtures/sample_churn.csv"))

    processed = preprocess(df)

    assert processed["Total Charges"].dtype.kind in {"f", "i"}


def test_build_features_removes_target_leakage_and_id_columns():
    df = preprocess(load_data(Path("tests/fixtures/sample_churn.csv")))

    X, y = build_features(df)

    assert "Churn Label" not in X.columns
    assert "Churn Value" not in X.columns
    assert "Churn Score" not in X.columns
    assert "Churn Reason" not in X.columns
    assert "CustomerID" not in X.columns
    assert y.tolist() == ["Yes", "No", "Yes", "No"]


def test_validate_training_data_accepts_valid_fixture():
    df = load_data(Path("tests/fixtures/sample_churn.csv"))

    validated = validate_training_data(df)

    assert validated.equals(df)


def test_validate_training_data_rejects_missing_required_columns():
    df = load_data(Path("tests/fixtures/sample_churn.csv")).drop(columns=["CLTV"])

    try:
        validate_training_data(df)
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
    else:
        raise AssertionError("Expected validation to reject missing columns")


def test_validate_inference_data_rejects_invalid_categories():
    df = load_data(Path("tests/fixtures/sample_churn.csv")).drop(columns=["Churn Label"])
    df.loc[0, "Contract"] = "Forever"

    try:
        validate_inference_data(df)
    except ValueError as exc:
        assert "Invalid categorical values" in str(exc)
        assert "Contract" in str(exc)
    else:
        raise AssertionError("Expected validation to reject invalid categories")


def test_validate_inference_data_rejects_invalid_numeric_ranges():
    df = load_data(Path("tests/fixtures/sample_churn.csv")).drop(columns=["Churn Label"])
    df.loc[0, "Monthly Charges"] = -10

    try:
        validate_inference_data(df)
    except ValueError as exc:
        assert "Invalid numeric ranges" in str(exc)
        assert "Monthly Charges" in str(exc)
    else:
        raise AssertionError("Expected validation to reject invalid numeric ranges")


def test_prepare_inference_features_allows_blank_numeric_values():
    df = load_data(Path("tests/fixtures/sample_churn.csv")).drop(columns=["Churn Label"])
    df["Total Charges"] = df["Total Charges"].astype(str)
    df.loc[0, "Total Charges"] = " "

    features = prepare_inference_features(df)

    assert features["Total Charges"].isna().sum() == 1
