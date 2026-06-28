from pathlib import Path

from src.data.load_data import load_data
from src.data.split_data import split_data
from src.features.feature_builder import build_features
from src.features.preprocessing import preprocess
from src.models.evaluate import evaluate, tune_threshold
from src.models.predict import predict_churn
from src.models.train import get_model_candidates, train_model


EXPECTED_METRICS = {"roc_auc", "accuracy", "precision", "recall", "f1", "brier_score"}


def test_training_pipeline_components_fit_and_evaluate():
    df = preprocess(load_data(Path("tests/fixtures/sample_churn.csv")))
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5)

    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    assert set(metrics) == EXPECTED_METRICS
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_model_candidates_include_baseline_models():
    df = preprocess(load_data(Path("tests/fixtures/sample_churn.csv")))
    X, _ = build_features(df)

    candidates = get_model_candidates(X)

    assert "logistic_regression" in candidates
    assert "random_forest" in candidates
    assert "extra_trees" in candidates
    assert "gradient_boosting" in candidates
    assert "hist_gradient_boosting" in candidates
    assert "adaboost" in candidates
    assert "svc_rbf" in candidates
    assert "mlp_neural_network" in candidates
    assert "lightgbm" in candidates
    assert "catboost" in candidates


def test_threshold_tuning_and_prediction_bands():
    df = preprocess(load_data(Path("tests/fixtures/sample_churn.csv")))
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5)

    model = train_model(X_train, y_train)
    threshold, metrics = tune_threshold(model, X_test, y_test)
    predictions = predict_churn(model, X_test, threshold=threshold)

    assert 0.1 <= threshold <= 0.9
    assert set(metrics) == EXPECTED_METRICS
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert set(predictions[0]) == {
        "prediction",
        "churn_probability",
        "risk_band",
        "risk_drivers",
    }
    assert predictions[0]["risk_drivers"]
    assert set(predictions[0]["risk_drivers"][0]) == {
        "feature",
        "display_name",
        "value",
        "importance",
        "importance_percent",
    }
