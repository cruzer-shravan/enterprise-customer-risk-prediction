from fastapi.testclient import TestClient

from api.app import app
from api.routes import predict as predict_route


VALID_RECORD = {
    "Latitude": 33.973951,
    "Longitude": -118.248405,
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 89.1,
    "Total Charges": 1069.2,
    "CLTV": 2700,
}


def test_health_endpoint_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_endpoint_returns_metadata(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(
        predict_route.model_service,
        "metadata",
        lambda: {"metadata_available": False, "threshold": 0.3},
    )

    response = client.get("/model")

    assert response.status_code == 200
    assert response.json() == {"metadata_available": False, "threshold": 0.3}


def test_predict_endpoint_returns_predictions(monkeypatch):
    client = TestClient(app)

    def fake_predict(records):
        return [{"prediction": "No", "churn_probability": 0.12, "risk_band": "Low"}]

    monkeypatch.setattr(predict_route.model_service, "predict", fake_predict)

    response = client.post("/predict", json={"records": [VALID_RECORD]})

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [
            {
                "prediction": "No",
                "churn_probability": 0.12,
                "risk_band": "Low",
                "risk_drivers": [],
            }
        ]
    }


def test_predict_endpoint_rejects_incomplete_records():
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"records": [{"Tenure Months": 12, "Monthly Charges": 70.0}]},
    )

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_category_before_scoring(monkeypatch):
    client = TestClient(app)
    record = {**VALID_RECORD, "Contract": "Forever"}

    def fail_if_called(records):
        raise AssertionError("model service should not be called for invalid schema")

    monkeypatch.setattr(predict_route.model_service, "predict", fail_if_called)

    response = client.post("/predict", json={"records": [record]})

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_numeric_range_before_scoring(monkeypatch):
    client = TestClient(app)
    record = {**VALID_RECORD, "Monthly Charges": -10}

    def fail_if_called(records):
        raise AssertionError("model service should not be called for invalid schema")

    monkeypatch.setattr(predict_route.model_service, "predict", fail_if_called)

    response = client.post("/predict", json={"records": [record]})

    assert response.status_code == 422


def test_predict_endpoint_converts_pipeline_validation_errors_to_422(monkeypatch):
    client = TestClient(app)

    def fake_predict(records):
        raise ValueError("Invalid categorical values: {'Contract': ['Forever']}")

    monkeypatch.setattr(predict_route.model_service, "predict", fake_predict)

    response = client.post("/predict", json={"records": [VALID_RECORD]})

    assert response.status_code == 422
    assert "Invalid categorical values" in response.json()["detail"]
