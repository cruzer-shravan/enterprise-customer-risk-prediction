import json

import pandas as pd

from src.config.settings import (
    DEFAULT_THRESHOLD,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PREDICTION_LOG_PATH,
)
from src.monitoring.model_performance import append_prediction_log
from src.models.predict import predict_churn
from src.models.registry import load_model
from src.pipelines.inference_pipeline import prepare_inference_features


class ModelService:
    def __init__(self):
        self._model = None
        self._threshold = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(MODEL_PATH)
        return self._model

    @property
    def threshold(self):
        if self._threshold is None:
            self._threshold = DEFAULT_THRESHOLD
            if METRICS_PATH.exists():
                metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                self._threshold = metrics.get("threshold", DEFAULT_THRESHOLD)
        return self._threshold

    def predict(self, records):
        normalized_records = [
            record.model_dump(by_alias=True) if hasattr(record, "model_dump") else record
            for record in records
        ]
        raw_data = pd.DataFrame(normalized_records)
        features = prepare_inference_features(raw_data)
        predictions = predict_churn(self.model, features, threshold=self.threshold)
        append_prediction_log(predictions, PREDICTION_LOG_PATH, source="api")
        return predictions

    def metadata(self):
        if not MODEL_METADATA_PATH.exists():
            return {
                "model_path": str(MODEL_PATH),
                "threshold": self.threshold,
                "metadata_available": False,
            }
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        metadata["metadata_available"] = True
        return metadata


model_service = ModelService()
