from src.config.settings import DEFAULT_THRESHOLD, POSITIVE_CLASS
from src.models.explain import explain_prediction


def risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def predict_churn(model, data, threshold: float = DEFAULT_THRESHOLD, explain: bool = True):
    probabilities = model.predict_proba(data)
    positive_class_index = list(model.classes_).index(POSITIVE_CLASS)
    positive_probabilities = probabilities[:, positive_class_index]

    predictions = []
    for row_index, probability in enumerate(positive_probabilities):
        result = {
            "prediction": POSITIVE_CLASS if probability >= threshold else "No",
            "churn_probability": float(probability),
            "risk_band": risk_band(float(probability)),
        }
        if explain:
            result["risk_drivers"] = explain_prediction(model, data.iloc[row_index])
        predictions.append(result)

    return predictions
