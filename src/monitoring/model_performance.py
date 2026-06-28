from datetime import datetime, timezone

import pandas as pd


def prediction_log_rows(predictions, source: str):
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []

    for prediction in predictions:
        drivers = prediction.get("risk_drivers", [])
        rows.append(
            {
                "timestamp": timestamp,
                "source": source,
                "prediction": prediction["prediction"],
                "churn_probability": prediction["churn_probability"],
                "risk_band": prediction["risk_band"],
                "top_risk_driver": drivers[0]["feature"] if drivers else None,
            }
        )

    return rows


def append_prediction_log(predictions, path, source: str = "unknown"):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(prediction_log_rows(predictions, source=source))

    if path.exists():
        existing_rows = pd.read_csv(path)
        new_rows = pd.concat([existing_rows, new_rows], ignore_index=True)

    new_rows.to_csv(path, index=False)
