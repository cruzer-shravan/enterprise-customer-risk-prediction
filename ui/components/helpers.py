"""
helpers.py
Reusable helper functions.
"""

import json

from src.config.settings import (
    DEFAULT_THRESHOLD,
    METRICS_PATH,
    MODEL_METADATA_PATH,
)


def load_json(path):

    if path.exists():
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    return None


def load_threshold():

    metrics = load_json(METRICS_PATH)

    if metrics:

        return metrics.get(
            "threshold",
            DEFAULT_THRESHOLD,
        )

    return DEFAULT_THRESHOLD


def load_model_metadata():

    metadata = load_json(
        MODEL_METADATA_PATH
    )

    if metadata:

        return metadata

    return {}


def format_risk_drivers(drivers):

    return "; ".join(
        f"{driver['display_name']}={driver['value']} ({driver['importance_percent']:.1f}%)"
        for driver in drivers
    )