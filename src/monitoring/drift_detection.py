import json

import pandas as pd

from src.config.settings import CATEGORICAL_VALUES, NUMERIC_RANGES


def build_reference_profile(df):
    profile = {"row_count": int(len(df)), "numeric": {}, "categorical": {}}

    for column in NUMERIC_RANGES:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        profile["numeric"][column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "missing_rate": float(values.isna().mean()),
        }

    for column in CATEGORICAL_VALUES:
        if column not in df.columns:
            continue
        profile["categorical"][column] = {
            "distribution": df[column].value_counts(normalize=True, dropna=False).to_dict(),
            "missing_rate": float(df[column].isna().mean()),
        }

    return profile


def summarize_batch(df, predictions, reference_profile=None):
    summary = build_reference_profile(df)
    summary["prediction"] = {
        "row_count": int(len(predictions)),
        "average_churn_probability": float(predictions["churn_probability"].mean()),
        "risk_band_distribution": predictions["risk_band"]
        .value_counts(normalize=True)
        .to_dict(),
    }

    if reference_profile:
        summary["drift"] = compare_to_reference(summary, reference_profile)

    return summary


def compare_to_reference(batch_profile, reference_profile):
    drift = {"numeric_mean_shift": {}, "categorical_top_change": {}}

    for column, batch_stats in batch_profile["numeric"].items():
        reference_stats = reference_profile.get("numeric", {}).get(column)
        if not reference_stats:
            continue
        reference_std = reference_stats.get("std") or 1.0
        drift["numeric_mean_shift"][column] = round(
            (batch_stats["mean"] - reference_stats["mean"]) / reference_std,
            4,
        )

    for column, batch_stats in batch_profile["categorical"].items():
        reference_stats = reference_profile.get("categorical", {}).get(column)
        if not reference_stats:
            continue
        batch_top = max(batch_stats["distribution"], key=batch_stats["distribution"].get)
        reference_top = max(
            reference_stats["distribution"], key=reference_stats["distribution"].get
        )
        drift["categorical_top_change"][column] = {
            "reference": reference_top,
            "batch": batch_top,
            "changed": batch_top != reference_top,
        }

    return drift


def save_profile(profile, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_profile(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
