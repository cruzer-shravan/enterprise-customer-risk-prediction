import pandas as pd

from src.config.settings import (
    CATEGORICAL_VALUES,
    INFERENCE_REQUIRED_COLUMNS,
    NUMERIC_RANGES,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


def _missing_columns(df, required_columns):
    return [column for column in required_columns if column not in df.columns]


def _validate_required_columns(df, required_columns):
    missing_columns = _missing_columns(df, required_columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _validate_categorical_values(df):
    errors = {}
    for column, allowed_values in CATEGORICAL_VALUES.items():
        if column not in df.columns:
            continue
        invalid_values = sorted(set(df[column].dropna().astype(str)) - allowed_values)
        if invalid_values:
            errors[column] = invalid_values

    if errors:
        raise ValueError(f"Invalid categorical values: {errors}")


def _validate_numeric_ranges(df):
    errors = {}
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        invalid_numeric = df[column].notna() & values.isna()
        if invalid_numeric.any():
            errors[column] = "must be numeric"
            continue
        values = values.dropna()
        if minimum is not None and (values < minimum).any():
            errors[column] = f"must be >= {minimum}"
        if maximum is not None and (values > maximum).any():
            errors[column] = f"must be <= {maximum}"

    if errors:
        raise ValueError(f"Invalid numeric ranges: {errors}")


def validate_training_data(df):
    _validate_required_columns(df, REQUIRED_COLUMNS)
    _validate_categorical_values(df)
    _validate_numeric_ranges(df)

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column contains missing values: {TARGET_COLUMN}")

    invalid_targets = sorted(set(df[TARGET_COLUMN].dropna()) - {"Yes", "No"})
    if invalid_targets:
        raise ValueError(f"Invalid target values: {invalid_targets}")

    if "CustomerID" in df.columns and df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID contains duplicate values")

    return df


def validate_inference_data(df):
    _validate_required_columns(df, INFERENCE_REQUIRED_COLUMNS)
    _validate_categorical_values(df)
    _validate_numeric_ranges(df)

    required_non_numeric_columns = [
        column for column in INFERENCE_REQUIRED_COLUMNS if column not in NUMERIC_RANGES
    ]
    missing_values = [
        column for column in required_non_numeric_columns if df[column].isna().any()
    ]
    if missing_values:
        raise ValueError(f"Missing inference values in columns: {missing_values}")

    return df
