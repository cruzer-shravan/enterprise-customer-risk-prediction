from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "model.pkl"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "trained" / "model_metadata.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "metrics.json"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_comparison.json"
BUSINESS_IMPACT_PATH = PROJECT_ROOT / "reports" / "metrics" / "business_impact.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "metrics" / "feature_importance.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "reports" / "metrics" / "confusion_matrix.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REFERENCE_PROFILE_PATH = PROJECT_ROOT / "reports" / "monitoring" / "reference_profile.json"
PREDICTION_LOG_PATH = PROJECT_ROOT / "reports" / "monitoring" / "prediction_log.csv"
BATCH_MONITORING_PATH = PROJECT_ROOT / "reports" / "monitoring" / "latest_batch_summary.json"

TARGET_COLUMN = "Churn Label"
POSITIVE_CLASS = "Yes"
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.5

# Model selection
MODEL_SELECTION_TOLERANCE = 0.01

LEAKAGE_COLUMNS = [
    "Churn Value",
    "Churn Score",
    "Churn Reason",
]

ID_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
]

REQUIRED_COLUMNS = [
    TARGET_COLUMN,
    "Latitude",
    "Longitude",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]

INFERENCE_REQUIRED_COLUMNS = [column for column in REQUIRED_COLUMNS if column != TARGET_COLUMN]

CATEGORICAL_VALUES = {
    "Gender": {"Female", "Male"},
    "Senior Citizen": {"No", "Yes"},
    "Partner": {"No", "Yes"},
    "Dependents": {"No", "Yes"},
    "Phone Service": {"No", "Yes"},
    "Multiple Lines": {"No", "No phone service", "Yes"},
    "Internet Service": {"DSL", "Fiber optic", "No"},
    "Online Security": {"No", "No internet service", "Yes"},
    "Online Backup": {"No", "No internet service", "Yes"},
    "Device Protection": {"No", "No internet service", "Yes"},
    "Tech Support": {"No", "No internet service", "Yes"},
    "Streaming TV": {"No", "No internet service", "Yes"},
    "Streaming Movies": {"No", "No internet service", "Yes"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "Paperless Billing": {"No", "Yes"},
    "Payment Method": {
        "Bank transfer",
        "Bank transfer (automatic)",
        "Credit card",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    },
}

NUMERIC_RANGES = {
    "Latitude": (-90, 90),
    "Longitude": (-180, 180),
    "Tenure Months": (0, 120),
    "Monthly Charges": (0, None),
    "Total Charges": (0, None),
    "CLTV": (0, None),
}

FEATURE_DISPLAY_NAMES = {
    # Numeric Features
    "Latitude": "Customer Latitude",
    "Longitude": "Customer Longitude",
    "Tenure Months": "Customer Tenure (Months)",
    "Monthly Charges": "Monthly Charges ($)",
    "Total Charges": "Total Charges ($)",
    "CLTV": "Customer Lifetime Value",

    # Binary Features
    "Gender": "Gender",
    "Senior Citizen": "Senior Citizen",
    "Partner": "Partner status",
    "Dependents": "Has Dependents",
    "Phone Service": "Phone Subscription",
    "Multiple Lines": "Multiple Phone Lines",

    # Internet
    "Internet Service": "Internet Connection",
    "Online Security": "Online Security Service",
    "Online Backup": "Online Backup Service",
    "Device Protection": "Device Protection Plan",
    "Tech Support": "Technical Support",

    # Entertainment
    "Streaming TV": "TV Streaming",
    "Streaming Movies": "Movie Streaming",

    # Billing
    "Contract": "Contract Type",
    "Paperless Billing": "Paperless Billing",
    "Payment Method": "Payment Method",
}
