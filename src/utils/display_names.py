FEATURE_DISPLAY_NAMES = {
    "Latitude": "Customer Latitude",
    "Longitude": "Customer Longitude",
    "Gender": "Gender",
    "Senior Citizen": "Senior Citizen",
    "Partner": "Partner Status",
    "Dependents": "Has Dependents",
    "Tenure Months": "Customer Tenure",
    "Phone Service": "Phone Subscription",
    "Multiple Lines": "Multiple Phone Lines",
    "Internet Service": "Internet Connection",
    "Online Security": "Online Security Service",
    "Online Backup": "Online Backup Service",
    "Device Protection": "Device Protection Plan",
    "Tech Support": "Technical Support",
    "Streaming TV": "TV Streaming",
    "Streaming Movies": "Movie Streaming",
    "Contract": "Contract Type",
    "Paperless Billing": "Paperless Billing",
    "Payment Method": "Payment Method",
    "Monthly Charges": "Monthly Bill",
    "Total Charges": "Total Amount Paid",
    "CLTV": "Customer Lifetime Value",
}


def display_name(feature_name: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)