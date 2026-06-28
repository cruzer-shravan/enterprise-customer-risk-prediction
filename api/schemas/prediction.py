from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


YesNo = Literal["No", "Yes"]


class CustomerRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    latitude: float = Field(..., alias="Latitude", ge=-90, le=90)
    longitude: float = Field(..., alias="Longitude", ge=-180, le=180)
    gender: Literal["Female", "Male"] = Field(..., alias="Gender")
    senior_citizen: YesNo = Field(..., alias="Senior Citizen")
    partner: YesNo = Field(..., alias="Partner")
    dependents: YesNo = Field(..., alias="Dependents")
    tenure_months: int = Field(..., alias="Tenure Months", ge=0, le=120)
    phone_service: YesNo = Field(..., alias="Phone Service")
    multiple_lines: Literal["No", "No phone service", "Yes"] = Field(..., alias="Multiple Lines")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(..., alias="Internet Service")
    online_security: Literal["No", "No internet service", "Yes"] = Field(..., alias="Online Security")
    online_backup: Literal["No", "No internet service", "Yes"] = Field(..., alias="Online Backup")
    device_protection: Literal["No", "No internet service", "Yes"] = Field(..., alias="Device Protection")
    tech_support: Literal["No", "No internet service", "Yes"] = Field(..., alias="Tech Support")
    streaming_tv: Literal["No", "No internet service", "Yes"] = Field(..., alias="Streaming TV")
    streaming_movies: Literal["No", "No internet service", "Yes"] = Field(..., alias="Streaming Movies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., alias="Contract")
    paperless_billing: YesNo = Field(..., alias="Paperless Billing")
    payment_method: Literal[
        "Bank transfer",
        "Bank transfer (automatic)",
        "Credit card",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(..., alias="Payment Method")
    monthly_charges: float = Field(..., alias="Monthly Charges", ge=0)
    total_charges: float = Field(..., alias="Total Charges", ge=0)
    cltv: int = Field(..., alias="CLTV", ge=0)


class PredictionRequest(BaseModel):
    records: list[CustomerRecord] = Field(..., min_length=1)


class RiskDriver(BaseModel):
    feature: str
    display_name: str
    value: str | None = None
    importance: float
    importance_percent: float


class PredictionResult(BaseModel):
    prediction: str
    churn_probability: float
    risk_band: str
    risk_drivers: list[RiskDriver] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    predictions: list[PredictionResult]
