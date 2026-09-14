"""Pydantic v2 request/response models: the API contract, validated before the model ever sees the data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from churn_service.config import MAX_BATCH_SIZE

YesNo = Literal["Yes", "No"]
InternetAddOn = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "customer_id": "7590-VHVEG", "gender": "Female", "senior_citizen": 0, "partner": "Yes", "dependents": "No",
    "tenure": 1, "phone_service": "No", "multiple_lines": "No phone service", "internet_service": "DSL",
    "online_security": "No", "online_backup": "Yes", "device_protection": "No", "tech_support": "No",
    "streaming_tv": "No", "streaming_movies": "No", "contract": "Month-to-month", "paperless_billing": "Yes",
    "payment_method": "Electronic check", "monthly_charges": 29.85, "total_charges": 29.85,
}
# API field (snake_case) → dataset column name used by the trained pipeline
COLUMN_NAMES = {
    "gender": "gender", "senior_citizen": "SeniorCitizen", "partner": "Partner", "dependents": "Dependents",
    "tenure": "tenure", "phone_service": "PhoneService", "multiple_lines": "MultipleLines",
    "internet_service": "InternetService", "online_security": "OnlineSecurity", "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection", "tech_support": "TechSupport", "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies", "contract": "Contract", "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod", "monthly_charges": "MonthlyCharges", "total_charges": "TotalCharges",
}
INTERNET_ADD_ONS = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    customer_id: str | None = Field(default=None, max_length=64)
    gender: Literal["Female", "Male"]
    senior_citizen: Literal[0, 1]
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="months with the company")
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: InternetAddOn
    online_backup: InternetAddOn
    device_protection: InternetAddOn
    tech_support: InternetAddOn
    streaming_tv: InternetAddOn
    streaming_movies: InternetAddOn
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
    monthly_charges: float = Field(gt=0, le=500, allow_inf_nan=False)
    total_charges: float | None = Field(default=None, ge=0, le=60_000, allow_inf_nan=False,
                                        description="may be omitted only for new customers (tenure 0)")

    @model_validator(mode="after")
    def check_consistency(self) -> Customer:
        no_internet = self.internet_service == "No"
        for name in INTERNET_ADD_ONS:
            if no_internet != (getattr(self, name) == "No internet service"):
                raise ValueError(f"{name} must be 'No internet service' exactly when internet_service is 'No'")
        if (self.phone_service == "No") != (self.multiple_lines == "No phone service"):
            raise ValueError("multiple_lines must be 'No phone service' exactly when phone_service is 'No'")
        if self.total_charges is None and self.tenure > 0:
            raise ValueError("total_charges is required when tenure > 0")
        return self

    def to_record(self) -> dict:
        return {column: getattr(self, field) for field, column in COLUMN_NAMES.items()}


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: float = Field(ge=0, le=1)
    contact: bool = Field(description="True when churn_probability ≥ threshold: send a retention offer")
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    model_version: str
    threshold: float
    n_customers: int
    n_contact: int
    predictions: list[Prediction]
