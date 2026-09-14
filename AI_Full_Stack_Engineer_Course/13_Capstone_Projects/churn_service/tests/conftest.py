"""Shared fixtures. Unit tests use small hand-written frames; API and quality tests need trained artifacts."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_service import config
from churn_service.schemas import EXAMPLE_CUSTOMER

RAW_ROWS = [
    # customerID, gender, Senior, Partner, Dependents, tenure, Phone, MultipleLines, Internet, 6 add-ons, Contract, Paperless, Payment, Monthly, Total, Churn
    ["0001-A", "Female", 0, "Yes", "No", 1, "No", "No phone service", "DSL", "No", "Yes", "No", "No", "No", "No",
     "Month-to-month", "Yes", "Electronic check", 29.85, "29.85", "No"],
    ["0002-B", "Male", 0, "No", "No", 34, "Yes", "No", "DSL", "Yes", "No", "Yes", "No", "No", "No",
     "One year", "No", "Mailed check", 56.95, "1889.5", "No"],
    ["0003-C", "Male", 1, "No", "No", 2, "Yes", "Yes", "Fiber optic", "No", "No", "No", "No", "Yes", "Yes",
     "Month-to-month", "Yes", "Electronic check", 99.65, "199.3", "Yes"],
    ["0004-D", "Female", 0, "Yes", "Yes", 0, "Yes", "No", "No", "No internet service", "No internet service",
     "No internet service", "No internet service", "No internet service", "No internet service",
     "Two year", "No", "Bank transfer (automatic)", 20.25, " ", "No"],
]


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    from churn_service.data import RAW_COLUMNS

    return pd.DataFrame(RAW_ROWS, columns=RAW_COLUMNS)


@pytest.fixture
def customer() -> dict:
    return dict(EXAMPLE_CUSTOMER)


@pytest.fixture(scope="session")
def artifacts_dir():
    directory = config.artifacts_dir()
    if not (directory / "model.joblib").is_file():
        pytest.fail(f"no trained model in '{directory.name}/' — run `python -m churn_service.train` before the tests", pytrace=False)
    return directory
