"""Project-wide constants, paths, and the business assumptions behind the decision threshold."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# IBM Telco Customer Churn sample data (Apache-2.0, IBM/telco-customer-churn-on-icp4d on GitHub)
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

ID_COLUMN = "customerID"
TARGET = "Churn"
SEED = 42
MODEL_NAME = "churn-classifier"
MAX_BATCH_SIZE = 1000

# src/churn_service/config.py -> parents[2] is the project folder when running from a source checkout.
# Installed packages (e.g. in Docker) should set CHURN_ARTIFACTS_DIR / CHURN_DATA_DIR instead.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def artifacts_dir() -> Path:
    """Where the trained model, metadata, and monitoring reference live."""
    return Path(os.environ.get("CHURN_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))


def data_dir() -> Path:
    """Where the raw download and the saved train/validation/test splits live."""
    return Path(os.environ.get("CHURN_DATA_DIR", PROJECT_ROOT / "data"))


@dataclass(frozen=True)
class BusinessAssumptions:
    """ASSUMPTIONS, not facts from the dataset. Replace them with numbers from your finance/retention team.

    - offer_cost: what one retention offer costs (discount + agent time), paid for every contacted customer.
    - acceptance_rate: share of would-be churners who stay because they received the offer.
    - months_retained: months of revenue kept when a churner is saved.
    """

    offer_cost: float = 60.0
    acceptance_rate: float = 0.35
    months_retained: int = 12
