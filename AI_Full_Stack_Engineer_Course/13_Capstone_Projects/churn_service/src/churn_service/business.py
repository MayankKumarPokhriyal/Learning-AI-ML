"""The business metric: expected profit of a retention campaign, and the profit-maximizing threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_service.config import BusinessAssumptions

DEFAULT_THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def expected_profit(y_true, contact, monthly_charges, assumptions: BusinessAssumptions) -> float:
    """Total expected profit (USD) versus doing nothing.

    Contacting a real churner:  acceptance_rate × monthly_charges × months_retained − offer_cost
    Contacting a non-churner:   − offer_cost (they would have stayed anyway)
    Not contacting anyone:      0
    """
    y_true = np.asarray(y_true, dtype=float)
    contact = np.asarray(contact, dtype=float)
    value_if_saved = assumptions.acceptance_rate * np.asarray(monthly_charges, dtype=float) * assumptions.months_retained
    return float(np.sum(contact * (y_true * value_if_saved - assumptions.offer_cost)))


def profit_per_1000(y_true, contact, monthly_charges, assumptions: BusinessAssumptions) -> float:
    return expected_profit(y_true, contact, monthly_charges, assumptions) / len(np.asarray(y_true)) * 1000


def break_even_probability(monthly_charges: float, assumptions: BusinessAssumptions) -> float:
    """Contact when p × acceptance × value > offer_cost, i.e. p > offer_cost / (acceptance × value)."""
    return assumptions.offer_cost / (assumptions.acceptance_rate * monthly_charges * assumptions.months_retained)


def profit_curve(y_true, proba, monthly_charges, assumptions: BusinessAssumptions, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    rows = []
    for threshold in thresholds:
        contact = proba >= threshold
        n_contact = int(contact.sum())
        true_pos = int((contact & (y_true == 1)).sum())
        rows.append({
            "threshold": float(threshold),
            "profit_per_1000": profit_per_1000(y_true, contact, monthly_charges, assumptions),
            "contact_rate": n_contact / len(y_true),
            "precision": true_pos / n_contact if n_contact else 0.0,
            "recall": true_pos / max(int((y_true == 1).sum()), 1),
        })
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> float:
    """Highest profit; on an exact tie prefer the higher threshold (fewer offers for the same money)."""
    best_profit = curve["profit_per_1000"].max()
    return float(curve.loc[np.isclose(curve["profit_per_1000"], best_profit), "threshold"].max())
