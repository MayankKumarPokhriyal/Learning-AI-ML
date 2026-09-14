"""Feature engineering and preprocessing pipelines (all fitted inside scikit-learn pipelines)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from churn_service.data import FEATURE_COLUMNS, INTERNET_DEPENDENT

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]
SERVICE_COLUMNS = ["PhoneService", "MultipleLines", *INTERNET_DEPENDENT]
ENGINEERED_FEATURES = ["n_services", "charges_ratio"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only (no statistics learned from data), so they can never leak across splits."""
    out = X.copy()
    out["n_services"] = X[SERVICE_COLUMNS].eq("Yes").sum(axis=1).astype("int64")
    expected_total = X["tenure"] * X["MonthlyCharges"]
    # > 1: paid more than today's price implies (price went down); < 1: price went up. New customers → 1.0
    out["charges_ratio"] = (X["TotalCharges"] / expected_total.where(expected_total > 0)).fillna(1.0)
    return out


class FeatureAdder(TransformerMixin, BaseEstimator):
    """Pipeline step wrapping `add_features`, so the saved model computes features itself at serving time."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_features(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*self.feature_names_in_, *ENGINEERED_FEATURES], dtype=object)


def build_preprocessor(kind: str) -> ColumnTransformer:
    """'linear': impute + scale numbers, one-hot categories. 'tree': ordinal-encode categories, pass numbers through."""
    numeric = NUMERIC_FEATURES + ENGINEERED_FEATURES
    if kind == "linear":
        return ColumnTransformer(
            [
                ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ],
            verbose_feature_names_out=False,
        )
    if kind == "tree":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan)
        return ColumnTransformer(
            [("categorical", encoder, CATEGORICAL_FEATURES), ("numeric", "passthrough", numeric)],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")  # keep column names so the booster knows which columns are categorical
    raise ValueError(f"unknown preprocessor kind {kind!r} (use 'linear' or 'tree')")
