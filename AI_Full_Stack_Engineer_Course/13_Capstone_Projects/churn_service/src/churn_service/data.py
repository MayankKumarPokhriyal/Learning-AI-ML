"""Download, validate, clean, and split the IBM Telco Customer Churn data."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_service.config import DATA_SHA256, DATA_URL, ID_COLUMN, SEED, TARGET

YES_NO = ("No", "Yes")
INTERNET_DEPENDENT = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORY_VALUES: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ("No", "No phone service", "Yes"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    **{column: ("No", "No internet service", "Yes") for column in INTERNET_DEPENDENT},
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ("Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"),
}
NUMERIC_RANGES = {"SeniorCitizen": (0, 1), "tenure": (0, 120), "MonthlyCharges": (0.01, 500.0), "TotalCharges": (0.0, 60_000.0)}
RAW_COLUMNS = [
    ID_COLUMN, "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
    "InternetService", *INTERNET_DEPENDENT, "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges", TARGET,
]
FEATURE_COLUMNS = [c for c in RAW_COLUMNS if c not in (ID_COLUMN, TARGET)]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(dest: str | Path, url: str = DATA_URL, expected_sha256: str | None = DATA_SHA256, timeout: int = 60) -> Path:
    """Download once, verify the checksum (so a silently changed upstream file can't change your model), reuse later."""
    dest = Path(dest)
    if dest.exists() and (expected_sha256 is None or file_sha256(dest) == expected_sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response, tempfile.NamedTemporaryFile(
        dir=dest.parent, suffix=".part", delete=False
    ) as tmp:
        shutil.copyfileobj(response, tmp)
    digest = file_sha256(tmp.name)
    if expected_sha256 is not None and digest != expected_sha256:
        Path(tmp.name).unlink()
        raise ValueError(f"checksum mismatch for {url}: got {digest[:12]}…, expected {expected_sha256[:12]}…")
    Path(tmp.name).replace(dest)
    return dest


def load_raw(path: str | Path) -> pd.DataFrame:
    # TotalCharges is text in the source file because 11 rows contain a single space instead of a number
    return pd.read_csv(path, dtype={ID_COLUMN: "str", "TotalCharges": "str"})


def parse_total_charges(values: pd.Series) -> pd.Series:
    """Text or numbers → float64; blanks, None, and non-numbers become NaN."""
    if pd.api.types.is_numeric_dtype(values):
        return values.astype("float64")
    text = values.astype("string").str.strip()
    parsed = pd.to_numeric(text, errors="coerce").astype("Float64")
    return pd.Series(parsed.to_numpy(dtype="float64", na_value=np.nan), index=values.index, name=values.name)


@dataclass
class ValidationReport:
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("data validation failed:\n- " + "\n- ".join(self.errors))

    def __str__(self) -> str:
        lines = [f"{'✅ passed' if self.ok else '❌ failed'}: {self.n_rows:,} rows, {len(self.errors)} errors, {len(self.warnings)} warnings"]
        lines += [f"  ERROR   {e}" for e in self.errors] + [f"  WARNING {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_raw(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """Schema + data-quality checks. Errors block training; warnings are known, handled quirks."""
    report = ValidationReport(n_rows=len(df))
    expected = RAW_COLUMNS if require_target else [c for c in RAW_COLUMNS if c != TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")
        return report
    extra = [c for c in df.columns if c not in RAW_COLUMNS]
    if extra:
        report.warnings.append(f"unexpected extra columns ignored: {extra}")
    if len(df) == 0:
        report.errors.append("no rows")
        return report

    n_dup = int(df[ID_COLUMN].duplicated().sum())
    if n_dup:
        report.errors.append(f"{n_dup} duplicate {ID_COLUMN} values")
    if require_target:
        bad_target = sorted(set(df[TARGET].dropna().unique()) - set(YES_NO))
        if bad_target or df[TARGET].isna().any():
            report.errors.append(f"{TARGET} must be Yes/No (found {bad_target}, {int(df[TARGET].isna().sum())} missing)")

    for column, allowed in CATEGORY_VALUES.items():
        n_missing = int(df[column].isna().sum())
        unexpected = sorted(set(df[column].dropna().unique()) - set(allowed))
        if n_missing:
            report.errors.append(f"{column}: {n_missing} missing values")
        if unexpected:
            report.errors.append(f"{column}: unexpected values {unexpected[:5]}")

    numeric = {}
    for column in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        numeric[column] = pd.to_numeric(df[column], errors="coerce")
        if numeric[column].isna().any():
            report.errors.append(f"{column}: {int(numeric[column].isna().sum())} missing or non-numeric values")
    total = parse_total_charges(df["TotalCharges"])
    for column, values in [*numeric.items(), ("TotalCharges", total)]:
        low, high = NUMERIC_RANGES[column]
        n_out = int(((values < low) | (values > high)).sum())
        if n_out:
            report.errors.append(f"{column}: {n_out} values outside [{low}, {high}]")

    tenure, monthly = numeric["tenure"], numeric["MonthlyCharges"]
    blank_with_tenure = int((total.isna() & tenure.gt(0)).sum())
    if blank_with_tenure:
        report.errors.append(f"TotalCharges: {blank_with_tenure} missing/non-numeric values for customers with tenure > 0")
    blank_new = int((total.isna() & tenure.eq(0)).sum())
    if blank_new:
        report.warnings.append(f"TotalCharges: {blank_new} blank values, all with tenure == 0 (not billed yet) → cleaned to 0.0")

    no_internet = df["InternetService"].eq("No")
    for column in INTERNET_DEPENDENT:
        n_bad = int((no_internet != df[column].eq("No internet service")).sum())
        if n_bad:
            report.errors.append(f"{column}: {n_bad} rows inconsistent with InternetService")
    n_bad_phone = int((df["PhoneService"].eq("No") != df["MultipleLines"].eq("No phone service")).sum())
    if n_bad_phone:
        report.errors.append(f"MultipleLines: {n_bad_phone} rows inconsistent with PhoneService")

    billed = tenure.gt(0) & total.notna()
    ratio = total[billed] / (tenure[billed] * monthly[billed])
    n_odd = int(((ratio < 0.5) | (ratio > 2.0)).sum())
    if n_odd:
        report.warnings.append(f"TotalCharges: {n_odd} rows differ from tenure × MonthlyCharges by more than 2×")
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """The ONE cleaning function used by both training and the API (no training/serving skew)."""
    out = df.copy()
    out["TotalCharges"] = parse_total_charges(out["TotalCharges"])
    not_billed_yet = out["TotalCharges"].isna() & out["tenure"].eq(0)
    out.loc[not_billed_yet, "TotalCharges"] = 0.0
    out["SeniorCitizen"] = out["SeniorCitizen"].astype("int64")
    out["tenure"] = out["tenure"].astype("int64")
    out["MonthlyCharges"] = out["MonthlyCharges"].astype("float64")
    return out


def make_splits(df: pd.DataFrame, seed: int = SEED, val_size: float = 0.2, test_size: float = 0.2) -> dict[str, pd.DataFrame]:
    """Stratified train/validation/test split, done ONCE before any exploration or modelling."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), stratify=train_val[TARGET], random_state=seed)
    return {"train": train.reset_index(drop=True), "validation": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def save_splits(splits: dict[str, pd.DataFrame], directory: str | Path) -> dict[str, str]:
    """Write each split to CSV and return its SHA-256 (recorded in the model metadata)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, frame in splits.items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        hashes[name] = file_sha256(path)
    return hashes


def load_split(directory: str | Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / f"{name}.csv", dtype={ID_COLUMN: "str"})


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS], df[TARGET].eq("Yes").astype("int64").rename("churn")
