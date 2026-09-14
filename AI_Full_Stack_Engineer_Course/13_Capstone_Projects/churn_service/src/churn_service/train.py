"""End-to-end training: download → validate → split → compare models → threshold → test once → save.

    python -m churn_service.train [--data-dir data] [--artifacts-dir artifacts]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn_service import __version__
from churn_service import config as cfg
from churn_service.business import best_threshold, profit_curve
from churn_service.data import FEATURE_COLUMNS, clean, download_raw, file_sha256, load_raw, make_splits, save_splits, split_features_target, validate_raw
from churn_service.evaluate import bootstrap_metrics, metric_row, slice_metrics
from churn_service.features import CATEGORICAL_FEATURES, ENGINEERED_FEATURES, FeatureAdder, build_preprocessor
from churn_service.monitoring import reference_statistics
from churn_service.predict import save_artifacts

SIMPLICITY_ORDER = ("logistic_regression", "gradient_boosting")
CV_SCORING = {"roc_auc": "roc_auc", "average_precision": "average_precision", "brier": "neg_brier_score", "log_loss": "neg_log_loss"}


def candidate_models(seed: int = cfg.SEED) -> dict[str, Pipeline]:
    booster = HistGradientBoostingClassifier(
        categorical_features=CATEGORICAL_FEATURES, learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed,
    )
    return {
        "logistic_regression": Pipeline([("features", FeatureAdder()), ("preprocess", build_preprocessor("linear")),
                                         ("model", LogisticRegression(max_iter=2000))]),
        "gradient_boosting": Pipeline([("features", FeatureAdder()), ("preprocess", build_preprocessor("tree")), ("model", booster)]),
    }


def cross_validate_models(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, seed: int = cfg.SEED,
                          n_splits: int = 5) -> pd.DataFrame:
    """5-fold stratified CV on the TRAINING split only. Brier and log loss are reported as positive losses."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        result = cross_validate(model, X, y, cv=cv, scoring=CV_SCORING)
        row = {"model": name, "n_folds": n_splits}
        for metric in CV_SCORING:
            scores = result[f"test_{metric}"] * (-1 if CV_SCORING[metric].startswith("neg_") else 1)
            row[f"{metric}_mean"], row[f"{metric}_std"] = float(scores.mean()), float(scores.std())
        row["fit_seconds"] = float(result["fit_time"].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def select_model(cv_table: pd.DataFrame, metric: str = "roc_auc") -> tuple[str, str]:
    """One-standard-error rule: the simplest model within 1 SE of the best mean score wins."""
    best = cv_table[f"{metric}_mean"].idxmax()
    best_mean = cv_table.loc[best, f"{metric}_mean"]
    standard_error = cv_table.loc[best, f"{metric}_std"] / np.sqrt(cv_table.loc[best, "n_folds"])
    for name in SIMPLICITY_ORDER:
        if name in cv_table.index and cv_table.loc[name, f"{metric}_mean"] >= best_mean - standard_error:
            gap = best_mean - cv_table.loc[name, f"{metric}_mean"]
            reason = (f"{name} has the best mean CV {metric} ({best_mean:.4f})" if name == best else
                      f"{name} is within one standard error ({standard_error:.4f}) of {best} (gap {gap:.4f}) and is simpler")
            return name, reason
    return best, f"{best} has the best mean CV {metric}"


def build_metadata(*, model_type: str, pipeline: Pipeline, threshold: float, assumptions: cfg.BusinessAssumptions,
                   raw_sha256: str, split_hashes: dict, splits: dict, cv_table: pd.DataFrame, selection_reason: str,
                   validation_metrics: dict, test_metrics: pd.DataFrame, test_slices: list[dict]) -> dict:
    fingerprint_source = json.dumps({"train_sha256": split_hashes["train"], "model_type": model_type,
                                     "params": {k: repr(v) for k, v in pipeline["model"].get_params().items()},
                                     "threshold": round(threshold, 4), "assumptions": asdict(assumptions)}, sort_keys=True)
    fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
    return {
        "model_name": cfg.MODEL_NAME,
        "model_version": f"{__version__}+{fingerprint[:8]}",
        "model_type": model_type,
        "estimator": type(pipeline["model"]).__name__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "data": {"source_url": cfg.DATA_URL, "raw_sha256": raw_sha256,
                 "splits": {name: {"rows": len(frame), "sha256": split_hashes[name],
                                   "churn_rate": float(frame[cfg.TARGET].eq("Yes").mean())} for name, frame in splits.items()}},
        "features": {"input_columns": FEATURE_COLUMNS, "engineered": ENGINEERED_FEATURES},
        "business_assumptions": asdict(assumptions),
        "threshold": float(threshold),
        "threshold_selection": "maximizes expected profit per 1,000 customers on the validation split",
        "selection_reason": selection_reason,
        "cross_validation": cv_table.reset_index().to_dict(orient="records"),
        "validation_metrics": {k: float(v) for k, v in validation_metrics.items()},
        "test_metrics": {name: {k: float(v) for k, v in row.items()} for name, row in test_metrics.iterrows()},
        "test_slices": test_slices,
    }


def evaluation_slices(y, proba, threshold: float, X: pd.DataFrame) -> list[dict]:
    records = []
    for column, label in [("gender", "gender"), ("SeniorCitizen", "senior citizen"), ("Contract", "contract")]:
        table = slice_metrics(y, proba, threshold, X[column])
        for row in table.to_dict(orient="records"):
            records.append({"slice": f"{label} = {row[column]}", "n": int(row["n"]), "churn_rate": float(row["churn_rate"]),
                            "roc_auc": float(row["roc_auc"]), "recall": float(row["recall"])})
    return records


def run_training(data_dir: Path, artifacts_dir: Path, assumptions: cfg.BusinessAssumptions | None = None,
                 seed: int = cfg.SEED, n_boot: int = 1000, log=print) -> dict:
    assumptions = assumptions or cfg.BusinessAssumptions()
    start = time.perf_counter()
    raw_path = download_raw(Path(data_dir) / "raw" / "telco_customer_churn.csv")
    raw = load_raw(raw_path)
    report = validate_raw(raw)
    log(str(report))
    report.raise_if_failed()

    splits = make_splits(clean(raw), seed=seed)
    split_hashes = save_splits(splits, Path(data_dir) / "splits")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = (split_features_target(splits[s]) for s in ("train", "validation", "test"))

    models = candidate_models(seed)
    cv_table = cross_validate_models(models, X_train, y_train, seed=seed)
    model_type, reason = select_model(cv_table)
    log(f"model selection: {reason}")
    pipeline = clone(models[model_type]).fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = best_threshold(profit_curve(y_val, val_proba, X_val["MonthlyCharges"], assumptions))
    validation_metrics = metric_row(y_val, val_proba, X_val["MonthlyCharges"], threshold, assumptions)
    log(f"threshold chosen on validation: {threshold:.2f}")

    test_proba = pipeline.predict_proba(X_test)[:, 1]  # the only time the test split is scored
    test_metrics = bootstrap_metrics(y_test, test_proba, X_test["MonthlyCharges"], threshold, assumptions, n_boot=n_boot, seed=seed)
    slices = evaluation_slices(y_test, test_proba, threshold, X_test)

    metadata = build_metadata(model_type=model_type, pipeline=pipeline, threshold=threshold, assumptions=assumptions,
                              raw_sha256=file_sha256(raw_path), split_hashes=split_hashes, splits=splits, cv_table=cv_table,
                              selection_reason=reason, validation_metrics=validation_metrics, test_metrics=test_metrics,
                              test_slices=slices)
    reference = reference_statistics(X_train, pipeline.predict_proba(X_train)[:, 1])
    save_artifacts(pipeline, metadata, reference, artifacts_dir)
    log(f"saved model {metadata['model_version']} | test ROC-AUC {test_metrics.loc['roc_auc', 'estimate']:.3f} | "
        f"{time.perf_counter() - start:.1f} s")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=cfg.data_dir())
    parser.add_argument("--artifacts-dir", type=Path, default=cfg.artifacts_dir())
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    run_training(args.data_dir, args.artifacts_dir, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
