"""Reference statistics saved at training time, and a PSI-based drift report for production data."""

from __future__ import annotations

import numpy as np
import pandas as pd

OTHER = "__other__"
PSI_WATCH, PSI_DRIFT = 0.1, 0.25  # common rules of thumb: < 0.1 stable, 0.1–0.25 watch, > 0.25 investigate


def _numeric_bins(values: np.ndarray, n_bins: int) -> list[float]:
    inner = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return [float(v) for v in np.unique(inner)]


def _proportions_numeric(values, inner_edges) -> list[float]:
    counts = np.bincount(np.digitize(np.asarray(values, dtype=float), inner_edges, right=True), minlength=len(inner_edges) + 1)
    return (counts / counts.sum()).tolist()


def reference_statistics(X: pd.DataFrame, proba, n_bins: int = 10) -> dict:
    """Bins and proportions for every input column and for the model's scores (JSON-serializable)."""
    features = {}
    for column in X.columns:
        values = X[column]
        if pd.api.types.is_numeric_dtype(values):
            edges = _numeric_bins(values.to_numpy(dtype=float), n_bins)
            features[column] = {"kind": "numeric", "inner_edges": edges, "proportions": _proportions_numeric(values, edges),
                                "mean": float(values.mean()), "std": float(values.std())}
        else:
            shares = values.astype("string").value_counts(normalize=True)
            features[column] = {"kind": "categorical", "proportions": {str(k): float(v) for k, v in shares.items()}}
    proba = np.asarray(proba, dtype=float)
    edges = _numeric_bins(proba, n_bins)
    return {"n_rows": len(X), "features": features,
            "prediction": {"inner_edges": edges, "proportions": _proportions_numeric(proba, edges), "mean": float(proba.mean())}}


def population_stability_index(expected, actual, eps: float = 1e-4) -> float:
    """PSI = Σ (actual − expected) · ln(actual / expected) over bins; 0 means identical distributions."""
    expected = np.clip(np.asarray(expected, dtype=float), eps, None)
    actual = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(reference: dict, X: pd.DataFrame, proba=None) -> pd.DataFrame:
    rows = []
    for column, ref in reference["features"].items():
        if ref["kind"] == "numeric":
            psi = population_stability_index(ref["proportions"], _proportions_numeric(X[column], ref["inner_edges"]))
        else:
            current = X[column].astype("string").value_counts(normalize=True)
            categories = list(ref["proportions"])
            unseen = float(current[~current.index.isin(categories)].sum())
            psi = population_stability_index([*ref["proportions"].values(), 0.0],
                                              [float(current.get(c, 0.0)) for c in categories] + [unseen])
        rows.append({"feature": column, "kind": ref["kind"], "psi": psi})
    if proba is not None:
        pred = reference["prediction"]
        rows.append({"feature": "prediction_score", "kind": "model output",
                     "psi": population_stability_index(pred["proportions"], _proportions_numeric(proba, pred["inner_edges"]))})
    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    report["status"] = np.select([report["psi"] >= PSI_DRIFT, report["psi"] >= PSI_WATCH], ["drift", "watch"], "ok")
    return report
