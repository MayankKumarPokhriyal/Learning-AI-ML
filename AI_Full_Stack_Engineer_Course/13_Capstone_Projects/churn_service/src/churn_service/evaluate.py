"""Metrics with bootstrap confidence intervals, calibration, explainability, slices, and the model card."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from churn_service.business import profit_per_1000
from churn_service.config import SEED, BusinessAssumptions


def metric_row(y, proba, monthly, threshold: float, assumptions: BusinessAssumptions) -> dict[str, float]:
    y, proba, monthly = np.asarray(y), np.asarray(proba), np.asarray(monthly)
    contact = proba >= threshold
    n_contact = contact.sum()
    true_pos = (contact & (y == 1)).sum()
    profit = profit_per_1000(y, contact, monthly, assumptions)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "average_precision": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "log_loss": log_loss(y, proba, labels=[0, 1]),
        "precision": true_pos / n_contact if n_contact else 0.0,
        "recall": true_pos / max((y == 1).sum(), 1),
        "contact_rate": contact.mean(),
        "profit_per_1000": profit,
        "profit_gain_vs_contact_all": profit - profit_per_1000(y, np.ones_like(y), monthly, assumptions),
        "profit_gain_vs_threshold_0.5": profit - profit_per_1000(y, proba >= 0.5, monthly, assumptions),
    }


def bootstrap_metrics(y, proba, monthly, threshold: float, assumptions: BusinessAssumptions,
                      n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05) -> pd.DataFrame:
    """Point estimates plus percentile bootstrap intervals (resample customers with replacement)."""
    y, proba, monthly = np.asarray(y), np.asarray(proba), np.asarray(monthly)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():  # AUC needs both classes
            continue
        rows.append(metric_row(y[idx], proba[idx], monthly[idx], threshold, assumptions))
    boot = pd.DataFrame(rows)
    return pd.DataFrame({
        "estimate": pd.Series(metric_row(y, proba, monthly, threshold, assumptions)),
        "ci_low": boot.quantile(alpha / 2),
        "ci_high": boot.quantile(1 - alpha / 2),
    })


def calibration_table(y, proba, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Equal-count bins: mean predicted probability vs observed churn rate, plus expected calibration error."""
    y, proba = np.asarray(y), np.asarray(proba)
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(proba, edges[1:-1], right=True), 0, len(edges) - 2)
    table = (pd.DataFrame({"bin": bins, "predicted": proba, "observed": y})
             .groupby("bin").agg(mean_predicted=("predicted", "mean"), observed_rate=("observed", "mean"), n=("observed", "size")))
    ece = float((table["n"] * (table["observed_rate"] - table["mean_predicted"]).abs()).sum() / len(y))
    return table, ece


def permutation_importance_table(model, X: pd.DataFrame, y, scoring: str = "roc_auc", n_repeats: int = 10, seed: int = SEED) -> pd.DataFrame:
    """Drop in score when one raw input column is shuffled (model-agnostic, computed on held-out data)."""
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed)
    return (pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def slice_metrics(y, proba, threshold: float, groups: pd.Series) -> pd.DataFrame:
    """Performance per group (e.g. gender, senior citizens) to spot uneven quality."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba), "group": np.asarray(groups)})
    rows = []
    for name, part in frame.groupby("group"):
        contact = part["p"] >= threshold
        positives = part["y"].sum()
        rows.append({
            groups.name or "group": name, "n": len(part), "churn_rate": part["y"].mean(),
            "roc_auc": roc_auc_score(part["y"], part["p"]) if part["y"].nunique() == 2 else np.nan,
            "recall": (contact & part["y"].eq(1)).sum() / positives if positives else np.nan,
            "contact_rate": contact.mean(),
        })
    return pd.DataFrame(rows)


def model_card_markdown(metadata: dict) -> str:
    """A short model card generated from the saved metadata, so every number in it is real."""
    a = metadata["business_assumptions"]
    tm = metadata["test_metrics"]

    def ci(name, fmt="{:.3f}"):
        m = tm[name]
        return f"{fmt.format(m['estimate'])} (95% CI {fmt.format(m['ci_low'])}–{fmt.format(m['ci_high'])})"

    splits = metadata["data"]["splits"]
    slices = "\n".join(
        f"| {row['slice']} | {row['n']} | {row['churn_rate']:.1%} | {row['roc_auc']:.3f} | {row['recall']:.1%} |"
        for row in metadata.get("test_slices", [])
    )
    return f"""# Model Card — {metadata['model_name']} {metadata['model_version']}

## Model details
- **Type:** {metadata['estimator']} inside a scikit-learn pipeline (feature engineering + preprocessing + model)
- **Trained:** {metadata['trained_at']} with scikit-learn {metadata['sklearn_version']}, pandas {metadata['pandas_version']}
- **Selection:** {metadata['selection_reason']}
- **Decision threshold:** contact a customer when P(churn) ≥ **{metadata['threshold']:.2f}** (maximizes expected profit on the validation split)

## Intended use
Rank existing telecom customers by churn risk so a retention team can decide who receives a retention offer.
**Not** for pricing, credit, or any decision that denies service. A human team owns the final contact decision.

## Data
IBM Telco Customer Churn sample ({metadata['data']['source_url']}), Apache-2.0. One snapshot of 7,043 fictional customers of one company.
Stratified splits — train {splits['train']['rows']:,} / validation {splits['validation']['rows']:,} / test {splits['test']['rows']:,} rows
(train SHA-256 `{splits['train']['sha256'][:16]}…`).

## Business assumptions (not from the data — replace with real numbers)
offer cost ${a['offer_cost']:.0f} per contacted customer · {a['acceptance_rate']:.0%} of contacted churners stay · {a['months_retained']} months of revenue kept

## Test-set results (used once)
- ROC-AUC {ci('roc_auc')} · average precision {ci('average_precision')} · Brier {ci('brier')}
- At the threshold: precision {ci('precision')}, recall {ci('recall')}, contacting {tm['contact_rate']['estimate']:.1%} of customers
- Expected profit per 1,000 customers: {ci('profit_per_1000', '${:,.0f}')}
- Gain vs contacting everyone: {ci('profit_gain_vs_contact_all', '${:,.0f}')} · vs threshold 0.5: {ci('profit_gain_vs_threshold_0.5', '${:,.0f}')}

| slice | n | churn rate | ROC-AUC | recall |
|---|---|---|---|---|
{slices}

## Limitations and risks
- Profit numbers are only as good as the assumptions above; the threshold must be re-tuned when they change.
- Single, old, fictional snapshot: no seasonality, no time-based validation, no causal effect of offers (uplift) measured.
- Probabilities are from one company's customers; re-validate before using elsewhere. Monitor drift with `reference_stats.json`.
"""
