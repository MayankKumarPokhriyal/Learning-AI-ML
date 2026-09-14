"""Evaluation: accuracy and macro-F1 with bootstrap CIs, per-class metrics, confusions, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from image_classifier.config import SEED


def _ci(values: np.ndarray, estimate: float, alpha: float = 0.05) -> dict[str, float]:
    return {"estimate": float(estimate), "ci_low": float(np.quantile(values, alpha / 2)), "ci_high": float(np.quantile(values, 1 - alpha / 2))}


def accuracy_with_ci(y_true, y_pred, n_boot: int = 1000, seed: int = SEED) -> dict[str, float]:
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(correct), size=(n_boot, len(correct)))
    return _ci(correct[idx].mean(axis=1), correct.mean())


def macro_f1_with_ci(y_true, y_pred, n_boot: int = 1000, seed: int = SEED) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(y_true)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
    return _ci(np.array(scores), f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_class_table(y_true, y_pred, classes) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    return pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=list(classes))


def top_confusions(y_true, y_pred, classes, k: int = 5) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    rows = [{"true": classes[i], "predicted": classes[j], "count": int(cm[i, j]), "share_of_true_class": cm[i, j] / cm[i].sum()}
            for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(k).reset_index(drop=True)


def expected_calibration_error(probs: np.ndarray, y_true, n_bins: int = 15) -> tuple[float, pd.DataFrame]:
    """ECE over equal-width confidence bins: Σ (bin size / N) · |accuracy − mean confidence|."""
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    confidence, correct = probs.max(axis=1), (probs.argmax(axis=1) == y_true)
    bins = np.clip(np.digitize(confidence, np.linspace(0, 1, n_bins + 1)[1:-1], right=True), 0, n_bins - 1)
    table = (pd.DataFrame({"bin": bins, "confidence": confidence, "correct": correct})
             .groupby("bin").agg(mean_confidence=("confidence", "mean"), accuracy=("correct", "mean"), n=("correct", "size")))
    ece = float((table["n"] * (table["accuracy"] - table["mean_confidence"]).abs()).sum() / len(y_true))
    return ece, table


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T > 0 minimizing NLL of softmax(logits / T) on VALIDATION data."""
    logits, labels = logits.detach().float().cpu(), labels.cpu()
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.detach().exp().item()


def coverage_table(probs: np.ndarray, y_true, thresholds=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99)) -> pd.DataFrame:
    """If we auto-accept predictions with top-1 confidence ≥ threshold (and send the rest to a human), how many are
    accepted (coverage) and how accurate are they?"""
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    confidence, correct = probs.max(axis=1), probs.argmax(axis=1) == y_true
    rows = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        rows.append({"threshold": float(threshold), "coverage": float(accepted.mean()),
                     "accuracy_accepted": float(correct[accepted].mean()) if accepted.any() else np.nan,
                     "errors_accepted": int((accepted & ~correct).sum())})
    return pd.DataFrame(rows)


def choose_review_threshold(probs: np.ndarray, y_true, target_accuracy: float = 0.99) -> float:
    """Lowest confidence cut-off whose auto-accepted predictions reach `target_accuracy`. Choose it on VALIDATION data."""
    table = coverage_table(probs, y_true, np.round(np.arange(0.30, 1.0, 0.01), 2))
    reached = table[table["accuracy_accepted"] >= target_accuracy]
    return float(reached["threshold"].iloc[0]) if len(reached) else 1.0


def summarize_test(logits: torch.Tensor, labels: torch.Tensor, temperature: float, n_boot: int = 1000, seed: int = SEED) -> dict:
    y, pred = labels.numpy(), logits.argmax(1).numpy()
    raw, scaled = torch.softmax(logits, 1), torch.softmax(logits / temperature, 1)
    return {
        "n": int(len(y)),
        "accuracy": accuracy_with_ci(y, pred, n_boot, seed),
        "macro_f1": macro_f1_with_ci(y, pred, n_boot, seed),
        "nll_raw": float(F.cross_entropy(logits, labels)),
        "nll_calibrated": float(F.cross_entropy(logits / temperature, labels)),
        "ece_raw": expected_calibration_error(raw.numpy(), y)[0],
        "ece_calibrated": expected_calibration_error(scaled.numpy(), y)[0],
    }
