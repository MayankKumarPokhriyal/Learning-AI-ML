"""Quality gate: CI fails if a retrained model is worse than these thresholds on the frozen test split."""

import json

import pytest
from sklearn.metrics import roc_auc_score

from churn_service import config
from churn_service.business import profit_per_1000
from churn_service.config import BusinessAssumptions
from churn_service.data import file_sha256, load_split, split_features_target
from churn_service.predict import ChurnModel

MIN_TEST_ROC_AUC = 0.80  # a logistic-regression baseline scores ≈0.84 on this data; below 0.80 something broke


@pytest.fixture(scope="module")
def scored_test_split(artifacts_dir):
    splits_dir = config.data_dir() / "splits"
    if not (splits_dir / "test.csv").is_file():
        pytest.fail("data/splits/test.csv missing — run `python -m churn_service.train` first", pytrace=False)
    model = ChurnModel.load(artifacts_dir)
    X, y = split_features_target(load_split(splits_dir, "test"))
    return model, X, y, model.predict_proba(X), file_sha256(splits_dir / "test.csv")


def test_test_split_is_the_one_the_model_was_evaluated_on(scored_test_split, artifacts_dir):
    *_, sha = scored_test_split
    metadata = json.loads((artifacts_dir / "metadata.json").read_text())
    assert sha == metadata["data"]["splits"]["test"]["sha256"]


def test_roc_auc_above_minimum(scored_test_split):
    model, _, y, proba, _ = scored_test_split
    auc = roc_auc_score(y, proba)
    assert auc >= MIN_TEST_ROC_AUC, f"test ROC-AUC {auc:.3f} < {MIN_TEST_ROC_AUC}"
    assert auc == pytest.approx(model.metadata["test_metrics"]["roc_auc"]["estimate"])  # reproducible scoring


def test_threshold_policy_beats_simple_policies(scored_test_split):
    model, X, y, proba, _ = scored_test_split
    assumptions = BusinessAssumptions(**model.metadata["business_assumptions"])
    model_profit = profit_per_1000(y, proba >= model.threshold, X["MonthlyCharges"], assumptions)
    everyone = profit_per_1000(y, [1] * len(y), X["MonthlyCharges"], assumptions)
    assert model_profit > max(everyone, 0.0), f"model ${model_profit:,.0f} vs contact-all ${everyone:,.0f} vs nobody $0"
