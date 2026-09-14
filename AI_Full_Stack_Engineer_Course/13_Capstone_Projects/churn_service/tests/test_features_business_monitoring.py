import numpy as np
import pandas as pd
import pytest

from churn_service.business import best_threshold, break_even_probability, expected_profit, profit_curve
from churn_service.config import BusinessAssumptions
from churn_service.data import clean, split_features_target
from churn_service.features import ENGINEERED_FEATURES, FeatureAdder, add_features, build_preprocessor
from churn_service.monitoring import drift_report, population_stability_index, reference_statistics


def test_add_features_counts_services_and_handles_new_customers(raw_frame):
    X, _ = split_features_target(clean(raw_frame))
    out = add_features(X)
    assert out["n_services"].tolist() == [1, 3, 4, 1]  # row 3: phone, multiple lines, streaming TV, streaming movies
    assert out.loc[3, "charges_ratio"] == 1.0  # tenure 0 → no division by zero
    assert out.loc[1, "charges_ratio"] == pytest.approx(1889.5 / (34 * 56.95))
    assert list(X.columns) + ENGINEERED_FEATURES == list(out.columns)
    assert not out[ENGINEERED_FEATURES].isna().any().any()


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_preprocessors_accept_unseen_categories(raw_frame, kind):
    X, _ = split_features_target(clean(raw_frame))
    pre = build_preprocessor(kind).fit(FeatureAdder().fit_transform(X))
    unseen = X.copy()
    unseen.loc[0, "PaymentMethod"] = "Crypto"
    transformed = pre.transform(FeatureAdder().fit(X).transform(unseen))
    assert np.asarray(transformed).shape[0] == len(X)


def test_expected_profit_matches_hand_calculation():
    a = BusinessAssumptions(offer_cost=10, acceptance_rate=0.5, months_retained=2)
    y = [1, 0, 1, 0]
    contact = [1, 1, 0, 0]
    monthly = [100, 50, 100, 50]
    # contacted churner: 0.5 × 100 × 2 − 10 = 90; contacted non-churner: −10; others: 0
    assert expected_profit(y, contact, monthly, a) == pytest.approx(80.0)
    assert break_even_probability(100, a) == pytest.approx(0.1)


def test_best_threshold_finds_the_profitable_cutoff():
    a = BusinessAssumptions(offer_cost=10, acceptance_rate=0.5, months_retained=2)
    y = np.array([1, 1, 0, 0, 0])
    proba = np.array([0.9, 0.6, 0.55, 0.2, 0.1])
    curve = profit_curve(y, proba, np.full(5, 100.0), a)
    assert best_threshold(curve) == pytest.approx(0.6)  # contacts exactly the two churners


def test_psi_is_zero_for_identical_and_large_for_shifted_data():
    rng = np.random.default_rng(0)
    reference_frame = pd.DataFrame({"x": rng.normal(size=2000), "c": rng.choice(["a", "b"], size=2000)})
    ref = reference_statistics(reference_frame, rng.uniform(size=2000))
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == 0.0
    same = drift_report(ref, reference_frame)
    assert (same["psi"] < 1e-9).all()
    shifted = pd.DataFrame({"x": rng.normal(loc=1.5, size=2000), "c": rng.choice(["a", "b", "new"], size=2000)})
    report = drift_report(ref, shifted).set_index("feature")
    assert report.loc["x", "status"] == "drift" and report.loc["c", "status"] == "drift"
