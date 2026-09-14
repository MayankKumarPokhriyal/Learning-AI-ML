import json

import pytest
from fastapi.testclient import TestClient

from churn_service.api import create_app
from churn_service.config import MAX_BATCH_SIZE
from churn_service.predict import ChurnModel
from churn_service.schemas import Customer


@pytest.fixture(scope="module")
def client(artifacts_dir):
    with TestClient(create_app(artifacts_dir)) as test_client:  # `with` runs the lifespan (model loading)
        yield test_client


@pytest.fixture(scope="module")
def metadata(artifacts_dir):
    return json.loads((artifacts_dir / "metadata.json").read_text())


def test_health_and_ready(client, metadata):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["model_version"] == metadata["model_version"]


def test_single_prediction_uses_saved_threshold_and_version(client, metadata, customer, artifacts_dir):
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == metadata["model_version"]
    assert body["threshold"] == metadata["threshold"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["contact"] == (body["churn_probability"] >= metadata["threshold"])
    direct = ChurnModel.load(artifacts_dir).predict_records([Customer(**customer).to_record()])[0]
    assert body["churn_probability"] == pytest.approx(direct["churn_probability"])


def test_batch_matches_single_predictions_in_order(client, customer):
    loyal = {**customer, "customer_id": "loyal", "tenure": 70, "contract": "Two year", "total_charges": 2100.0,
             "payment_method": "Credit card (automatic)"}
    new = {**customer, "customer_id": "new", "tenure": 0, "total_charges": None}
    response = client.post("/predict/batch", json={"customers": [customer, loyal, new]})
    assert response.status_code == 200
    body = response.json()
    assert [p["customer_id"] for p in body["predictions"]] == [customer["customer_id"], "loyal", "new"]
    assert body["n_customers"] == 3 and body["n_contact"] == sum(p["contact"] for p in body["predictions"])
    for single_input, batch_output in zip([customer, loyal, new], body["predictions"], strict=True):
        single = client.post("/predict", json=single_input).json()
        assert batch_output["churn_probability"] == pytest.approx(single["churn_probability"])


@pytest.mark.parametrize("change", [
    {"contract": "Three year"},                              # unknown category
    {"tenure": -1},                                          # out of range
    {"monthly_charges": "a lot"},                            # wrong type
    {"favourite_colour": "blue"},                            # unexpected field
    {"internet_service": "No"},                              # add-ons inconsistent with no internet
    {"total_charges": None},                                 # required because tenure > 0
])
def test_invalid_customers_are_rejected_with_422(client, customer, change):
    assert client.post("/predict", json={**customer, **change}).status_code == 422


def test_batch_size_limits(client, customer):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    assert client.post("/predict/batch", json={"customers": [customer] * (MAX_BATCH_SIZE + 1)}).status_code == 422


def test_service_without_artifacts_is_alive_but_not_ready(tmp_path, customer):
    with TestClient(create_app(tmp_path)) as broken:
        assert broken.get("/health").status_code == 200
        assert broken.get("/ready").status_code == 503
        assert broken.post("/predict", json=customer).status_code == 503
