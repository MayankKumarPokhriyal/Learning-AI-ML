import pytest
from fastapi.testclient import TestClient

from image_classifier import config as cfg
from image_classifier.api import create_app
from image_classifier.predict import ImageClassifier
from image_classifier.transforms import decode_image_bytes


@pytest.fixture(scope="module")
def client(tiny_artifacts):
    with TestClient(create_app(tiny_artifacts)) as test_client:
        yield test_client


def upload(client, data, content_type="image/png", k=None, filename="tile.png"):
    params = {"k": k} if k is not None else {}
    return client.post("/predict", files={"file": (filename, data, content_type)}, params=params)


def test_health_ready_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready", "model_version": "0.0.0+tiny"}
    info = client.get("/model").json()
    assert info["classes"] == list(cfg.CLASSES) and info["image_size"] == 64


@pytest.mark.parametrize(("fmt", "content_type"), [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_prediction_returns_sorted_top_k(client, make_image, fmt, content_type):
    response = upload(client, make_image(fmt, (64, 64)), content_type)
    assert response.status_code == 200
    body = response.json()
    probabilities = [item["probability"] for item in body["top_k"]]
    assert len(body["top_k"]) == 3 and probabilities == sorted(probabilities, reverse=True)
    assert {item["label"] for item in body["top_k"]} <= set(cfg.CLASSES)
    assert body["model_version"] == "0.0.0+tiny" and body["filename"] == "tile.png"
    assert body["needs_review"] == (probabilities[0] < 0.5)  # the tiny model is unsure, so this also exercises True


def test_all_classes_sum_to_one_and_match_offline_prediction(client, make_image, tiny_artifacts):
    data = make_image("PNG", (120, 80), "RGBA", (200, 30, 30, 255))
    body = upload(client, data, k=len(cfg.CLASSES)).json()
    assert sum(item["probability"] for item in body["top_k"]) == pytest.approx(1.0, abs=1e-5)
    offline = ImageClassifier.load(tiny_artifacts).predict_image(decode_image_bytes(data), k=len(cfg.CLASSES))
    assert [item["label"] for item in body["top_k"]] == [item["label"] for item in offline]
    assert [item["probability"] for item in body["top_k"]] == pytest.approx([item["probability"] for item in offline])


@pytest.mark.parametrize("k", [0, len(cfg.CLASSES) + 1])
def test_k_out_of_range_is_422(client, make_image, k):
    assert upload(client, make_image(), k=k).status_code == 422


def test_rejections(client, make_image):
    assert upload(client, b"hello", "text/plain").status_code == 415                           # wrong media type
    assert upload(client, b"x" * (cfg.MAX_UPLOAD_BYTES + 1)).status_code == 413                 # too large
    assert upload(client, b"not really a png").status_code == 400                               # undecodable
    assert upload(client, make_image("JPEG"), "image/png").status_code == 400                   # content type lies
    assert client.post("/predict").status_code == 422                                          # no file at all


def test_service_without_artifacts_is_alive_but_not_ready(tmp_path, make_image):
    with TestClient(create_app(tmp_path)) as broken:
        assert broken.get("/health").status_code == 200
        assert broken.get("/ready").status_code == 503
        assert upload(broken, make_image()).status_code == 503
