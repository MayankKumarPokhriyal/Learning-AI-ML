"""Quality gate on a fixed, class-balanced subset of the test split (200 images, 20 per class).

Runs the exported .pt2 file — exactly what the service ships. Fails (never skips) when artifacts or data are missing.
"""

import json

import pytest

from image_classifier import config as cfg
from image_classifier.data import load_images, load_split
from image_classifier.export import file_sha256
from image_classifier.predict import ImageClassifier

pytestmark = pytest.mark.quality

MIN_ACCURACY = 0.90        # a small CNN trained from scratch for a few epochs already passes ~0.8; fine-tuned ResNet-18 should be far above
MIN_CLASS_RECALL = 0.70    # no class may collapse, even if the average looks fine


@pytest.fixture(scope="module")
def quality_run(real_artifacts):
    splits_dir, root = cfg.data_dir() / "splits", cfg.image_root(cfg.data_dir())
    if not (splits_dir / "quality_subset.csv").is_file() or not root.is_dir():
        pytest.fail("EuroSAT data or splits missing — run `python -m image_classifier.data` first", pytrace=False)
    subset = load_split(splits_dir, "quality_subset")
    images, labels = load_images(subset, root)
    classifier = ImageClassifier.load(real_artifacts)
    predictions = classifier.predict_proba(images).argmax(1)
    return classifier, subset, labels, predictions, file_sha256(splits_dir / "quality_subset.csv")


def test_subset_is_the_one_recorded_in_metadata(quality_run, real_artifacts):
    *_, sha = quality_run
    metadata = json.loads((real_artifacts / "metadata.json").read_text())
    assert sha == metadata["data"]["quality_subset"]["sha256"]


def test_accuracy_above_minimum(quality_run):
    _, _, labels, predictions, _ = quality_run
    accuracy = (predictions == labels).float().mean().item()
    assert accuracy >= MIN_ACCURACY, f"accuracy {accuracy:.3f} < {MIN_ACCURACY}"


def test_every_class_recall_above_floor(quality_run):
    classifier, _, labels, predictions, _ = quality_run
    for index, name in enumerate(classifier.classes):
        mask = labels == index
        recall = (predictions[mask] == index).float().mean().item()
        assert recall >= MIN_CLASS_RECALL, f"{name} recall {recall:.2f} < {MIN_CLASS_RECALL}"
