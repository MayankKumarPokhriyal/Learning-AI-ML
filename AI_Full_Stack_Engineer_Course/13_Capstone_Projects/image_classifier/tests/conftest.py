"""Fixtures. API and preprocessing tests use a tiny exported model (seconds, no data or GPU);
the model-quality tests (marker `quality`) need the real trained artifacts and the EuroSAT data."""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from image_classifier import config as cfg
from image_classifier.export import save_bundle


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, len(cfg.CLASSES))

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).flatten(1))


@pytest.fixture(scope="session")
def tiny_artifacts(tmp_path_factory):
    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("tiny_artifacts")
    metadata = {"model_name": "tiny-test-model", "model_version": "0.0.0+tiny", "classes": list(cfg.CLASSES), "temperature": 1.5, "review_threshold": 0.5}
    save_bundle(TinyNet().eval(), metadata, directory)
    return directory


def image_bytes(fmt: str = "PNG", size: tuple[int, int] = (64, 64), mode: str = "RGB", color=(40, 120, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color if mode != "L" else 128).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def make_image():
    return image_bytes


@pytest.fixture(scope="session")
def real_artifacts():
    directory = cfg.artifacts_dir()
    if not (directory / "model.pt2").is_file():
        pytest.fail(f"no exported model in '{directory.name}/' — train first, or run `pytest -m 'not quality'`", pytrace=False)
    return directory
