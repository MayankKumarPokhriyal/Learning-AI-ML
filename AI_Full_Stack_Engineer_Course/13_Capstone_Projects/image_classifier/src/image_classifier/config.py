"""Constants, paths, and request limits."""

from __future__ import annotations

import os
from pathlib import Path

CLASSES = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)
IMAGE_SIZE = 64          # EuroSAT RGB tiles are 64×64 pixels (Sentinel-2, 10 m per pixel)
SEED = 42
MODEL_NAME = "eurosat-landuse-classifier"
DATA_SOURCE = "https://github.com/phelber/EuroSAT"

# API limits: reject before decoding anything expensive
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000                   # 5000 × 5000; protects against "decompression bomb" images
ALLOWED_CONTENT_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}
MAX_TOP_K = len(CLASSES)

# src/image_classifier/config.py -> parents[2] is the project folder in a source checkout.
# Installed copies (e.g. Docker) set IMAGE_CLASSIFIER_ARTIFACTS_DIR / IMAGE_CLASSIFIER_DATA_DIR.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("IMAGE_CLASSIFIER_DATA_DIR", PROJECT_ROOT / "data"))


def artifacts_dir() -> Path:
    return Path(os.environ.get("IMAGE_CLASSIFIER_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))


def image_root(data_directory: str | Path) -> Path:
    """Where torchvision extracts the RGB JPEGs: <data>/eurosat/2750/<ClassName>/<ClassName>_<n>.jpg."""
    return Path(data_directory) / "eurosat" / "2750"
