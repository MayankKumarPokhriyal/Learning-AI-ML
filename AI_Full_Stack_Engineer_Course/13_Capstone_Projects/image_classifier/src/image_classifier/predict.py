"""Load the exported model bundle and predict top-k classes. Needs only torch, numpy, and Pillow."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from PIL import Image

from image_classifier.export import METADATA_FILE, MODEL_FILE
from image_classifier.transforms import to_float, to_uint8_tensor

logger = logging.getLogger("uvicorn.error")


class ImageClassifier:
    def __init__(self, module, metadata: dict, device: str = "cpu"):
        self.module, self.metadata, self.device = module, metadata, torch.device(device)

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> ImageClassifier:
        directory = Path(directory)
        for name in (MODEL_FILE, METADATA_FILE):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"{name} not found in the artifacts directory — train and export a model first")
        metadata = json.loads((directory / METADATA_FILE).read_text())
        trained_with = metadata.get("export", {}).get("torch_version", "").split("+")[0]
        if trained_with and trained_with != torch.__version__.split("+")[0]:
            logger.warning("model exported with torch %s but running %s", trained_with, torch.__version__)
        module = torch.export.load(directory / MODEL_FILE).module()  # no model class or torchvision needed
        if device != "cpu":
            module = module.to(device)
        return cls(module, metadata, device)

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def classes(self) -> list[str]:
        return list(self.metadata["classes"])

    @property
    def temperature(self) -> float:
        return float(self.metadata.get("temperature", 1.0))

    @property
    def review_threshold(self) -> float | None:
        """Top-1 probabilities below this value should go to a human (chosen on validation data)."""
        value = self.metadata.get("review_threshold")
        return None if value is None else float(value)

    def predict_proba(self, images_uint8: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        """uint8 (N, 3, 64, 64) → calibrated class probabilities (N, C) on the CPU."""
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images_uint8), batch_size):
                logits = self.module(to_float(images_uint8[start:start + batch_size]).to(self.device))
                outputs.append(torch.softmax(logits.float() / self.temperature, dim=1).cpu())
        return torch.cat(outputs)

    def predict_image(self, image: Image.Image, k: int = 3) -> list[dict]:
        probs = self.predict_proba(to_uint8_tensor(image).unsqueeze(0))[0]
        values, indices = probs.topk(min(k, len(self.classes)))
        return [{"label": self.classes[i], "probability": float(v)} for v, i in zip(values.tolist(), indices.tolist(), strict=True)]
