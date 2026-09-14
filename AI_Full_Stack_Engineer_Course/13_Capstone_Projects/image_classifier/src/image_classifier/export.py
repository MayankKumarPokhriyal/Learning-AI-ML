"""Export with torch.export, save the model bundle, and measure latency."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from image_classifier.config import IMAGE_SIZE

MODEL_FILE, METADATA_FILE = "model.pt2", "metadata.json"
MAX_EXPORT_BATCH = 1024


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(model: nn.Module, image_size: int = IMAGE_SIZE, max_batch: int = MAX_EXPORT_BATCH) -> torch.export.ExportedProgram:
    """Trace an eval-mode CPU copy into a standalone graph; the batch dimension stays dynamic."""
    cpu_model = copy.deepcopy(model).cpu().eval()
    batch = torch.export.Dim("batch", min=1, max=max_batch)
    example = torch.rand(2, 3, image_size, image_size)
    return torch.export.export(cpu_model, (example,), dynamic_shapes={"x": {0: batch}})


def save_bundle(model: nn.Module, metadata: dict, directory: str | Path) -> tuple[dict[str, Path], dict]:
    """Write model.pt2 (loadable without this package's model classes) and metadata.json next to it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"model": directory / MODEL_FILE, "metadata": directory / METADATA_FILE}
    torch.export.save(export_model(model), paths["model"])
    metadata = {**metadata, "export": {"format": "torch.export ExportedProgram (.pt2)", "dynamic_batch_max": MAX_EXPORT_BATCH,
                                       "input": f"float32 RGB in [0, 1], shape (batch, 3, {IMAGE_SIZE}, {IMAGE_SIZE})",
                                       "torch_version": torch.__version__, "file_sha256": file_sha256(paths["model"])}}
    paths["metadata"].write_text(json.dumps(metadata, indent=2))
    return paths, metadata


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency(fn, batch: torch.Tensor, device: torch.device, repeats: int = 30, warmup: int = 5) -> dict[str, float]:
    """p50/p95 milliseconds per call and images per second; GPU work is synchronized before reading the clock."""
    batch = batch.to(device)
    times = []
    with torch.inference_mode():
        for i in range(warmup + repeats):
            start = time.perf_counter()
            fn(batch)
            synchronize(device)
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1000)
    times_ms = np.array(times)
    return {"p50_ms": float(np.percentile(times_ms, 50)), "p95_ms": float(np.percentile(times_ms, 95)),
            "images_per_s": float(len(batch) / (times_ms.mean() / 1000))}
