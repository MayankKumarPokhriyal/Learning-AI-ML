"""Decoding, preprocessing, and augmentation. Training, tests, and the API all use these functions (no skew)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from image_classifier.config import IMAGE_SIZE, MAX_IMAGE_PIXELS


class InvalidImageError(ValueError):
    """The bytes are not a decodable image of an allowed format within the size limits."""


def decode_image_bytes(data: bytes, allowed_formats: tuple[str, ...] = ("JPEG", "PNG"), max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    """Validate format and pixel count from the header first, then decode fully (so corrupt files fail here)."""
    try:
        image = Image.open(io.BytesIO(data))  # lazy: reads only the header
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
        raise InvalidImageError("not a readable image") from err
    if image.format not in allowed_formats:
        raise InvalidImageError(f"format {image.format} is not allowed (allowed: {', '.join(allowed_formats)})")
    width, height = image.size
    if width * height > max_pixels:
        raise InvalidImageError(f"image is {width}×{height} pixels; the limit is {max_pixels:,} pixels")
    try:
        image.load()
    except (OSError, Image.DecompressionBombError) as err:
        raise InvalidImageError("image data is corrupt or truncated") from err
    return image


def to_uint8_tensor(image: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Any PIL image (RGBA, grayscale, any size) → uint8 tensor of shape (3, size, size)."""
    rgb = image.convert("RGB")
    if rgb.size != (size, size):
        rgb = rgb.resize((size, size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(rgb, dtype=np.uint8)).permute(2, 0, 1).contiguous()


def load_image_file(path: str | Path, size: int = IMAGE_SIZE) -> torch.Tensor:
    with Image.open(path) as image:
        return to_uint8_tensor(image, size)


def to_float(batch: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] → float32 [0, 1]. Normalization happens inside the model."""
    if batch.dtype != torch.uint8:
        raise TypeError(f"expected a uint8 tensor, got {batch.dtype}")
    return batch.float().div_(255)


def augment_batch(x: torch.Tensor, generator: torch.Generator | None = None, brightness: float = 0.1) -> torch.Tensor:
    """Per-image random horizontal/vertical flips, transpose (→ all 8 flips/90° rotations), and brightness jitter.

    Overhead imagery has no "up", so every rotation is a valid view of the same land. Random numbers are drawn on
    the CPU generator (reproducible), and the ops run on the batch's device (fast on a GPU).
    """
    n = x.shape[0]

    def coin() -> torch.Tensor:
        return (torch.rand(n, generator=generator) < 0.5).to(x.device).view(n, 1, 1, 1)

    x = torch.where(coin(), x.flip(-1), x)
    x = torch.where(coin(), x.flip(-2), x)
    x = torch.where(coin(), x.transpose(-1, -2), x)
    if brightness > 0:
        factor = 1 + (torch.rand(n, generator=generator) * 2 - 1) * brightness
        x = (x * factor.to(x.device).view(n, 1, 1, 1)).clamp(0, 1)
    return x
