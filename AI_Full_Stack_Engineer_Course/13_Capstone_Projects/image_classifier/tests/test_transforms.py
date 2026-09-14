import io

import numpy as np
import pytest
import torch
from PIL import Image

from image_classifier.transforms import InvalidImageError, augment_batch, decode_image_bytes, to_float, to_uint8_tensor


@pytest.mark.parametrize(("mode", "size"), [("RGB", (64, 64)), ("RGBA", (200, 100)), ("L", (31, 47))])
def test_any_image_becomes_uint8_3x64x64(make_image, mode, size):
    tensor = to_uint8_tensor(decode_image_bytes(make_image("PNG", size, mode)))
    assert tensor.shape == (3, 64, 64) and tensor.dtype == torch.uint8


def test_native_size_rgb_is_not_resampled():
    pixels = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")  # PNG is lossless
    tensor = to_uint8_tensor(decode_image_bytes(buffer.getvalue()))
    assert torch.equal(tensor, torch.from_numpy(pixels).permute(2, 0, 1))


def test_grayscale_is_repeated_across_channels(make_image):
    tensor = to_uint8_tensor(decode_image_bytes(make_image("PNG", (64, 64), "L")))
    assert torch.equal(tensor[0], tensor[1]) and torch.equal(tensor[1], tensor[2])


def test_invalid_inputs_raise_invalid_image_error(make_image):
    with pytest.raises(InvalidImageError, match="not a readable image"):
        decode_image_bytes(b"definitely not an image")
    with pytest.raises(InvalidImageError, match="not allowed"):
        decode_image_bytes(make_image("GIF"))
    with pytest.raises(InvalidImageError, match="corrupt or truncated"):
        decode_image_bytes(make_image("PNG", (256, 256))[:200])
    with pytest.raises(InvalidImageError, match="limit"):
        decode_image_bytes(make_image("PNG", (64, 64)), max_pixels=1000)


def test_to_float_scales_to_unit_range_and_rejects_floats():
    batch = torch.tensor([[[[0, 255]]]], dtype=torch.uint8)
    assert torch.equal(to_float(batch), torch.tensor([[[[0.0, 1.0]]]]))
    with pytest.raises(TypeError):
        to_float(batch.float())


def test_augmentation_only_moves_pixels_and_is_reproducible():
    x = torch.rand(16, 3, 64, 64)
    out = augment_batch(x, torch.Generator().manual_seed(0), brightness=0.0)
    assert out.shape == x.shape
    for original, augmented in zip(x, out, strict=True):  # flips/rotations permute pixels: same multiset of values
        assert torch.equal(original.flatten().sort().values, augmented.flatten().sort().values)
    assert not torch.equal(out, x)
    again = augment_batch(x, torch.Generator().manual_seed(0), brightness=0.0)
    assert torch.equal(out, again)
    bright = augment_batch(x, torch.Generator().manual_seed(1), brightness=0.1)
    assert bright.min() >= 0 and bright.max() <= 1
