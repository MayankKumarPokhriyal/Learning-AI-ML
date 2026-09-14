"""EuroSAT download, stratified splits saved to CSV, a fixed quality subset, and in-memory image loading.

    python -m image_classifier.data [--data-dir data]      # download (94 MB) + write data/splits/*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from image_classifier import config as cfg
from image_classifier.export import file_sha256
from image_classifier.transforms import load_image_file


def download_eurosat(data_dir: str | Path) -> Path:
    """Download and extract the RGB EuroSAT archive via torchvision (skipped when already present)."""
    from torchvision.datasets import EuroSAT  # training-only dependency

    EuroSAT(root=str(data_dir), download=True)
    return cfg.image_root(data_dir)


def list_samples(image_root: str | Path) -> pd.DataFrame:
    root = Path(image_root)
    rows = [{"path": p.relative_to(root).as_posix(), "label": p.parent.name} for p in sorted(root.glob("*/*.jpg"))]
    samples = pd.DataFrame(rows)
    found = tuple(sorted(samples["label"].unique())) if len(samples) else ()
    if found != cfg.CLASSES:
        raise ValueError(f"expected class folders {cfg.CLASSES}, found {found}")
    samples["label_idx"] = samples["label"].map({name: i for i, name in enumerate(cfg.CLASSES)}).astype("int64")
    return samples


def make_splits(samples: pd.DataFrame, seed: int = cfg.SEED, val_size: float = 0.15, test_size: float = 0.15) -> dict[str, pd.DataFrame]:
    """Stratified train/validation/test split by class, made once and saved to disk."""
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(samples, test_size=test_size, stratify=samples["label"], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), stratify=train_val["label"], random_state=seed)
    return {name: frame.sort_values("path").reset_index(drop=True) for name, frame in [("train", train), ("validation", val), ("test", test)]}


def quality_subset(test: pd.DataFrame, per_class: int = 20, seed: int = cfg.SEED) -> pd.DataFrame:
    """A small, fixed, class-balanced slice of the test split for the CI model-quality gate."""
    return test.groupby("label").sample(n=per_class, random_state=seed).sort_values("path").reset_index(drop=True)


def save_splits(splits: dict[str, pd.DataFrame], directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, frame in splits.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
        hashes[name] = file_sha256(directory / f"{name}.csv")
    return hashes


def load_split(directory: str | Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(directory) / f"{name}.csv")


def load_images(samples: pd.DataFrame, image_root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode every listed image once into a uint8 tensor (N, 3, 64, 64): 27,000 tiles ≈ 330 MB."""
    root = Path(image_root)
    images = torch.stack([load_image_file(root / path) for path in samples["path"]])
    labels = torch.tensor(samples["label_idx"].to_numpy(copy=True), dtype=torch.long)  # pandas 3 arrays are read-only views
    return images, labels


def prepare_data(data_dir: str | Path, seed: int = cfg.SEED, per_class_quality: int = 20) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    root = download_eurosat(data_dir)
    splits = make_splits(list_samples(root), seed=seed)
    splits["quality_subset"] = quality_subset(splits["test"], per_class_quality, seed)
    return splits, save_splits(splits, Path(data_dir) / "splits")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=cfg.data_dir())
    args = parser.parse_args(argv)
    splits, hashes = prepare_data(args.data_dir)
    for name, frame in splits.items():
        print(f"{name:15s} {len(frame):6,d} images  sha256 {hashes[name][:12]}…")


if __name__ == "__main__":
    main()
