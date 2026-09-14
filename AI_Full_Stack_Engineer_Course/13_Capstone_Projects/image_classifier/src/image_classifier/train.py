"""Train end to end: data → fine-tune ResNet-18 → temperature scaling → test once → export.

    python -m image_classifier.train [--data-dir data] [--artifacts-dir artifacts] [--epochs 6] [--max-train-images N]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR

from image_classifier import __version__
from image_classifier import config as cfg
from image_classifier.data import load_images, prepare_data
from image_classifier.evaluate import choose_review_threshold, fit_temperature, summarize_test
from image_classifier.export import save_bundle
from image_classifier.models import TransferResNet
from image_classifier.transforms import augment_batch

REVIEW_TARGET_ACCURACY = 0.99  # auto-accepted predictions must be at least this accurate; the rest go to a human


@dataclass
class TrainConfig:
    epochs: int = 6
    batch_size: int = 128
    max_lr: float = 1e-3
    body_lr_factor: float = 0.1       # pretrained layers learn 10× slower than the new head
    weight_decay: float = 1e-4
    patience: int = 2                 # early stopping: epochs without validation-loss improvement
    label_smoothing: float = 0.0
    augment: bool = True
    seed: int = cfg.SEED


@dataclass
class FitResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    seconds: float


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def param_groups(model: nn.Module, config: TrainConfig) -> list[dict]:
    if hasattr(model, "head_parameters"):
        return [{"params": model.body_parameters(), "lr": config.max_lr * config.body_lr_factor},
                {"params": model.head_parameters(), "lr": config.max_lr}]
    return [{"params": list(model.parameters()), "lr": config.max_lr}]


def predict_logits(model: nn.Module, images: torch.Tensor, device: torch.device, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            outputs.append(model(images[start:start + batch_size].to(device).float().div_(255)).float().cpu())
    return torch.cat(outputs)


def train_one_epoch(model, images, labels, device, optimizer, scheduler, config: TrainConfig, generator) -> tuple[float, float]:
    model.train()
    order = torch.randperm(len(images), generator=generator)
    total_loss, correct, seen = torch.zeros((), device=device), torch.zeros((), device=device), 0
    for start in range(0, len(order), config.batch_size):
        idx = order[start:start + config.batch_size]
        if len(idx) < 2:  # BatchNorm can't train on a batch of one
            continue
        x, y = images[idx].to(device).float().div_(255), labels[idx].to(device)
        if config.augment:
            x = augment_batch(x, generator)
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=config.label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.detach() * len(idx)       # stay on the device: no CPU sync per step
        correct += (logits.argmax(1) == y).sum()
        seen += len(idx)
    return total_loss.item() / seen, correct.item() / seen


def fit(model: nn.Module, train_images, train_labels, val_images, val_labels, config: TrainConfig, device: torch.device, log=print) -> FitResult:
    """AdamW + one-cycle LR schedule; keeps the weights of the epoch with the lowest validation loss."""
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(param_groups(model, config), weight_decay=config.weight_decay)
    steps = config.epochs * math.ceil(len(train_images) / config.batch_size)
    scheduler = OneCycleLR(optimizer, max_lr=[g["lr"] for g in optimizer.param_groups], total_steps=steps, pct_start=0.25)
    best_loss, best_state, best_epoch, bad_epochs, rows = math.inf, None, 0, 0, []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_images, train_labels, device, optimizer, scheduler, config, generator)
        val_logits = predict_logits(model, val_images, device)
        val_loss = F.cross_entropy(val_logits, val_labels).item()
        val_acc = (val_logits.argmax(1) == val_labels).float().mean().item()
        rows.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
                     "lr_head": optimizer.param_groups[-1]["lr"], "seconds": time.perf_counter() - epoch_start})
        improved = val_loss < best_loss - 1e-4
        log(f"epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{rows[-1]['seconds']:.1f} s{' ← best' if improved else ''}")
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                log(f"early stopping: no validation improvement for {config.patience} epochs")
                break
    model.load_state_dict(best_state)
    return FitResult(model, pd.DataFrame(rows), best_epoch, time.perf_counter() - start)


def weights_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_metadata(*, model: nn.Module, config: TrainConfig, result: FitResult, temperature: float, review_threshold: float,
                   split_hashes: dict, split_sizes: dict, test_metrics: dict, device: torch.device) -> dict:
    return {
        "model_name": cfg.MODEL_NAME,
        "model_version": f"{__version__}+{weights_fingerprint(model)[:8]}",
        "architecture": f"{type(model).__name__} (input upsampled to {getattr(model, 'input_size', cfg.IMAGE_SIZE)} px inside the model)",
        "classes": list(cfg.CLASSES),
        "image_size": cfg.IMAGE_SIZE,
        "temperature": float(temperature),
        "review_threshold": float(review_threshold),
        "review_threshold_selection": f"lowest top-1 confidence whose accepted validation predictions reach {REVIEW_TARGET_ACCURACY:.0%} accuracy",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "training_device": device.type,
        "data": {"source": cfg.DATA_SOURCE, "license": "MIT",
                 "splits": {name: {"rows": split_sizes[name], "sha256": split_hashes[name]} for name in split_hashes if name != "quality_subset"},
                 "quality_subset": {"rows": split_sizes["quality_subset"], "sha256": split_hashes["quality_subset"]}},
        "training": {**asdict(config), "best_epoch": result.best_epoch, "epochs_run": len(result.history), "seconds": round(result.seconds, 1),
                     "history": result.history.round(5).to_dict(orient="records")},
        "test_metrics": test_metrics,
    }


def run_training(data_dir: Path, artifacts_dir: Path, config: TrainConfig | None = None, max_train_images: int | None = None, log=print) -> dict:
    config = config or TrainConfig()
    device = pick_device()
    splits, split_hashes = prepare_data(data_dir, seed=config.seed)
    train_df = splits["train"]
    if max_train_images:
        train_df = train_df.sample(n=min(max_train_images, len(train_df)), random_state=config.seed)
    root = cfg.image_root(data_dir)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = (load_images(df, root) for df in (train_df, splits["validation"], splits["test"]))
    log(f"device {device.type} | train {len(train_x):,} | validation {len(val_x):,} | test {len(test_x):,} images")

    result = fit(TransferResNet(len(cfg.CLASSES)), train_x, train_y, val_x, val_y, config, device, log)
    val_logits = predict_logits(result.model, val_x, device)
    temperature = fit_temperature(val_logits, val_y)
    review_threshold = choose_review_threshold(torch.softmax(val_logits / temperature, 1).numpy(), val_y.numpy(), REVIEW_TARGET_ACCURACY)
    test_metrics = summarize_test(predict_logits(result.model, test_x, device), test_y, temperature)  # the only test evaluation
    sizes = {name: len(frame) for name, frame in splits.items()} | {"train": len(train_df)}
    metadata = build_metadata(model=result.model, config=config, result=result, temperature=temperature, review_threshold=review_threshold,
                              split_hashes=split_hashes, split_sizes=sizes, test_metrics=test_metrics, device=device)
    _, metadata = save_bundle(result.model, metadata, artifacts_dir)
    log(f"saved {metadata['model_version']} | test accuracy {test_metrics['accuracy']['estimate']:.4f} | temperature {temperature:.3f}")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=cfg.data_dir())
    parser.add_argument("--artifacts-dir", type=Path, default=cfg.artifacts_dir())
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--max-train-images", type=int, default=None, help="train on a random subset (quick smoke runs)")
    args = parser.parse_args(argv)
    run_training(args.data_dir, args.artifacts_dir, TrainConfig(epochs=args.epochs), args.max_train_images)


if __name__ == "__main__":
    main()
