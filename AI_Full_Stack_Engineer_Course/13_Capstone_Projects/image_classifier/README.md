# image-classifier — EuroSAT land-use classification, from training to a container

Classifies 64×64 Sentinel-2 satellite tiles into 10 land-use classes (forest, river, highway, residential, …). It fine-tunes an ImageNet-pretrained ResNet-18, calibrates its confidence, exports it with `torch.export`, and serves it through a FastAPI image-upload endpoint in a CPU-only Docker image.

It is the project folder of the course notebook [`02_End_to_End_DL_Project.ipynb`](../02_End_to_End_DL_Project.ipynb), which runs and explains every step.

## What's inside

| Step | Where |
|---|---|
| Download EuroSAT (via torchvision), stratified 70/15/15 split saved to `data/splits/*.csv`, fixed 200-image quality subset | `src/image_classifier/data.py` |
| One decoding/preprocessing path for training, tests, and the API; GPU batch augmentation (flips, 90° rotations, brightness) | `transforms.py` |
| Small CNN baseline and ResNet-18 transfer model (normalization + resizing inside the model) | `models.py` |
| Training loop: AdamW, one-cycle LR, lower LR for pretrained layers, early stopping on validation loss | `train.py` |
| Accuracy / macro-F1 with bootstrap CIs, per-class metrics, confusions, ECE, temperature scaling | `evaluate.py` |
| Grad-CAM heatmaps | `gradcam.py` |
| `torch.export` to `model.pt2`, metadata, latency measurement | `export.py` |
| Inference without the model classes or torchvision | `predict.py` |
| FastAPI upload service: type/size/pixel validation, top-k response, health/readiness | `api.py` |
| Preprocessing, API, and model-quality tests | `tests/` |
| Multi-stage, non-root, CPU-PyTorch image with a health check | `Dockerfile`, `.dockerignore` |
| Example GitHub Actions pipeline | `ci/github-actions.yml` |

## Quickstart

```bash
# 1. Environment — CPU PyTorch wheels on Linux; on macOS the default wheels include MPS support
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[train,dev]" --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

# 2. Data: downloads EuroSAT RGB (94 MB) into data/ and writes the splits
python -m image_classifier.data

# 3. Train + calibrate + evaluate once on test + export to artifacts/ (model.pt2, metadata.json).
#    A few minutes on an Apple Silicon GPU (MPS); much slower on CPU (try --epochs 2).
python -m image_classifier.train

# 4. Tests: everything (the quality gate needs steps 2–3) or only the fast ones
pytest
pytest -m "not quality"

# 5. Serve and call it
uvicorn image_classifier.api:app --reload
curl -s -F "file=@data/eurosat/2750/Forest/Forest_1.jpg;type=image/jpeg" "localhost:8000/predict?k=3"

# 6. Container
docker build -t image-classifier:1.0.0 .
docker run --rm -p 8000:8000 image-classifier:1.0.0
```

`data/` and `artifacts/` are git-ignored and recreated by steps 2–3. The model version is `1.0.0+<first 8 hex digits of the SHA-256 of the weights>`, so any retrain produces a new version.

`requirements.lock` pins the serving dependencies for the Docker image (the `.pt2` file should be loaded by the same PyTorch version that exported it). Regenerate:

```bash
uv pip compile pyproject.toml -o requirements.lock --python-version 3.12 --python-platform linux \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | readiness: model loaded and warmed up (503 otherwise) |
| GET | `/model` | version, classes, temperature, test metrics, request limits |
| POST | `/predict?k=3` | multipart upload `file` (JPEG/PNG, ≤ 5 MB, ≤ 25 megapixels) → top-k labels with calibrated probabilities |

Errors: **415** unsupported content type · **413** file too large · **400** undecodable image, or bytes that don't match the declared type · **422** missing file or `k` outside 1–10 · **503** model not loaded.

## Monitoring plan

- **Inputs:** image size and format mix, mean brightness per channel vs training data (seasonal changes, a new satellite, or clouds show up here).
- **Outputs:** predicted-class distribution and mean top-1 confidence vs the test set; a drop in confidence is the earliest sign of out-of-distribution tiles.
- **Quality:** send a small random sample to human labelers every week; alert when accuracy on it falls below the CI gate (0.90).
- **Service:** request rate, 4xx/5xx rates, p95 latency; the container health check covers liveness.

## Data and license

[EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019), RGB version: 27,000 labelled 64×64 tiles from Sentinel-2, 10 classes (2,000–3,000 per class), MIT license. Satellite images from nearby places can look alike, so a random split slightly overestimates accuracy on a new region; a region-based split is the stricter test.
