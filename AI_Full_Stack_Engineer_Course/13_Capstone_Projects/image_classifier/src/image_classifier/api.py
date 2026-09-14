"""FastAPI image-upload service: `uvicorn image_classifier.api:app --host 0.0.0.0 --port 8000`."""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from image_classifier import __version__
from image_classifier import config as cfg
from image_classifier.predict import ImageClassifier
from image_classifier.transforms import InvalidImageError, decode_image_bytes

logger = logging.getLogger("uvicorn.error")


class LabelProbability(BaseModel):
    label: str
    probability: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    model_version: str
    filename: str | None
    top_k: list[LabelProbability]
    needs_review: bool = Field(description="top-1 probability is below the model's review threshold: ask a human")
    inference_ms: float


def get_classifier(request: Request) -> ImageClassifier:
    classifier = request.app.state.classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {request.app.state.load_error}")
    return classifier


ClassifierDep = Annotated[ImageClassifier, Depends(get_classifier)]


def create_app(artifacts_dir: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        directory = Path(artifacts_dir) if artifacts_dir is not None else cfg.artifacts_dir()
        app.state.classifier, app.state.load_error = None, None
        try:
            classifier = ImageClassifier.load(directory, device=os.environ.get("IMAGE_CLASSIFIER_DEVICE", "cpu"))
            classifier.predict_proba(torch.zeros(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, dtype=torch.uint8))  # warm-up
            app.state.classifier = classifier
            logger.info("loaded model %s with %d classes", classifier.version, len(classifier.classes))
        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as err:
            app.state.load_error = str(err)
            logger.error("model not loaded: %s", err)
        yield
        app.state.classifier = None

    app = FastAPI(title="EuroSAT Land-Use Classifier", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        classifier = request.app.state.classifier
        if classifier is None:
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": request.app.state.load_error})
        return {"status": "ready", "model_version": classifier.version}

    @app.get("/model")
    def model_info(classifier: ClassifierDep) -> dict:
        meta = classifier.metadata
        return {"model_version": classifier.version, "classes": classifier.classes, "image_size": cfg.IMAGE_SIZE,
                "temperature": classifier.temperature, "review_threshold": classifier.review_threshold, "test_metrics": meta.get("test_metrics", {}),
                "limits": {"max_upload_bytes": cfg.MAX_UPLOAD_BYTES, "max_image_pixels": cfg.MAX_IMAGE_PIXELS,
                           "content_types": sorted(cfg.ALLOWED_CONTENT_TYPES)}}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        classifier: ClassifierDep,
        file: Annotated[UploadFile, File(description="a JPEG or PNG image (EuroSAT tiles are 64×64 RGB)")],
        k: Annotated[int, Query(ge=1, le=cfg.MAX_TOP_K, description="number of classes to return")] = 3,
    ) -> PredictionResponse:
        if file.content_type not in cfg.ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail=f"content type {file.content_type!r} not supported; use image/jpeg or image/png")
        data = await file.read(cfg.MAX_UPLOAD_BYTES + 1)
        if len(data) > cfg.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"file larger than {cfg.MAX_UPLOAD_BYTES:,} bytes")
        try:
            image = decode_image_bytes(data, allowed_formats=(cfg.ALLOWED_CONTENT_TYPES[file.content_type],))
        except InvalidImageError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        start = time.perf_counter()
        top_k = await run_in_threadpool(classifier.predict_image, image, k)  # CPU-bound: keep the event loop free
        review_below = classifier.review_threshold
        return PredictionResponse(model_version=classifier.version, filename=file.filename, top_k=top_k,
                                  needs_review=review_below is not None and top_k[0]["probability"] < review_below,
                                  inference_ms=round((time.perf_counter() - start) * 1000, 2))

    return app


app = create_app()
