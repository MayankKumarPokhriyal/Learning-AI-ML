"""FastAPI service: `uvicorn churn_service.api:app --host 0.0.0.0 --port 8000`."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from churn_service import __version__
from churn_service import config as cfg
from churn_service.predict import ChurnModel
from churn_service.schemas import BatchRequest, BatchResponse, Customer, Prediction

logger = logging.getLogger("uvicorn.error")


def get_model(request: Request) -> ChurnModel:
    """Dependency: the loaded model, or 503 while it is missing."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {request.app.state.load_error}")
    return model


# Module level (not inside create_app) so FastAPI can resolve the annotation
ModelDep = Annotated[ChurnModel, Depends(get_model)]


def create_app(artifacts_dir: str | Path | None = None) -> FastAPI:
    """App factory: tests pass their own artifacts folder; production reads CHURN_ARTIFACTS_DIR at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        directory = Path(artifacts_dir) if artifacts_dir is not None else cfg.artifacts_dir()
        app.state.model, app.state.load_error = None, None
        try:
            app.state.model = ChurnModel.load(directory)  # once per process, not once per request
            logger.info("loaded model %s (threshold %.2f)", app.state.model.version, app.state.model.threshold)
        except (FileNotFoundError, ValueError, KeyError) as err:
            app.state.load_error = str(err)  # stay alive so /health works, but /ready reports 503
            logger.error("model not loaded: %s", err)
        yield
        app.state.model = None

    app = FastAPI(title="Churn Prediction Service", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        """Liveness: the process is up (does not check the model)."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        """Readiness: the model is loaded and the instance can take traffic."""
        model = request.app.state.model
        if model is None:
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": request.app.state.load_error})
        return {"status": "ready", "model_version": model.version}

    @app.get("/model")
    def model_info(model: ModelDep) -> dict:
        meta = model.metadata
        return {"model_version": model.version, "model_type": meta["model_type"], "trained_at": meta["trained_at"],
                "threshold": model.threshold, "business_assumptions": meta["business_assumptions"],
                "test_metrics": {k: meta["test_metrics"][k] for k in ("roc_auc", "profit_per_1000")}}

    # Plain `def` (not `async def`): scikit-learn is CPU-bound, so FastAPI runs it in a worker thread
    @app.post("/predict", response_model=Prediction)
    def predict(customer: Customer, model: ModelDep) -> Prediction:
        result = model.predict_records([customer.to_record()])[0]
        return Prediction(customer_id=customer.customer_id, threshold=model.threshold, model_version=model.version, **result)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, model: ModelDep) -> BatchResponse:
        results = model.predict_records([c.to_record() for c in batch.customers])  # one vectorized call
        predictions = [Prediction(customer_id=c.customer_id, threshold=model.threshold, model_version=model.version, **r)
                       for c, r in zip(batch.customers, results, strict=True)]
        return BatchResponse(model_version=model.version, threshold=model.threshold, n_customers=len(predictions),
                             n_contact=sum(p.contact for p in predictions), predictions=predictions)

    return app


app = create_app()
