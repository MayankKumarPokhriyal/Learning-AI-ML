"""Save and load model artifacts, and make predictions with the saved threshold."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn_service.data import FEATURE_COLUMNS, clean
from churn_service.evaluate import model_card_markdown

logger = logging.getLogger("uvicorn.error")
MODEL_FILE, METADATA_FILE, REFERENCE_FILE, CARD_FILE = "model.joblib", "metadata.json", "reference_stats.json", "MODEL_CARD.md"


def save_artifacts(pipeline, metadata: dict, reference_stats: dict, directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / name for name in (MODEL_FILE, METADATA_FILE, REFERENCE_FILE, CARD_FILE)}
    joblib.dump(pipeline, paths[MODEL_FILE])
    paths[METADATA_FILE].write_text(json.dumps(metadata, indent=2))
    paths[REFERENCE_FILE].write_text(json.dumps(reference_stats))
    paths[CARD_FILE].write_text(model_card_markdown(metadata))
    return paths


class ChurnModel:
    """The trained pipeline plus its metadata. The decision threshold always comes from the metadata."""

    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata

    @classmethod
    def load(cls, directory: str | Path) -> ChurnModel:
        directory = Path(directory)
        for name in (MODEL_FILE, METADATA_FILE):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"{name} not found in the artifacts directory — run `python -m churn_service.train` first")
        metadata = json.loads((directory / METADATA_FILE).read_text())
        if metadata["sklearn_version"] != sklearn.__version__:
            logger.warning("model trained with scikit-learn %s but running %s — retrain or pin the version",
                           metadata["sklearn_version"], sklearn.__version__)
        # joblib/pickle files can execute code when loaded: only load artifacts you built or trust
        return cls(joblib.load(directory / MODEL_FILE), metadata)

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(clean(X[FEATURE_COLUMNS]))[:, 1]

    def predict_records(self, records: list[dict]) -> list[dict]:
        """Records use the dataset's column names (e.g. 'MonthlyCharges'); returns probability and decision."""
        frame = pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS)
        proba = self.predict_proba(frame)
        return [{"churn_probability": float(p), "contact": bool(p >= self.threshold)} for p in proba]
