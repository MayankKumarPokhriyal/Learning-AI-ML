"""Embedding models and rerankers.

`SentenceTransformerEmbedder` and `CrossEncoderReranker` are the real models. `HashingEmbedder` and `OverlapReranker` are
deterministic test doubles for fast offline unit tests — they are not semantic models and are never used for reported results.
"""

from __future__ import annotations

import gc
import hashlib
import re

import numpy as np

WORD = re.compile(r"[a-z0-9]+")


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, *, device: str = "cpu", query_prompt: str = "", batch_size: int = 32):
        self.name, self.device, self.query_prompt, self.batch_size = model_name, device, query_prompt, batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: importing torch is slow

            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], prompt=self.query_prompt or None, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vector[0], dtype=np.float32)

    def release(self) -> None:
        self._model = None
        gc.collect()


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: str = "cpu", batch_size: int = 32):
        self.name, self.device, self.batch_size = model_name, device, batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name, device=self.device)
        return self._model

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False), dtype=float)

    def release(self) -> None:
        self._model = None
        gc.collect()


class HashingEmbedder:
    """TEST DOUBLE: signed feature hashing of words and word pairs. Deterministic and instant, but not semantic."""

    def __init__(self, dimension: int = 384):
        self.name, self.dimension = f"test-hashing-{dimension}", dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        words = WORD.findall(text.lower())
        for feature in words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dimension] += 1.0 if digest[4] & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dimension), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def release(self) -> None:
        pass


class OverlapReranker:
    """TEST DOUBLE: scores a passage by the share of query words it contains."""

    name = "test-word-overlap"

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        words = set(WORD.findall(query.lower()))
        return np.array([len(words & set(WORD.findall(t.lower()))) / max(1, len(words)) for t in texts], dtype=float)

    def release(self) -> None:
        pass
