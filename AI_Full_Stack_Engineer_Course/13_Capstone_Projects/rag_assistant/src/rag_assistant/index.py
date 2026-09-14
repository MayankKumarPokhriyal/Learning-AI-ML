"""A persisted, versioned index with incremental ingestion.

Layout of the index folder:

    index/CURRENT                      name of the active version (switched atomically)
    index/versions/<version>/manifest.json   settings, per-document SHA-256 and chunk ids
    index/versions/<version>/chunks.jsonl    chunk text + metadata, one per line
    index/versions/<version>/embeddings.npy  float32 matrix, row i = chunk i
    index/versions/<version>/documents.jsonl parsed document text (for evidence lookup and debugging)

Only documents whose content hash changed are parsed, chunked and embedded again; unchanged documents reuse their rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_assistant.chunking import Chunk, chunk_document
from rag_assistant.corpus import SourceFile, sha256_bytes
from rag_assistant.guardrails import detect_injection
from rag_assistant.parsing import PARSER_VERSION, parse_document

SCHEMA_VERSION = 1


@dataclass
class Index:
    version: str
    manifest: dict
    chunks: list[Chunk]
    embeddings: np.ndarray
    documents: dict[str, str]
    directory: Path

    def __post_init__(self):
        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(f"index {self.version} is corrupt: {len(self.chunks)} chunks but {self.embeddings.shape[0]} embeddings")
        self.row_of = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def chunk(self, chunk_id: str) -> Chunk:
        return self.chunks[self.row_of[chunk_id]]


@dataclass
class IngestReport:
    index_version: str = ""
    previous_version: str | None = None
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks_total: int = 0
    chunks_embedded: int = 0
    flagged_chunks: dict[str, list[str]] = field(default_factory=dict)
    full_rebuild_reason: str | None = None
    wrote_new_version: bool = False
    seconds: float = 0.0
    embed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        text = (f"index {self.index_version}: {len(self.added)} added, {len(self.updated)} updated, {len(self.unchanged)} unchanged, "
                f"{len(self.removed)} removed | embedded {self.chunks_embedded} of {self.chunks_total} chunks in {self.embed_seconds:.1f}s "
                f"| {len(self.flagged_chunks)} chunks flagged by the injection detector | {self.seconds:.1f}s total")
        return text + (f" | full rebuild: {self.full_rebuild_reason}" if self.full_rebuild_reason else "")


class IndexStore:
    def __init__(self, root: Path, keep_versions: int = 3):
        self.root, self.keep_versions = Path(root), keep_versions

    @property
    def pointer(self) -> Path:
        return self.root / "CURRENT"

    def current_version(self) -> str | None:
        return self.pointer.read_text().strip() if self.pointer.is_file() else None

    def version_dir(self, version: str) -> Path:
        return self.root / "versions" / version

    def load(self, version: str | None = None) -> Index:
        version = version or self.current_version()
        if version is None:
            raise FileNotFoundError(f"no index in '{self.root.name}/' — run `python -m rag_assistant ingest` first")
        directory = self.version_dir(version)
        manifest = json.loads((directory / "manifest.json").read_text())
        chunks = [Chunk.from_dict(json.loads(line)) for line in (directory / "chunks.jsonl").read_text().splitlines() if line]
        documents = {row["doc_id"]: row["text"] for row in map(json.loads, (directory / "documents.jsonl").read_text().splitlines())}
        return Index(version, manifest, chunks, np.load(directory / "embeddings.npy"), documents, directory)

    def save(self, version: str, manifest: dict, chunks: list[Chunk], embeddings: np.ndarray, documents: dict[str, str]) -> Path:
        directory = self.version_dir(version)
        tmp = directory.with_name(directory.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "chunks.jsonl").write_text("".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in chunks))
        np.save(tmp / "embeddings.npy", embeddings.astype(np.float32))
        (tmp / "documents.jsonl").write_text("".join(json.dumps({"doc_id": d, "text": t}, ensure_ascii=False) + "\n" for d, t in documents.items()))
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.rmtree(directory, ignore_errors=True)
        tmp.rename(directory)
        pointer_tmp = self.root / "CURRENT.tmp"
        pointer_tmp.write_text(version + "\n")
        os.replace(pointer_tmp, self.pointer)  # atomic: readers see the old version or the new one, never a mix
        self._prune(keep=version)
        return directory

    def _prune(self, keep: str) -> None:
        versions = sorted((p for p in (self.root / "versions").iterdir() if p.is_dir() and not p.name.endswith(".tmp")),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        for old in versions[self.keep_versions:]:
            if old.name != keep:
                shutil.rmtree(old, ignore_errors=True)


def _with_source(chunk: Chunk, source: SourceFile) -> Chunk:
    anchor = chunk.url.partition("#")[2]
    return replace(chunk, url=f"{source.url}#{anchor}" if anchor else source.url, version=source.version)


def index_params(embedding_model: str, chunk_tokens: int, chunk_overlap: int) -> dict:
    """Everything that changes chunk contents or vectors. If any of it changes, every document must be re-indexed."""
    return {"schema_version": SCHEMA_VERSION, "parser_version": PARSER_VERSION, "embedding_model": embedding_model,
            "chunk_tokens": chunk_tokens, "chunk_overlap": chunk_overlap}


def ingest(sources: list[SourceFile], store: IndexStore, embedder, *, chunk_tokens: int = 300, chunk_overlap: int = 50) -> IngestReport:
    start = time.perf_counter()
    params = index_params(embedder.name, chunk_tokens, chunk_overlap)
    report = IngestReport()
    try:
        previous = store.load()
        report.previous_version = previous.version
        if previous.manifest["params"] != params:
            changed = sorted(k for k in params if previous.manifest["params"].get(k) != params[k])
            report.full_rebuild_reason = f"settings changed ({', '.join(changed)})"
    except FileNotFoundError:
        previous, report.full_rebuild_reason = None, "no previous index"
    reusable = previous.manifest["documents"] if previous is not None and report.full_rebuild_reason is None else {}

    plan = []  # (source, sha256, title, chunks, reused vectors or None)
    documents: dict[str, str] = {}
    for source in sorted(sources, key=lambda s: s.doc_id):
        raw = source.path.read_bytes()
        digest = sha256_bytes(raw)
        old = reusable.get(source.doc_id)
        if old and old["sha256"] == digest:
            rows = [previous.row_of[cid] for cid in old["chunk_ids"]]
            chunks = [_with_source(previous.chunks[r], source) for r in rows]  # same text, but maybe a newer release URL/tag
            plan.append((source, digest, old["title"], chunks, previous.embeddings[rows]))
            documents[source.doc_id] = previous.documents[source.doc_id]
            report.unchanged.append(source.doc_id)
            continue
        doc = parse_document(raw, doc_id=source.doc_id, url=source.url, version=source.version)
        chunks = chunk_document(doc, chunk_tokens, chunk_overlap)
        for chunk in chunks:
            chunk.flags = detect_injection(chunk.text)
        plan.append((source, digest, doc.title, chunks, None))
        documents[source.doc_id] = doc.text
        (report.updated if old else report.added).append(source.doc_id)
    report.removed = sorted(set(reusable) - {s.doc_id for s in sources})

    to_embed = [c.indexed_text for *_, chunks, vectors in plan if vectors is None for c in chunks]
    t0 = time.perf_counter()
    fresh = embedder.embed_documents(to_embed) if to_embed else None
    report.embed_seconds, report.chunks_embedded = time.perf_counter() - t0, len(to_embed)

    all_chunks, blocks, doc_entries, cursor = [], [], {}, 0
    for source, digest, title, chunks, vectors in plan:
        if vectors is None:
            vectors = fresh[cursor:cursor + len(chunks)] if chunks else np.zeros((0, fresh.shape[1] if fresh is not None else 0), np.float32)
            cursor += len(chunks)
        all_chunks.extend(chunks)
        if len(chunks):
            blocks.append(np.asarray(vectors, dtype=np.float32))
        doc_entries[source.doc_id] = {"sha256": digest, "url": source.url, "version": source.version, "title": title,
                                      "chunk_ids": [c.chunk_id for c in chunks]}
    embeddings = np.vstack(blocks) if blocks else np.zeros((0, 1), dtype=np.float32)
    report.chunks_total = len(all_chunks)
    report.flagged_chunks = {c.chunk_id: c.flags for c in all_chunks if c.flags}

    documents_fingerprint = {d: [e["sha256"], e["url"], e["version"]] for d, e in sorted(doc_entries.items())}  # metadata changes need a new version too
    fingerprint = json.dumps({"params": params, "documents": documents_fingerprint}, sort_keys=True)
    report.index_version = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    if previous is None or report.index_version != previous.version:
        manifest = {"version": report.index_version, "params": params, "documents": doc_entries, "n_chunks": len(all_chunks),
                    "flagged_chunks": report.flagged_chunks, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
        store.save(report.index_version, manifest, all_chunks, embeddings, documents)
        report.wrote_new_version = True
    report.seconds = time.perf_counter() - start
    return report
