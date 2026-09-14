"""Download the pinned documentation pages and record their provenance (source URL, release tag, SHA-256)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from rag_assistant import config as cfg

SOURCES_FILE = "sources.json"


@dataclass(frozen=True)
class SourceFile:
    doc_id: str  # page path without ".md", e.g. "concepts/cache"
    path: Path
    url: str  # pinned URL of the exact text that was indexed
    version: str  # release tag


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_corpus(dest: Path, tag: str = cfg.UV_DOCS_TAG, pages: tuple[str, ...] = cfg.DOCS_PAGES, timeout: float = 30.0) -> dict:
    """Download each page once (idempotent) and write `sources.json` with URL + SHA-256 per page. Returns the manifest plus counts."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / SOURCES_FILE
    previous = json.loads(manifest_path.read_text()).get("pages", {}) if manifest_path.exists() else {}
    entries, missing, downloaded = {}, [], 0
    with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": "rag-assistant-course/1.0 (educational)"}) as http:
        for page in pages:
            path = dest / page
            known = previous.get(page)
            if known and path.exists() and sha256_bytes(path.read_bytes()) == known["sha256"]:
                entries[page] = known
                continue
            response = http.get(cfg.RAW_URL.format(repo=cfg.UV_REPO, tag=tag, page=page))
            if response.status_code == 404:  # the page does not exist at this tag
                missing.append(page)
                continue
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            downloaded += 1
            entries[page] = {"url": cfg.SOURCE_URL.format(repo=cfg.UV_REPO, tag=tag, page=page), "sha256": sha256_bytes(response.content),
                             "bytes": len(response.content)}
    manifest = {"repo": cfg.UV_REPO, "tag": tag, "license": cfg.UV_DOCS_LICENSE, "pages": entries, "missing": missing}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**manifest, "downloaded": downloaded, "total_bytes": sum(e["bytes"] for e in entries.values())}


def load_sources(corpus_dir: Path) -> list[SourceFile]:
    """The files to index, from `sources.json` (files listed there but deleted on disk are skipped)."""
    corpus_dir = Path(corpus_dir)
    manifest = json.loads((corpus_dir / SOURCES_FILE).read_text())
    return [SourceFile(page.removesuffix(".md"), corpus_dir / page, entry["url"], manifest["tag"])
            for page, entry in sorted(manifest["pages"].items()) if (corpus_dir / page).is_file()]
