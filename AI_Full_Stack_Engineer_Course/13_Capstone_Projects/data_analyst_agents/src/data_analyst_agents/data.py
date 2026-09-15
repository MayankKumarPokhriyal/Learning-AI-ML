"""Fetch the BIRD mini-dev subset this project uses, without downloading the whole 800 MB archive.

    python -m data_analyst_agents.data                 # = `make data`
    python -m data_analyst_agents.data --dbs financial # one database only

BIRD mini-dev (https://github.com/bird-bench/mini_dev) is licensed CC BY-SA 4.0. The release is one zip file. The
server supports HTTP Range requests, so we read the zip's central directory (at the end of the file) and then only the
bytes of the members we need: three SQLite databases, their column-description CSVs, and the question file.
Everything lands in data/bird_minidev/ (git-ignored) with a manifest of sizes and sha256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from data_analyst_agents import config

MINIDEV_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
ZIP_ROOT = "minidev/MINIDEV/"
QUESTIONS_MEMBER = ZIP_ROOT + "mini_dev_sqlite.json"
USER_AGENT = "data-analyst-agents-capstone/1.0 (educational project; range requests; cached)"


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP Range requests, so `zipfile` can open a remote archive lazily."""

    def __init__(self, url: str, *, timeout_s: float = 120, retries: int = 4, opener=urllib.request.urlopen, sleep=time.sleep):
        self.url, self.timeout_s, self.retries, self._open, self._sleep = url, timeout_s, retries, opener, sleep
        self.pos, self.requests, self.bytes_downloaded = 0, 0, 0
        head = self._request(urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT}))
        self.size = int(head.headers["Content-Length"])
        self.last_modified = head.headers.get("Last-Modified", "")
        if head.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError(f"{url} does not advertise byte ranges; download the whole file instead")

    def _request(self, request):
        for attempt in range(self.retries + 1):
            try:
                self.requests += 1
                return self._open(request, timeout=self.timeout_s)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                if attempt == self.retries or (isinstance(err, urllib.error.HTTPError) and err.code < 500 and err.code != 429):
                    raise
                self._sleep(min(30.0, 2.0 * 2**attempt))  # exponential backoff for transient network errors
        raise AssertionError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": USER_AGENT})
        with self._request(request) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        self.bytes_downloaded += len(data)
        return len(data)


def members_for(db_ids: list[str], names: list[str]) -> list[str]:
    wanted = [QUESTIONS_MEMBER]
    for db_id in db_ids:
        prefix = f"{ZIP_ROOT}dev_databases/{db_id}/"
        wanted += [n for n in names if n.startswith(prefix) and (n.endswith(".sqlite") or "/database_description/" in n and n.endswith(".csv"))]
    return wanted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(db_ids: list[str] | None = None, dest: Path | None = None, *, url: str = MINIDEV_ZIP_URL, log=print) -> dict:
    """Download the chosen databases + descriptions + questions (skips files already present with the right size)."""
    db_ids = list(db_ids or config.DATABASES)
    dest = Path(dest or config.bird_dir())
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"source": url, "license": "CC BY-SA 4.0", "files": {}}
    remote = HttpRangeFile(url)
    started = time.perf_counter()
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)) as archive:
        infos = {i.filename: i for i in archive.infolist()}
        for member in members_for(db_ids, list(infos)):
            info = infos[member]
            target = dest / member.removeprefix(ZIP_ROOT).removeprefix("dev_databases/")
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            with archive.open(info) as src, partial.open("wb") as out:  # zipfile verifies the CRC-32 while reading
                shutil.copyfileobj(src, out, 1 << 20)
            partial.replace(target)  # atomic: a crash never leaves a half-written database behind
            rel = target.relative_to(dest).as_posix()
            manifest["files"][rel] = {"bytes": info.file_size, "compressed_bytes": info.compress_size, "sha256": sha256_file(target),
                                      "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
            log(f"  fetched {rel} ({info.file_size / 1e6:.1f} MB)")
    manifest.update(archive_bytes=remote.size, archive_last_modified=remote.last_modified)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return {"databases": db_ids, "dest": dest, "requests": remote.requests, "downloaded_mb": round(remote.bytes_downloaded / 1e6, 2),
            "archive_mb": round(remote.size / 1e6, 1), "seconds": round(time.perf_counter() - started, 1), "files": len(manifest["files"])}


def load_questions(path: Path | None = None) -> list[dict]:
    return json.loads(Path(path or config.questions_path()).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dbs", nargs="*", default=list(config.DATABASES))
    args = parser.parse_args(argv)
    report = fetch(args.dbs)
    print(f"BIRD mini-dev subset ready in {report['dest']}: {report['files']} files, downloaded {report['downloaded_mb']} MB of a "
          f"{report['archive_mb']} MB archive with {report['requests']} HTTP requests in {report['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
