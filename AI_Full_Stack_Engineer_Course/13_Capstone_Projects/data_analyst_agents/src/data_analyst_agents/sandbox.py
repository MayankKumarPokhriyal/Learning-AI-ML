"""Run model-written analysis code in a subprocess sandbox.

What it does: a static import/call allowlist (AST), a fresh temp working directory, a scrubbed environment (no API keys,
no tokens, PATH only), isolated Python mode (-I), CPU-time and file-size rlimits (plus address space on Linux), a
wall-clock timeout that kills the whole process group, output caps, and only allowlisted output files come back.

What it does NOT do: a subprocess is not a security boundary. It shares the kernel and the user account, can read
files by absolute path, and (on macOS) can open network connections. In production run it inside a container or
microVM with --network none, a read-only root filesystem and a memory limit (see the README).
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_IMPORTS = frozenset({"pandas", "numpy", "matplotlib", "json", "math", "statistics", "datetime", "collections", "itertools", "re"})
BLOCKED_CALLS = frozenset({"exec", "eval", "compile", "__import__", "open", "input", "breakpoint", "globals", "locals", "vars", "memoryview"})

_BOOTSTRAP = """import resource, runpy, sys
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({fsize}, {fsize}))
if sys.platform.startswith("linux"):
    resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
runpy.run_path("analysis.py", run_name="__main__")
"""


@dataclass
class SandboxResult:
    status: str  # ok | error | timeout | rejected
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0
    files: dict[str, bytes] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def summary_json(self) -> dict | None:
        for line in reversed(self.stdout.strip().splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            return value if isinstance(value, dict) else None
        return None


def static_check(code: str) -> list[str]:
    """Reject imports outside the allowlist, dangerous builtins, and dunder attribute tricks. Defense in depth only:
    a determined attacker can evade static checks, which is why the process limits and isolation exist too."""
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        return [f"syntax error: {err.msg} (line {err.lineno})"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [f"import {a.name} is not allowed" for a in node.names if a.name.split(".")[0] not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"from {node.module} import … is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            problems.append(f"calling {node.func.id}() is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            problems.append(f"dunder attribute access (.{node.attr}) is not allowed")
    return list(dict.fromkeys(problems))


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # the whole group: children the script spawned die too
    except (ProcessLookupError, PermissionError):
        pass


async def run_python(code: str, *, inputs: dict[str, bytes] | None = None, timeout_s: float = 30.0, cpu_s: int | None = None,
                     max_output_chars: int = 4000, max_file_bytes: int = 5_000_000, allowed_suffixes: tuple[str, ...] = (".png", ".json", ".csv"),
                     memory_bytes: int = 1_500_000_000, cache_dir: Path | None = None, check_imports: bool = True) -> SandboxResult:
    if check_imports:
        problems = static_check(code)
        if problems:
            return SandboxResult("rejected", problems=problems)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="analyst-sandbox-") as work:
        workdir = Path(work)
        for name, data in (inputs or {}).items():
            (workdir / Path(name).name).write_bytes(data)
        (workdir / "analysis.py").write_text(code)
        cpu = int(cpu_s or max(1, int(timeout_s) + 1))
        (workdir / "_bootstrap.py").write_text(_BOOTSTRAP.format(cpu=cpu, fsize=max_file_bytes * 2, mem=memory_bytes))
        mpl_cache = Path(cache_dir) if cache_dir else workdir / ".mpl"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        env = {"PATH": "/usr/bin:/bin", "HOME": work, "TMPDIR": work, "LANG": "C.UTF-8", "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(mpl_cache),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}  # nothing inherited: no API keys, tokens or proxies
        proc = await asyncio.create_subprocess_exec(sys.executable, "-I", "_bootstrap.py", cwd=work, env=env, stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            status = "ok" if proc.returncode == 0 else "error"
        except TimeoutError:
            _kill_group(proc)
            await proc.wait()
            stdout, stderr, status = b"", f"killed after the {timeout_s:g} s time limit".encode(), "timeout"
        except asyncio.CancelledError:
            _kill_group(proc)
            await proc.wait()
            raise

        def clean(raw: bytes) -> str:
            text = raw.decode("utf-8", "replace").replace(work, "<sandbox>").replace(os.path.realpath(work), "<sandbox>")
            text = text.replace(sys.prefix, "<python>").replace(str(Path.home()), "~")
            return text if len(text) <= max_output_chars else "…" + text[-max_output_chars:]

        files = {}
        for path in workdir.iterdir():
            if path.is_file() and path.suffix in allowed_suffixes and path.name not in (inputs or {}) and path.stat().st_size <= max_file_bytes:
                files[path.name] = path.read_bytes()
        return SandboxResult(status, proc.returncode, clean(stdout), clean(stderr), round(time.perf_counter() - start, 3), files)
