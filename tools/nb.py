#!/usr/bin/env python3
"""Notebook tooling for the course.

    python tools/nb.py build  SRC.py OUT.ipynb           jupytext percent script -> notebook
    python tools/nb.py run    NB.ipynb [--timeout S]     execute in place, save real outputs
    python tools/nb.py run    NB.ipynb --solutions       swap exercises for solutions; all must print ✅
    python tools/nb.py check  NB.ipynb [--kind KIND]     structure lint (see docs/NOTEBOOK_TEMPLATE.md)
    python tools/nb.py links  NB.ipynb [...] [--titles]  verify every link resolves

Exit code is non-zero when anything fails, so this doubles as a CI check.
"""
import argparse
import copy
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOPIC_SECTIONS = [
    r"🤔 What Is",
    r"🎯 Why It Matters",
    r"✅ By the End You Can",
    r"📋 Table of Contents",
    r"⚙️ Setup",
    r"__CONCEPTS__",
    r"🔧 Build It From Scratch",
    r"⚠️ Common Pitfalls",
    r"🏋️ Practice Exercises",
    r"🚀 Mini Project",
    r"🎤 Interview Q&A",
    r"🧪 Quick Quiz",
    r"📚 Resources",
    r"📝 Summary Cheat Sheet",
    r"➡️ What's Next",
]
KIND_SECTIONS = {
    "topic": TOPIC_SECTIONS,
    "capstone": [r"🤔 Problem", r"🗺️ Architecture", r"⚙️ Setup", r"__STEPS__", r"📏 Evaluation",
                 r"🚢 Deployment", r"🗣️ Interview Walkthrough", r"🎤 Interview Q&A", r"📚 Resources",
                 r"➡️ What's Next"],
    "template": [r"🗺️ How to Use This Template", r"__STEPS__", r"✅ Checklist", r"📚 Resources"],
    "prep": [r"🎯 How This Round Works", r"__STEPS__", r"🏋️ Timed Drills", r"📋 ", r"📚 Resources"],
}
HEADER_FIELDS = ["Difficulty", "Time", "Prerequisites", "Tested with", "Interview relevance"]
UA = {"User-Agent": "Mozilla/5.0 (course-link-checker)"}


def src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


# ─── build ────────────────────────────────────────────────────────────────────
def cmd_build(a):
    import jupytext
    import nbformat

    nb = jupytext.read(a.src, fmt="py:percent")
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for c in nb.cells:
        c.metadata = {k: v for k, v in c.metadata.items() if k == "tags"}
        if c.cell_type == "code":
            # jupytext turns "# %pip install ..." into a live magic; install lines must stay commented for learners
            c.source = re.sub(r"^([ \t]*)([%!]\s*pip\s+install)", r"\1# \2", c.source, flags=re.M)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, a.out)
    print(f"built {a.out} ({len(nb.cells)} cells)")


# ─── run ──────────────────────────────────────────────────────────────────────
def solution_code(md):
    m = re.search(r"```python\n(.*?)```", md, re.S)
    return m.group(1) if m else None


def pin_kernel_to_this_python():
    """A user-level 'python3' kernelspec can point at another interpreter; force the one running this script."""
    import os
    import tempfile

    kdir = Path(tempfile.mkdtemp(prefix="nbkernel_")) / "kernels" / "python3"
    kdir.mkdir(parents=True)
    (kdir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(kdir.parent.parent) + os.pathsep + os.environ.get("JUPYTER_PATH", "")


def cmd_run(a):
    pin_kernel_to_this_python()
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    path = Path(a.notebook).resolve()
    nb = nbformat.read(path, as_version=4)
    target = copy.deepcopy(nb) if a.solutions else nb
    swapped = []
    if a.solutions:
        cells = target.cells
        for i, c in enumerate(cells):
            if c.cell_type == "code" and "exercise" in c.metadata.get("tags", []):
                nxt = cells[i + 1] if i + 1 < len(cells) else None
                code = solution_code(src(nxt)) if nxt is not None and "solution" in nxt.metadata.get("tags", []) else None
                if code is None:
                    print(f"❌ cell {i}: exercise has no tagged solution cell with a ```python block")
                    return 1
                c.source = code
                swapped.append(i)
    client = NotebookClient(target, timeout=a.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except CellExecutionError:
        for i, c in enumerate(target.cells):
            errs = [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
            if errs:
                tb = "\n".join(errs[0].get("traceback", []))
                tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
                print(f"❌ cell {i} failed:\n--- source ---\n{src(c)[:1500]}\n--- error ---\n{tb[-3000:]}")
                break
        return 1
    if a.solutions:
        bad = []
        for i in swapped:
            text = "".join(o.get("text", "") for o in target.cells[i].get("outputs", []))
            if "✅" not in text:
                bad.append(i)
        if bad:
            print(f"❌ solution cells did not print ✅: {bad}")
            return 1
        print(f"✅ all {len(swapped)} exercise solutions pass ({path.name})")
        return 0
    nbformat.write(target, path)
    print(f"✅ executed {path.name} — outputs saved")
    return 0


# ─── check ────────────────────────────────────────────────────────────────────
def cmd_check(a):
    nb = json.loads(Path(a.notebook).read_text())
    cells = nb["cells"]
    errors, warnings = [], []
    heads = []
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown":
            for line in src(c).splitlines():
                if line.startswith("## "):
                    heads.append((i, line[3:].strip()))
    # section order
    wanted = KIND_SECTIONS[a.kind]
    pos, found_numbered = 0, 0
    for w in wanted:
        if w in ("__CONCEPTS__", "__STEPS__"):
            continue
        idx = next((k for k in range(pos, len(heads)) if heads[k][1].startswith(w)), None)
        if idx is None:
            errors.append(f"missing or out-of-order section: '## {w}'")
        else:
            pos = idx + 1
    found_numbered = sum(1 for _, h in heads if re.match(r"(\d+\.|Step \d+)", h))
    if found_numbered < 3:
        errors.append(f"only {found_numbered} numbered concept/step sections (need ≥3)")
    # header card
    first = src(cells[0]) if cells else ""
    if not first.startswith("# "):
        errors.append("first cell must start with '# <emoji> Title — tagline'")
    for f in HEADER_FIELDS if a.kind in ("topic", "capstone", "prep") else []:
        if f"**{f}**" not in first:
            errors.append(f"header card missing **{f}**")
    md_all = "\n".join(src(c) for c in cells if c["cell_type"] == "markdown")
    code_cells = [(i, c) for i, c in enumerate(cells) if c["cell_type"] == "code"]
    if a.kind == "topic":
        n_turn = md_all.count("### ✍️ Your Turn")
        if n_turn < 5:
            errors.append(f"{n_turn} '### ✍️ Your Turn' blocks (need ≥5)")
        for emoji, need in (("🟢", 3), ("🟡", 2), ("🔴", 1)):
            practice = md_all.split("## 🏋️ Practice Exercises", 1)[-1].split("## 🚀 Mini Project", 1)[0]
            if practice.count(emoji) < need:
                errors.append(f"practice exercises need {need}× {emoji}, found {practice.count(emoji)}")
        if "Interview angle" not in md_all:
            errors.append("no '💡 Interview angle' notes")
        if "How to talk about this in an interview" not in md_all:
            errors.append("mini project lacks '🗣️ How to talk about this in an interview'")
    if a.kind in ("topic", "capstone", "prep"):
        qa = md_all.split("## 🎤 Interview Q&A", 1)[-1] if a.kind != "prep" else md_all
        nq = len(re.findall(r"\*\*Q\d+\.", qa))
        need_q = 10 if a.kind == "topic" else 6
        if nq < need_q:
            errors.append(f"interview Q&A has {nq} questions formatted '**Qn.' (need ≥{need_q})")
        for key in ("30-second answer", "Common wrong answer"):
            if key not in qa:
                errors.append(f"interview answers missing '{key}'")
    if a.kind == "topic":
        quiz = md_all.split("## 🧪 Quick Quiz", 1)[-1].split("## 📚 Resources", 1)[0]
        if quiz.count("<details>") < 5:
            errors.append("quick quiz needs ≥5 questions with hidden answers")
        res = md_all.split("## 📚 Resources", 1)[-1]
        if len(re.findall(r"youtu", res)) < 2:
            errors.append("resources need ≥2 YouTube videos")
    # exercises paired with solutions
    for i, c in code_cells:
        if "exercise" in c.get("metadata", {}).get("tags", []):
            nxt = cells[i + 1] if i + 1 < len(cells) else {}
            if "solution" not in nxt.get("metadata", {}).get("tags", []) or not solution_code(src(nxt)):
                errors.append(f"cell {i}: exercise not followed by a tagged solution with a ```python block")
    # install lines must be commented out (learners uncomment them)
    for i, c in code_cells:
        if re.search(r"^[ \t]*[%!]\s*pip\s+install", src(c), re.M):
            errors.append(f"cell {i}: live '%pip install' — comment it out (rebuild with tools/nb.py build)")
    # execution state
    for i, c in code_cells:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors.append(f"cell {i}: saved output contains an error")
        if c.get("execution_count") is None and src(c).strip():
            warnings.append(f"cell {i}: not executed")
    # honesty
    for i, c in enumerate(cells):
        if re.search(r"simulat(ed|ion mode)|fake (result|output|metric)|hard-?coded result", src(c), re.I):
            warnings.append(f"cell {i}: mentions simulated/fake output — make sure nothing is invented")
    print(f"{Path(a.notebook).name}: {len(cells)} cells, {found_numbered} numbered sections")
    for w in warnings[:15]:
        print(f"  ⚠️  {w}")
    if len(warnings) > 15:
        print(f"  ⚠️  … {len(warnings) - 15} more warnings")
    for e in errors:
        print(f"  ❌ {e}")
    if not errors:
        print("  ✅ structure OK")
    return 1 if errors else 0


# ─── links ────────────────────────────────────────────────────────────────────
def check_url(url):
    try:
        if re.search(r"youtube\.com/watch|youtu\.be/|youtube\.com/playlist", url):
            q = "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe="")
            try:
                with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=20) as r:
                    title = json.loads(r.read())["title"]
                if "watch" in url or "youtu.be" in url:
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
                            m = re.search(r'"lengthSeconds":"(\d+)"', r.read().decode("utf8", "ignore"))
                        if m:
                            s = int(m.group(1))
                            title += f"  [length {s // 3600}h{s % 3600 // 60:02d}m]" if s >= 3600 else f"  [length {s // 60}m{s % 60:02d}s]"
                    except Exception:  # noqa: BLE001 — length is a nice-to-have
                        pass
                return url, True, title
            except urllib.error.HTTPError as e:
                if e.code == 401:  # exists, embedding disabled
                    return url, True, "(exists; embedding disabled)"
                return url, False, f"HTTP {e.code}"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read(200_000).decode("utf8", "ignore")
            m = re.search(r'<meta name="citation_title" content="([^"]+)"', body) or re.search(r"<title>(.*?)</title>", body, re.S)
            return url, True, " ".join(m.group(1).split())[:90] if m else ""
    except urllib.error.HTTPError as e:
        if e.code in (403, 429, 999):  # bot-blocked but reachable
            return url, True, f"(HTTP {e.code}, likely bot protection — verify manually)"
        return url, False, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return url, False, str(e)[:60]


def markdown_links(line):
    """Yield (label, url) for every [label](url) in a line, allowing brackets nested inside the label."""
    for m in re.finditer(r"\]\((https?://[^)\s]+)\)", line):
        depth, i = 0, m.start()
        while i >= 0:
            ch = line[i]
            if ch == "]":
                depth += 1
            elif ch == "[":
                depth -= 1
                if depth == 0:
                    break
            i -= 1
        label = line[i + 1:m.start()] if i >= 0 else ""
        yield label, m.group(1)


def local_links(nbp):
    """Return (label, target) for relative links to files that don't exist (anchors and URLs ignored)."""
    missing = []
    for c in json.loads(Path(nbp).read_text())["cells"]:
        if c["cell_type"] != "markdown":
            continue
        for m in re.finditer(r"\[([^\]]*)\]\((?!https?://|#|mailto:)([^)\s]+)\)", src(c)):
            target = urllib.parse.unquote(m.group(2).split("#", 1)[0])
            if target and not (Path(nbp).parent / target).exists():
                missing.append((m.group(1), target))
    return missing


def cmd_links(a):
    if a.local:
        bad = 0
        for nbp in a.notebooks:
            for label, target in local_links(nbp):
                bad += 1
                print(f"❌ {Path(nbp).name}: [{label}] → {target} (file not found)")
        print(f"local links: {bad} missing")
        return 1 if bad else 0
    urls = {}
    for nbp in a.notebooks:
        for c in json.loads(Path(nbp).read_text())["cells"]:
            if c["cell_type"] != "markdown":
                continue
            for line in src(c).splitlines():
                for label, url in markdown_links(line):
                    urls.setdefault(url, (Path(nbp).name, label))
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(check_url, urls))
    bad = 0
    for url, ok, info in results:
        nbname, label = urls[url]
        if not ok:
            bad += 1
            print(f"❌ {nbname}: [{label}] {url} → {info}")
        elif a.titles:
            print(f"✅ [{label[:50]}] → {info}")
    print(f"{len(results)} links checked, {bad} broken")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("src"); b.add_argument("out")
    r = sub.add_parser("run"); r.add_argument("notebook"); r.add_argument("--timeout", type=int, default=1800)
    r.add_argument("--solutions", action="store_true")
    c = sub.add_parser("check"); c.add_argument("notebook")
    c.add_argument("--kind", choices=list(KIND_SECTIONS), default="topic")
    l = sub.add_parser("links"); l.add_argument("notebooks", nargs="+"); l.add_argument("--titles", action="store_true")
    l.add_argument("--local", action="store_true", help="check relative links to other files instead of URLs")
    a = p.parse_args()
    fn = {"build": cmd_build, "run": cmd_run, "check": cmd_check, "links": cmd_links}[a.cmd]
    sys.exit(fn(a) or 0)


if __name__ == "__main__":
    main()
