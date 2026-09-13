# Notebook Template & Style Guide

Every topic notebook in this course follows **one** structure, so a learner builds a rhythm:
*understand → run → try it yourself → build it from scratch → practice → project → interview prep → resources.*

The course is **beginner-first**: early sections assume nothing, later sections ramp up to the depth interviewers expect.

---

## 1. Section order (topic notebooks)

Use these `##` headings **exactly** (emoji included), in this order. `tools/nb.py check` enforces it.

| # | Heading | What goes in it |
|---|---------|-----------------|
| 0 | `# <emoji> <Title> — <tagline>` | Header card (see §2) |
| 1 | `## 🤔 What Is <X>?` | Plain-English explanation + one everyday analogy. No jargon without a definition. |
| 2 | `## 🎯 Why It Matters` | Where it is used in real AI/ML jobs, and **where it shows up in interviews**. |
| 3 | `## ✅ By the End You Can` | 4–6 checkbox outcomes (`- [ ] Explain…`, `- [ ] Implement…`). |
| 4 | `## 📋 Table of Contents` | Links to every `##` section below. |
| 5 | `## ⚙️ Setup` | One markdown cell + one code cell: commented `%pip install` line with pinned major versions, imports, version print, random seed, and the `check()` helper (§4). |
| 6 | `## 1. <Concept>` … `## N. <Concept>` | Concept sections, each following the **section rhythm** (§3). Label difficulty in the heading when it ramps: `## 7. Broadcasting 🟡`. |
| 7 | `## 🔧 Build It From Scratch` | Implement the core idea without the library (or with only NumPy), then `assert` it matches the library. This is the #1 interview lever. |
| 8 | `## ⚠️ Common Pitfalls` | 4–6 pitfalls as runnable ❌ WRONG / ✅ RIGHT pairs. A WRONG example must either show the silent wrong result or catch and print the real error. |
| 9 | `## 🏋️ Practice Exercises` | Exactly 🟢 ×3 easy, 🟡 ×2 medium, 🔴 ×1 interview-style. Each has a checker and a hidden solution (§4). |
| 10 | `## 🚀 Mini Project: <Name>` | Real dataset. Goal → steps (each its own cells) → results → stretch goals → **🗣️ How to talk about this in an interview** (3–5 bullets). |
| 11 | `## 🎤 Interview Q&A` | 10–15 questions, **markdown only**, grouped (§5). |
| 12 | `## 🧪 Quick Quiz` | 5 "predict the output" / multiple-choice questions, answers hidden in `<details>`. |
| 13 | `## 📚 Resources` | Subsections: 📖 Official Docs · 🎥 Videos (2–3, **verified**) · 📄 Papers (ML topics) · 📘 Books & Courses · 🏋️ Practice. |
| 14 | `## 📝 Summary Cheat Sheet` | One table: concept → what it does → key API / formula. |
| 15 | `## ➡️ What's Next` | Link to the next notebook (relative path) and one sentence on why. |

---

## 2. Header card (first cell)

```markdown
# 🔢 NumPy — The Engine of Scientific Python

> **What you'll learn:** arrays, vectorization, broadcasting, linear algebra, and how to write fast numerical code without loops.

| | |
|---|---|
| **Difficulty** | 🟢 Beginner → 🟡 Intermediate |
| **Time** | ~4 hours (+2 hours exercises & project) |
| **Prerequisites** | [Python Basics](../00_Foundations/01_Python_Basics.ipynb) |
| **Tested with** | Python 3.12 · numpy 2.x |
| **Interview relevance** | ⭐⭐⭐ High — "vectorize this loop", broadcasting shapes, views vs copies |
```

---

## 3. Section rhythm (every concept section)

1. **Intuition** — short markdown: plain English, analogy or ASCII diagram, the one rule to remember.
2. **Code** — small runnable cells with real output. Comments explain *why*, not *what*.
3. **✍️ Your Turn** — `### ✍️ Your Turn` markdown prompt, then an exercise code cell (§4), then the hidden solution.
4. **💡 Interview angle** — one blockquote line:
   `> 💡 **Interview angle:** "Why is NumPy faster than a Python list?" — contiguous memory + compiled loops + no per-element type checks.`

Not every tiny section needs a Your Turn, but a notebook needs **at least 5** in total.

---

## 4. Exercises: checker + hidden solution

The notebook must **run top-to-bottom without errors** even when the learner has not attempted anything, so exercises use a placeholder and a checker that is friendly to `None`.

**Setup cell defines the helper once:**

```python
def check(name, got, expected, hint=""):
    """✅ if correct, ⏳ if not attempted yet (None / ...), ❌ AssertionError with a hint otherwise."""
    if got is None or got is ...:
        print(f"⏳ {name}: not attempted yet — replace None with your answer.")
        return
    try:
        import numpy as _np
        ok = _np.allclose(got, expected) if not isinstance(expected, (str, dict, set)) else got == expected
    except Exception:
        ok = got == expected
    assert ok, f"❌ {name}: not quite. {hint}"
    print(f"✅ {name}: correct!")
```

(Adapt the comparison to the topic — e.g. DataFrame equality — but keep the ✅ / ⏳ / ❌ contract.)

**Exercise code cell** — tag it `exercise`:

```python
# %% tags=["exercise"]
# ✍️ Normalize each row of X so it sums to 1 — no loops.
X = np.array([[1, 3], [2, 2]])
row_normalized = None  # TODO: your code here
check("row_normalize", row_normalized, [[0.25, 0.75], [0.5, 0.5]], hint="Try X / X.sum(axis=1, keepdims=True)")
```

**Solution markdown cell right after it** — tag it `solution`, keep the code in the first ```` ```python ```` fence:

````markdown
# %% [markdown] tags=["solution"]
# <details><summary>💡 Show solution</summary>
#
# ```python
# X = np.array([[1, 3], [2, 2]])
# row_normalized = X / X.sum(axis=1, keepdims=True)
# check("row_normalize", row_normalized, [[0.25, 0.75], [0.5, 0.5]])
# ```
#
# `keepdims=True` keeps the shape `(2, 1)` so broadcasting divides each row by its own sum.
# </details>
````

`python tools/nb.py run NB.ipynb --solutions` swaps every exercise cell for its solution and requires ✅ — so every solution is proven correct.

---

## 5. Interview Q&A format

Group questions under `### 🧠 Concepts`, `### 💻 Coding`, `### 🐛 Debugging Scenarios`, and (when relevant) `### 🏗️ Design`. Answers are hidden so the learner can answer out loud first:

```markdown
**Q3. What is the difference between a view and a copy in NumPy?**

<details><summary>Show answer</summary>

- **30-second answer:** Slicing returns a *view* that shares memory, so writing to it changes the original; fancy/boolean indexing returns a *copy*.
- **Go deeper:** Check with `np.shares_memory(a, b)`. Views make slicing O(1) and are why `a[::2]` is cheap. Use `.copy()` when you need independence.
- **❌ Common wrong answer:** "Slicing always copies, like Python lists."

</details>
```

---

## 6. Hard rules

1. **Everything is real.** No simulated, hard-coded, or invented outputs, metrics, curves, or "insights". Conclusions printed by code must be computed from the data (`if p < 0.05: ...`).
2. **No silent fallbacks.** If a cell genuinely needs something the laptop may not have (CUDA GPU, API key, Docker, Java), guard it and print an honest message, e.g. `⏭️ Skipped: needs a CUDA GPU — run this cell on Google Colab (Runtime → GPU).` Never print fake results instead.
3. **Runs on a CPU laptop** top-to-bottom in under ~10 minutes (Apple Silicon MPS may be used when available). GPU-only work lives in guarded cells with a Colab note.
4. **Current APIs** for the versions in the header card. No deprecated calls (check the installed version's docs).
5. **No leakage.** Tune, early-stop, and pick thresholds on a validation split; touch the test set once at the end.
6. **Real datasets** from scikit-learn, seaborn, OpenML, Hugging Face Datasets, or torchvision. Synthetic data only when the concept needs a controlled example — and say so.
7. **Verified links only.** `python tools/nb.py links NB.ipynb` must pass. Videos list the creator, length, and why to watch.
8. **Files a notebook writes** go under `./_outputs/` (git-ignored) or a temp dir.
9. **LLM notebooks default to a free local model** (Hugging Face small instruct model) so they run without keys; OpenAI / Anthropic / Gemini paths run only when the key is set and say so.
10. **Beginner tone.** Short sentences, define every term the first time, one idea per cell. Depth comes from exercises, from-scratch builds, and Q&A — not from walls of text.

---

## 7. Other notebook kinds

**Capstone projects** (`13_Capstone_Projects/`): header card → 🤔 Problem & Business Goal → 🗺️ Architecture (diagram) → ⚙️ Setup → Step 1…N → 📏 Evaluation → 🚢 Deployment (points to the real project folder with code, tests, Dockerfile) → 🗣️ Interview Walkthrough (a 2-minute project pitch + likely follow-up questions) → 🎤 Interview Q&A → 📚 Resources → ➡️ What's Next.

**Templates** (`14_Templates/`): header card → 🗺️ How to Use This Template → numbered fill-in steps with `# TODO` markers that still run on the example dataset → ✅ Checklist → 📚 Resources.

**Interview prep** (`15_Interview_Prep/`): header card → 🎯 How This Round Works → topic sections of questions with hidden answers (same Q&A format) → 🏋️ Timed Drills → 📋 Answer Framework / Cheat Sheet → 📚 Resources.

---

## 8. Authoring workflow

```bash
# 1. Write the notebook as a jupytext "percent" script (easy to diff and edit)
# 2. Build it into a notebook
.venv/bin/python tools/nb.py build my_notebook.py AI_Full_Stack_Engineer_Course/.../My_Notebook.ipynb
# 3. Execute it and save real outputs
.venv/bin/python tools/nb.py run AI_Full_Stack_Engineer_Course/.../My_Notebook.ipynb
# 4. Prove every exercise solution is correct
.venv/bin/python tools/nb.py run AI_Full_Stack_Engineer_Course/.../My_Notebook.ipynb --solutions
# 5. Structure + links
.venv/bin/python tools/nb.py check AI_Full_Stack_Engineer_Course/.../My_Notebook.ipynb
.venv/bin/python tools/nb.py links AI_Full_Stack_Engineer_Course/.../My_Notebook.ipynb
```
