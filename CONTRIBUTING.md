# Contributing

Thanks for helping make this course better. Every fix — a typo, a clearer explanation, a dead link, an outdated API — helps thousands of learners.

## Ways to contribute

| You want to… | Do this |
|---|---|
| Report a notebook that errors or gives wrong output | Open a **Notebook bug** issue (the form asks for the notebook, the cell and your `check_setup.py` output). |
| Suggest a clearer explanation, a missing topic or a new exercise | Open a **Content request** issue. |
| Fix a typo, link or small explanation | Open a pull request directly. |
| Change code in a notebook or add a new notebook | Open an issue first so we can agree on the approach, then follow the checklist below. |

## Ground rules

These keep the course trustworthy. Pull requests that break them can't be merged.

1. **Everything runs for real.** Never type or paste outputs. Every saved output must come from executing the notebook.
2. **No invented results.** Numbers, tables and conclusions are computed from the data in the notebook. If something can't run (GPU, API key, cluster), the cell prints an honest `⏭️ Skipped: …` message that says what to set up.
3. **Real, licence-checked data.** State the source and licence where a notebook downloads data or models.
4. **Current APIs.** No deprecation warnings in saved outputs.
5. **Verified links.** Every link must resolve and its label must match the page title.
6. **No secrets, no personal paths.** Never commit API keys, tokens or outputs containing your home directory.
7. **Beginner-first.** Define every term the first time it appears; build up to interview depth with 🟢/🟡/🔴 sections.

The full structure every notebook follows is in [docs/NOTEBOOK_TEMPLATE.md](docs/NOTEBOOK_TEMPLATE.md).

## Development setup

Follow [Getting started](README.md#getting-started-full-setup) in the README, then run:

```bash
python tools/check_setup.py
```

## Workflow

```bash
git switch main && git pull
git switch -c fix/<short-description>      # branch from main (not from learning)
# …make your change…
git push -u origin fix/<short-description>  # then open a pull request against main
```

Keep pull requests focused: one notebook or one topic per pull request is easiest to review.

## Notebook checklist

Run these from the repository root with the environment the notebook uses (see [docs/COURSE_MAP.md](docs/COURSE_MAP.md)):

```bash
NB=AI_Full_Stack_Engineer_Course/<module>/<Notebook>.ipynb
python tools/nb.py run   $NB               # execute top to bottom and save real outputs
python tools/nb.py run   $NB --solutions   # every exercise solution must print ✅ (does not save)
python tools/nb.py check $NB               # template structure (add --kind capstone|template|prep for those folders)
python tools/nb.py links $NB --titles      # every link resolves and matches its label
python tools/nb.py links --local $NB       # links to other notebooks resolve
```

Before opening the pull request, also confirm:

- [ ] Outputs are from a fresh run (**Kernel → Restart & Run All** or `nb.py run`), with no errors or warnings on stderr.
- [ ] No API keys, tokens or home-directory paths appear in code or outputs.
- [ ] Prerequisites and **What's Next** links still form a correct chain.
- [ ] If you added or renamed a notebook: `README.md`, `docs/COURSE_MAP.md` and `docs/STUDY_PLAN.md` are updated.
- [ ] If you changed a capstone project folder: its test suite passes (`pytest` inside the project folder).

## Commit messages

Use a short imperative summary line (e.g. `Fix the SMOTE leakage example in 17_Imbalanced_Data`), then a blank line and a few lines explaining *why* if it isn't obvious.

## Questions

Open a [Content request](../../issues/new/choose) issue, or reach out on [LinkedIn](https://www.linkedin.com/in/mayank-kumar-pokhriyal/). By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
