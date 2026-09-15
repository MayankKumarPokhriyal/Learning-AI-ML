## What does this change?

<!-- One or two sentences. Link the issue it fixes, e.g. "Fixes #12". -->

## Type of change

- [ ] Typo / wording / link fix
- [ ] Notebook content (explanation, exercise, interview question)
- [ ] New notebook or module
- [ ] Capstone project code
- [ ] Tooling, docs or CI

## Checklist

- [ ] The pull request targets `main` (not `learning`).
- [ ] Changed notebooks were re-executed: `python tools/nb.py run <notebook>` — outputs are real, with no errors or stderr warnings.
- [ ] `python tools/nb.py run <notebook> --solutions` passes (every exercise prints ✅).
- [ ] `python tools/nb.py check <notebook>` passes.
- [ ] `python tools/nb.py links <notebook> --titles` and `--local` pass.
- [ ] No API keys, tokens or home-directory paths in code or outputs.
- [ ] README, `docs/COURSE_MAP.md` and `docs/STUDY_PLAN.md` are updated if notebooks were added or renamed.
- [ ] Capstone project changes: the project's tests pass.

## Notes for the reviewer

<!-- Anything surprising in the results, data or licences worth a closer look. -->
