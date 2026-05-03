# Contributing to TruthLens AI

Thanks for contributing.

## Development Workflow

1. Create a branch from your latest `main`.
2. Implement the change with tests.
3. Run local checks.
4. Open a PR with a focused description.

## Local Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Required Local Checks

```bash
python -B -m pytest -q
```

If you changed training/tuning/config behavior, also run:

```bash
python main.py
python evaluate.py
```

## Code Style Guidelines

- Keep modules focused by responsibility.
- Prefer `pathlib.Path` over `os.path`.
- Use type hints for new/changed functions.
- Use central settings from `src/utils/settings.py`.
- Use shared validators from `src/utils/input_validation.py` when accepting dataframes or numeric params.
- Do not duplicate logging setup; use `src/utils/logging_utils.py`.

## Documentation Expectations

For user-visible behavior changes, update:

- `README.md`
- `QUICKSTART.md`
- `KNOWLEDGE.md` (if architecture or file responsibilities changed)

## Pull Request Checklist

- [ ] Tests pass locally
- [ ] New behavior is validated
- [ ] Config keys documented when added
- [ ] Docs updated for user-facing changes
- [ ] No unrelated refactors mixed into the PR
