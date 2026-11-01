---
name: check
description: Run the full verification gate for knowledge-agent — pytest, ruff lint, ruff format check, and the strict eval suite. Use before finishing a change, before committing, or when the user says "check", "verify", "run the gates", or asks whether the repo is green.
---

# /check

Run all four gates from the repo root. Never stop at the first failure — run all four, then report.

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m knowledge_agent eval --strict
```

Notes:
- Ruff is only in `.venv` (no global install). If `.venv` is missing, run `uv sync` first.
- Everything is offline-safe: no API key is needed, and `eval --strict` exits 1 on any failing case.
- `pytest` runs with `-q` via `addopts`; a single test can be narrowed with `-k <expr>` or `path::test_name`.

## Report

One short block:

```
pytest:   PASS (N tests)   | FAIL (<first failing test name>)
ruff:     PASS            | <count> issues
format:   PASS            | <files needing format>
eval:     PASS (N/N)      | FAIL (case ids)
```

If anything fails, fix it and re-run only the failing gate. Do not report success while any gate is red.
