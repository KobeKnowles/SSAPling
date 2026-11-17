# Developing SSAPling

## Setup

The project uses [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync                      # creates .venv with ssapling (editable) and the dev tools
uv run pytest                # all tests
uv run ruff check .          # lint
uv run ruff format .         # format (CI runs it with --check)
uv run mypy                  # type check src/ and examples/ (strict) and tests/ (helpers may be untyped)
```

`uv.lock` pins the development environment, and CI installs from it with
`--locked`. After changing dependencies in `pyproject.toml`, run `uv lock`.

CI tests every supported Python version, and the standard library does change
between them. For example, `json` error locations moved in 3.13. Run the tests
on each version before pushing. uv downloads any missing Python, and
`--isolated` leaves `.venv` untouched:

```bash
for v in 3.12 3.13 3.14; do uv run --isolated --python $v --locked pytest -q; done
```

## What lives where

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | Build (hatchling), metadata, the `yaml` extra, the `dev` dependency group, and settings for pytest, ruff and mypy. |
| `src/ssapling/` | The installed package, and nothing else. |
| `tests/` | Tests. `tests/fixtures/` holds tiny toy components (tasks, teachers, models, environments with native counterparts, collators, stages) used only by tests. |
| `tests/test_layering.py` | Enforces the import rules between areas (see `docs/overview.md`). |
| `tests/packaging/` | Checks the packaging commitments: no required dependencies, stdlib-only imports, `py.typed` shipped. |
| `tests/test_examples.py` | Runs each example as its README says, so the examples cannot go stale. The usage guide quotes the arithmetic example by hand: keep the two in step when changing either. |
| `tests/test_docs.py` | Checks that every link in the README, the docs and the examples' READMEs leads to a file and heading that exist, including the README's full GitHub addresses. |
| `examples/` | Runnable examples that use only the public interface. Each has a README. |
| `docs/` | The overview, the usage and extension guides, one explainer per area, and this file. |
| `.github/workflows/ci.yml` | CI: lint and type check, tests on Python 3.12–3.14 × Linux/macOS/Windows, and a wheel check in a clean environment. |
| `.github/workflows/release.yml` | Publishes to PyPI when a version tag is pushed. |

The version is defined once, as `__version__` in `src/ssapling/__init__.py`.

## The two structural tests

**Layering** (`tests/test_layering.py`). `ALLOWED` lists, for each area, the
module prefixes it may import. The test parses each module with `ast`, so
nothing is executed. It resolves relative imports and also counts imports made
only for type checking. It reports `from x import y` as `x.y`, so a rule like
`"teachers.types"` allows `from ssapling.teachers.types import TeacherResponse`
but rejects `from ssapling.teachers import TeacherResponse`. The test also
checks itself against examples of allowed and forbidden imports, and fails if
an area exists on disk without rules.

**Installation** (`tests/packaging/check_install.py`). This standalone script
imports every area in a fresh interpreter. It fails if any module outside the
standard library was loaded, or if `py.typed` is missing. The test suite runs
it against the development install. CI builds the wheel, installs it into an
empty environment with no extras, and runs the same script there.

## Adding to an area

1. Put the code in the area's package. Use separate modules for contracts
   (Protocols), types (small dataclasses), and machinery.
2. Export public names from the area's `__init__.py`. Anything not exported is
   internal.
3. Import other areas only as `docs/overview.md` allows. The layering test will
   tell you if you didn't.
4. Add tests under `tests/<area>/` and an entry under "Unreleased" in
   `CHANGELOG.md`.

## Releasing

The README is also the project's page on PyPI, where relative links do not
work, so its links are full GitHub addresses. `tests/test_docs.py` checks them
against the files here.

1. Set `__version__` in `src/ssapling/__init__.py`, and turn "Unreleased" in
   `CHANGELOG.md` into the version and date.
2. Merge to `main`, with CI passing.
3. Tag the merge and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

`.github/workflows/release.yml` then checks that the tag matches
`__version__`, runs the tests, builds the sdist and wheel, checks the wheel in
a clean environment, and publishes them to PyPI through trusted publishing,
so no token is stored anywhere. Publishing runs in the GitHub environment
`pypi`; a required reviewer on that environment makes each release wait for
approval. A version on PyPI can never be replaced, only yanked, so a mistake
is fixed with a new version.
