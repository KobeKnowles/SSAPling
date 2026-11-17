"""Enforces the import rules between package areas (see docs/overview.md).

Each area may import only itself and the modules listed for it in ``ALLOWED``.
The check is static: it parses every module under ``src/ssapling`` with ``ast``
and inspects its import statements, including relative imports and imports
made only for type checking. Imports from outside ``ssapling`` are not checked
here; tests/packaging checks that the package uses only the standard library.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "ssapling"

AREAS = (
    "core",
    "tasks",
    "teachers",
    "models",
    "environments",
    "data",
    "interaction",
    "evaluate",
    "train",
)

# Module prefixes, relative to ``ssapling``, that each area may import besides
# itself. A prefix covers that module and its submodules only: "teachers.types"
# allows ``ssapling.teachers.types`` but not the rest of ``ssapling.teachers``.
ALLOWED: dict[str, tuple[str, ...]] = {
    "core": (),
    "tasks": ("core",),
    "teachers": ("core",),
    "models": ("core",),
    "environments": ("core", "teachers.types"),
    "data": ("core", "tasks", "environments.types", "teachers.types"),
    "interaction": ("core", "tasks", "environments", "teachers", "models", "data"),
    "evaluate": ("core", "tasks", "environments", "teachers", "models", "data", "interaction"),
    "train": (
        "core",
        "tasks",
        "environments",
        "teachers",
        "models",
        "data",
        "interaction",
        "evaluate",
    ),
}


def imported_names(source: str, module: str, *, is_package: bool) -> list[tuple[int, str]]:
    """Return ``(line, absolute name)`` for every import in a module's source.

    ``from x import y`` is reported as ``x.y``, because ``y`` may be a
    submodule; a prefix rule on ``x`` still matches it.
    """
    package = module if is_package else module.rpartition(".")[0]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                if node.module:
                    base = f"{base}.{node.module}"
            else:
                base = node.module or ""
            found += [(node.lineno, f"{base}.{alias.name}") for alias in node.names]
    return found


def is_forbidden(area: str, name: str) -> bool:
    """Whether a module in ``area`` may not import ``name``."""
    if name != "ssapling" and not name.startswith("ssapling."):
        return False
    target = name.removeprefix("ssapling").lstrip(".")
    allowed = (area, *ALLOWED[area])
    return not any(target == prefix or target.startswith(prefix + ".") for prefix in allowed)


def violations(area: str) -> list[str]:
    found = []
    for path in sorted((PACKAGE_ROOT / area).rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
        is_package = relative.name == "__init__"
        module = ".".join(relative.parent.parts if is_package else relative.parts)
        for line, name in imported_names(path.read_text(), module, is_package=is_package):
            if is_forbidden(area, name):
                found.append(f"{module} (line {line}) imports {name}")
    return found


def test_every_area_has_rules():
    on_disk = {p.name for p in PACKAGE_ROOT.iterdir() if (p / "__init__.py").is_file()}
    assert on_disk == set(AREAS) == set(ALLOWED)


@pytest.mark.parametrize("area", AREAS)
def test_area_respects_import_rules(area):
    assert violations(area) == []


# A types module that other areas may import holds data, not behaviour, so it
# must not import the rest of its own area (its runner, say). Each is checked
# once its stage has built it.
SHARED_TYPES = sorted(
    module
    for module in {prefix for prefixes in ALLOWED.values() for prefix in prefixes} - set(AREAS)
    if (PACKAGE_ROOT / f"{module.replace('.', '/')}.py").is_file()
)


@pytest.mark.parametrize("module", SHARED_TYPES)
def test_a_shared_types_module_imports_nothing_else_from_its_area(module):
    path = PACKAGE_ROOT / f"{module.replace('.', '/')}.py"
    area = module.split(".")[0]
    names = imported_names(path.read_text(), f"ssapling.{module}", is_package=False)
    own_area = [
        name
        for _, name in names
        if name.startswith(f"ssapling.{area}.") and not name.startswith(f"ssapling.{module}")
    ]
    assert own_area == []


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("ssapling.environments.adapter", "import ssapling.teachers.runner"),
        ("ssapling.environments.adapter", "from ..teachers import runner"),
        ("ssapling.environments.adapter", "from ssapling.teachers import TeacherResponse"),
        ("ssapling.core.registry", "from ssapling.tasks import Instance"),
        ("ssapling.core", "from .. import registries"),
        ("ssapling.data.records", "import ssapling"),
    ],
)
def test_checker_flags_forbidden_imports(module, source):
    area = module.split(".")[1]
    is_package = module.count(".") == 1
    names = imported_names(source, module, is_package=is_package)
    assert any(is_forbidden(area, name) for _, name in names)


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("ssapling.environments.adapter", "from ssapling.teachers.types import TeacherResponse"),
        ("ssapling.environments.adapter", "from ..teachers.types import TeacherResponse"),
        ("ssapling.environments.adapter", "from . import types"),
        ("ssapling.data.records", "from ssapling.tasks import Instance"),
        ("ssapling.train.runner", "from ssapling.evaluate import Assessment"),
        ("ssapling.core.registry", "import dataclasses\nfrom typing import Any"),
    ],
)
def test_checker_allows_permitted_imports(module, source):
    area = module.split(".")[1]
    names = imported_names(source, module, is_package=False)
    assert not any(is_forbidden(area, name) for _, name in names)
