"""Helpers public for downstream code: `Execution`, which calls sync and async functions alike, and
`load_named`, which loads what a `MODULE:NAME` names."""

import asyncio
import collections
import sys
from pathlib import Path

import pytest

from ssapling.core import Execution, load_named


def test_execution_is_public_and_calls_sync_and_async_functions_alike():
    async def double(x: int) -> int:
        return 2 * x

    assert asyncio.run(Execution().call(double, 2)) == 4
    assert asyncio.run(Execution().call(lambda x: x + 1, 2)) == 3


def test_load_named_gives_what_module_name_names():
    assert load_named("collections:OrderedDict", "class") is collections.OrderedDict


def test_load_named_finds_a_module_in_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Given a module in the current directory, which isn't on sys.path, when load_named names
    it, then it is imported from there, as a command run from a project's root needs.
    Catches: a project's own run files unable to name its modules."""
    (tmp_path / "helpers_module_here.py").write_text("VALUE = 3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", str(tmp_path))])
    try:
        assert load_named("helpers_module_here:VALUE", "value") == 3
    finally:
        sys.modules.pop("helpers_module_here", None)


@pytest.mark.parametrize(
    ("spec", "named"),
    [
        ("OrderedDict", "MODULE:NAME"),
        ("nowhere_at_all:thing", "nowhere_at_all"),
        ("collections:Nothing", "Nothing"),
    ],
    ids=["no-colon", "no-module", "no-name"],
)
def test_load_named_refuses_what_it_cannot_load(spec: str, named: str):
    with pytest.raises(ValueError, match=named) as refused:
        load_named(spec, "factory")
    assert "factory" in str(refused.value)
