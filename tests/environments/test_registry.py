"""Environments as components, built through Core."""

from typing import TYPE_CHECKING, Any, assert_type

import pytest

from ssapling import registries
from ssapling.core import Registry
from ssapling.environments import Environment, registry
from tests.fixtures.environments import DigitPadEnv, NimEnv
from tests.fixtures.tasks import AdditionTask
from tests.fixtures.teachers import DigitTutor

# The environment registry builds Environments. mypy checks this; it never
# runs, so the shared registry is left empty.
if TYPE_CHECKING:
    assert_type(registry.create("digit_pad"), Environment)


@pytest.fixture
def local() -> Registry[Any]:
    local: Registry[Any] = Registry("environment")
    local.register("digit_pad", DigitPadEnv)
    local.register("nim", NimEnv)
    return local


def test_environments_are_built_through_core_with_live_objects(local):
    assert local.create("nim", stones=7).native.start == 7
    tutor = DigitTutor(task=AdditionTask())
    assert local.create("digit_pad", teacher=tutor).teacher is tutor


def test_the_environment_registry_is_collected_in_ssapling_registries():
    assert registries.environments is registry
    assert registry.kind == "environment"


def test_catalogue_describes_an_environment(local):
    entries = {entry["name"]: entry for entry in local.catalogue()}
    assert entries["nim"]["description"] == (
        "Adapts ``NativeNim``: two agents taking turns, saying whose turn it is."
    )
