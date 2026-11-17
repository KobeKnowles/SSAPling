"""Models as components: building, persistent state across save and load, and evaluation."""

import json
from typing import TYPE_CHECKING, Any, assert_type

import pytest

from ssapling import registries
from ssapling.core import ConfigError, Registry
from ssapling.models import Model, as_buffered, registry
from tests.fixtures.models import CountingLearner, LookupModel, SlowThinker

# The model registry builds Models. mypy checks this; it never runs, so the
# shared registry is left empty.
if TYPE_CHECKING:
    assert_type(registry.create("lookup"), Model)


@pytest.fixture
def local() -> Registry[Any]:
    local: Registry[Any] = Registry("model")
    local.register("lookup", LookupModel)
    local.register("thinker", SlowThinker)
    local.register("learner", CountingLearner)
    return local


def test_models_are_built_through_core(local):
    model = local.create("lookup", {"table": {"2+2": "4"}, "default": "0"})
    assert model.act("2+2") == "4"
    assert model.act("1+1") == "0"
    with pytest.raises(ConfigError, match="table"):
        local.create("lookup", {"table": ["2+2"]})


def test_models_with_live_dependencies_are_built_too(local):
    assert local.create("thinker", think_steps=4).think_steps == 4


def test_the_model_registry_is_collected_in_ssapling_registries():
    assert registries.models is registry
    assert registry.kind == "model"


def test_catalogue_describes_a_model_and_its_settings(local):
    entries = {entry["name"]: entry for entry in local.catalogue()}
    assert entries["lookup"]["description"] == (
        "Acts by looking each observation up in a table. The simplest model: act only."
    )
    assert entries["lookup"]["settings"]["properties"]["default"]["default"] == "?"


# Persistent state


def test_persistent_state_survives_save_and_load(local, tmp_path):
    learner = local.create("learner")
    for _ in range(3):
        learner.learn("learn_from_batch", batch=["x", "y"])
    path = tmp_path / "learner.json"
    path.write_text(json.dumps(learner.export_state()))  # plain data, as a checkpoint saves it

    restored = local.create("learner")
    restored.restore_state(json.loads(path.read_text()))
    assert restored.export_state() == {"updates": 3, "examples_seen": 6}
    assert restored.act("12") == learner.act("12")


def test_persistent_state_survives_an_episode_reset():
    """Episode state is reset at episode boundaries; persistent state is not."""
    learner = CountingLearner()
    learner.learn("consolidate")
    model: Any = as_buffered(learner)
    model.perceive("12")
    model.reset_episode()
    assert learner.export_state() == {"updates": 1, "examples_seen": 0}


def test_state_is_exported_through_the_wrapper():
    learner = CountingLearner()
    learner.learn("consolidate")
    model: Any = as_buffered(learner)
    assert model.export_state() == {"model": learner.export_state(), "queues": []}
    model.restore_state({"model": {"updates": 5, "examples_seen": 2}, "queues": []})
    assert learner.updates == 5


# Evaluation


def test_a_model_can_be_told_it_is_being_evaluated():
    learner = CountingLearner()
    model: Any = as_buffered(learner)
    model.set_evaluating(True)
    assert learner.evaluating is True
    model.set_evaluating(False)
    assert learner.evaluating is False
