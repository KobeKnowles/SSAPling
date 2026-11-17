"""Runs: run files, seeds, saved settings, checkpoints, and resuming."""

import asyncio
import logging
import textwrap
from pathlib import Path
from typing import Any

import pytest

from ssapling.core import ConfigError
from ssapling.data import Collections, RecordStore
from ssapling.environments import registry as environments
from ssapling.models import registry as models
from ssapling.tasks import registry as tasks
from ssapling.train import RunContext, Runner
from ssapling.train.context import derive_seed
from tests.fixtures.environments import DigitPadEnv
from tests.fixtures.models import Apprentice, Fragile
from tests.fixtures.tasks import AdditionTask
from tests.fixtures.training import addition


class Noisy(Apprentice):
    """An apprentice that keeps the seed it was built with, as a model with random weights would."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed = seed


tasks.register("addition", AdditionTask)
environments.register("digit_pad", DigitPadEnv)
models.register("apprentice", Apprentice)
models.register("noisy", Noisy)

RUN_FILE = """
components:
  task: {kind: tasks, name: addition}
  sampler:
    kind: samplers
    name: generated
    settings: {vary: digits, values: [1, 2]}
    uses: {source: task}
  collections:
    kind: collections
    settings:
      source: task
      build: {validation: {sampler: sampler, size: 4, seed: 1}}
  environment: {kind: environments, name: digit_pad}
  model: {kind: models, name: apprentice}
  monitoring:
    kind: assessments
    name: standard
    settings:
      collections: [validation]
      aggregations: {accuracy: {measurement: score, aggregator: mean}}
    uses: {collections: collections, task: task}
procedure:
  - stage: interact
    settings: {learn: {after: episode}}
  - stage: assess
    every: {episodes: 2}
    settings: {assessment: monitoring}
  - stage: checkpoint
    every: {episodes: 2}
stop: {episodes: 6}
"""


@pytest.fixture
def run_file(tmp_path: Path) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(RUN_FILE)
    return path


def run(runner: Runner) -> RunContext:
    return asyncio.run(runner.run())


class Idle:
    """A controller that stops at once."""

    def decide(self, context: RunContext, last_error: Exception | None) -> None:
        return None


def built(runner: Runner) -> RunContext:
    """Build ``runner``'s run, and stop before any stage runs."""
    runner.controller = Idle()
    return run(runner)


def steps(store: RecordStore, record_id: str) -> Any:
    """What happened in a record, without its id."""
    record = store.read(record_id)
    return record.header.instance_id, record.steps, record.summary


# Run files


def test_a_run_file_builds_the_components_and_the_run_writes_to_its_output(
    run_file: Path, tmp_path: Path
):
    context = run(Runner.from_config(run_file, seed=7, output=tmp_path / "run"))
    store = RecordStore(tmp_path / "run")
    assert context.counters["episodes"] == 6
    assert isinstance(context.components["collections"], Collections)
    assert store.read("episode-0").header.provenance.source == "addition"  # type: ignore[union-attr]
    # Each pass: interact, then every two episodes, assess and checkpoint, by stage number.
    assert store.document_names() == (
        "checkpoints/4",
        "checkpoints/8",
        "decisions",
        "results/monitoring-3",
        "results/monitoring-7",
        "run",
    )
    assert context.read("monitoring.accuracy") == 1.0  # learned on the way
    assert context.components["environment"].closed  # what the run built, it closes


def test_problems_in_a_run_file_are_reported_together_with_the_file(tmp_path: Path):
    path = tmp_path / "run.yaml"
    path.write_text(
        textwrap.dedent(
            """
            components:
              task: {kind: tasks, name: additon}
              sampler: {kind: samplers, name: generated, settings: {vary: 3}, uses: {source: tsk}}
              collections: {kind: collections, settings: {source: task, build: {v: {size: 1}}}}
              model: {kind: modles, name: apprentice}
            procedure: []
            stop: {episodes: 1}
            """
        )
    )
    with pytest.raises(ConfigError) as caught:
        Runner.from_config(path, seed=1)
    assert caught.value.subject == f"run file {path}"
    assert [(p.path, p.message[:40]) for p in caught.value.problems] == [
        ("components.model.kind", "expected one of 'tasks', 'teachers', 'mo"),
    ]
    path.write_text(path.read_text().replace("modles", "models"))
    with pytest.raises(ConfigError) as caught:
        Runner.from_config(path, seed=1)
    assert [(p.path, p.message[:40]) for p in caught.value.problems] == [
        ("components.task.name", "Unknown task 'additon'. Did you mean 'ad"),
        ("components.sampler.settings.vary", "expected str, got int 3"),
        ("components.collections.settings.build.v.sampler", "missing required setting"),
        ("components.collections.settings.build.v.seed", "missing required setting"),
    ]


def test_collections_are_built_as_the_run_file_says():
    procedure = {
        "components": {
            "task": {"kind": "tasks", "name": "addition"},
            "uniform": {"kind": "samplers", "name": "generated", "uses": {"source": "task"}},
            "collections": {
                "kind": "collections",
                "settings": {
                    "source": "task",
                    "build": {
                        "validation": {"sampler": "uniform", "size": 3, "seed": 1},
                        "test": {"sampler": "uniform", "size": 2, "seed": 2, "held_out": True},
                    },
                },
            },
        },
        "procedure": [],
        "stop": {"stages": 1},
    }
    collections = built(Runner(procedure, seed=1)).components["collections"]
    assert collections.names() == ("validation", "test")
    assert len(collections.get("validation")) == 3
    assert collections.held_out("test")
    assert not collections.held_out("validation")


@pytest.mark.parametrize(
    ("entry", "problem"),
    [
        ({"kind": "collections", "name": "mine"}, "collections have no name or uses"),
        ({"kind": "collections", "uses": {"source": "task"}}, "collections have no name or uses"),
        ({"kind": "models"}, "name the registered model"),
    ],
)
def test_a_component_entry_says_what_to_build(entry: Any, problem: str):
    procedure = {"components": {"thing": entry}, "procedure": [], "stop": {"stages": 1}}
    with pytest.raises(ConfigError, match=problem):
        Runner(procedure, seed=1)


def test_a_role_is_given_once():
    procedure = {
        "components": {"model": {"kind": "models", "name": "apprentice"}},
        "procedure": [],
        "stop": {"stages": 1},
    }
    with pytest.raises(ConfigError, match="'model' is given as a component already"):
        Runner(procedure, {"model": Apprentice()}, seed=1)
    with pytest.raises(ValueError, match="give it as output="):
        Runner({"procedure": [], "stop": {"stages": 1}}, {"store": None}, seed=1)


def test_a_component_uses_only_components_built_before_it(tmp_path: Path):
    procedure = {
        "components": {
            "sampler": {"kind": "samplers", "name": "uniform", "uses": {"source": "task"}},
            "task": {"kind": "tasks", "name": "addition"},
        },
        "procedure": [],
        "stop": {"episodes": 1},
    }
    with pytest.raises(ConfigError, match="'task' is not a component built before it"):
        Runner(procedure, seed=1)


# Seeds and saved settings


def test_component_seeds_are_derived_from_the_root_seed_and_saved(tmp_path: Path):
    procedure = {
        "components": {"model": {"kind": "models", "name": "noisy"}},
        "procedure": [],
        "stop": {"stages": 1},
    }
    context = built(Runner(procedure, seed=7, output=tmp_path))
    seed = derive_seed(7, "component-model")
    assert context.components["model"].seed == seed
    assert RecordStore(tmp_path).load_document("run")["seeds"] == {"model": seed}


def test_a_run_is_reproducible_from_its_root_seed(run_file: Path, tmp_path: Path):
    def records(seed: int, name: str) -> list[Any]:
        run(Runner.from_config(run_file, seed=seed, output=tmp_path / name))
        store = RecordStore(tmp_path / name)
        return [steps(store, f"episode-{number}") for number in range(6)]

    first = records(7, "first")
    assert records(7, "again") == first
    assert records(8, "other") != first


def test_a_run_can_be_rebuilt_from_the_settings_it_saved(run_file: Path, tmp_path: Path):
    first = run(Runner.from_config(run_file, seed=7, output=tmp_path / "first"))
    saved = RecordStore(tmp_path / "first").load_document("run")
    again = run(Runner(saved["procedure"], seed=saved["seed"], output=tmp_path / "again"))
    assert again.counters == first.counters
    assert again.values == first.values
    stores = RecordStore(tmp_path / "first"), RecordStore(tmp_path / "again")
    assert steps(stores[0], "episode-5") == steps(stores[1], "episode-5")


def test_a_second_run_into_the_same_output_is_refused(run_file: Path, tmp_path: Path):
    run(Runner.from_config(run_file, seed=7, output=tmp_path / "run"))
    with pytest.raises(ValueError, match="already holds a document 'run'") as caught:
        run(Runner.from_config(run_file, seed=7, output=tmp_path / "run"))
    assert "Resume that run with Runner.resume" in caught.value.__notes__[0]


# Checkpoints and resuming


CURRICULUM = {
    "procedure": [
        {"stage": "interact", "settings": {"learn": {"after": "episode"}}},
        {"stage": "assess", "settings": {"assessment": "monitoring"}},
        {
            "stage": "update_sampler",
            "settings": {"weights": {"2": 1}},
            "when": {"value": "monitoring.accuracy", "at_least": 1},
        },
        {"stage": "checkpoint", "every": {"episodes": 2}},
    ],
    "stop": {"episodes": 8},
}


def test_resuming_after_a_failure_gives_the_same_result_as_never_stopping(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    never_stopped = run(Runner(CURRICULUM, addition(), seed=7, output=tmp_path / "whole"))

    with pytest.raises(RuntimeError, match="the machine went down"):
        run(Runner(CURRICULUM, addition(Fragile(fail_at=5)), seed=7, output=tmp_path / "broken"))
    with caplog.at_level(logging.INFO, logger="ssapling.train"):
        resumed = run(Runner.resume(tmp_path / "broken", components=addition()))

    assert resumed.counters == never_stopped.counters
    assert resumed.values == never_stopped.values  # each stamped with the same counters
    for role in ("model", "sampler"):
        state = resumed.components[role].export_state()
        assert state == never_stopped.components[role].export_state()
    whole, broken = RecordStore(tmp_path / "whole"), RecordStore(tmp_path / "broken")
    assert broken.load_document("resumes/1") == {"checkpoint": "checkpoints/11"}
    assert steps(broken, "resume-1/episode-7") == steps(whole, "episode-7")
    assert steps(broken, "episode-5") == steps(whole, "episode-5")  # written before it failed
    assert "resumed from checkpoints/11" in caplog.messages
    assert (
        "These components do not export their state, so they start as built: "
        "environment, collections, monitoring" in caplog.messages
    )


def test_a_run_resumed_twice_continues_from_its_latest_checkpoint(tmp_path: Path):
    output = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run(Runner(CURRICULUM, addition(Fragile(fail_at=3)), seed=7, output=output))
    with pytest.raises(RuntimeError):
        run(Runner.resume(output, components=addition(Fragile(fail_at=5))))
    last = run(Runner.resume(output, components=addition()))
    store = RecordStore(output)
    assert store.load_document("resumes/1") == {"checkpoint": "checkpoints/5"}
    assert store.load_document("resumes/2") == {"checkpoint": "resume-1/checkpoints/11"}
    assert last.counters["episodes"] == 8
    assert last.session == 2


def test_a_run_without_a_checkpoint_resumes_from_the_start(tmp_path: Path):
    output = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run(Runner(CURRICULUM, addition(Fragile(fail_at=1)), seed=7, output=output))
    resumed = run(Runner.resume(output, components=addition()))
    store = RecordStore(output)
    assert store.load_document("resumes/1") == {"checkpoint": None}
    assert steps(store, "resume-1/episode-0")[0] == steps(store, "episode-0")[0]
    assert resumed.counters["episodes"] == 8


def test_resuming_needs_what_the_run_was_given(tmp_path: Path):
    run(Runner(CURRICULUM, addition(), seed=7, output=tmp_path))
    with pytest.raises(ValueError, match="give the run its components again: 'collections', "):
        Runner.resume(tmp_path)


class Amnesiac(Apprentice):
    """An apprentice that saves its state, but cannot take it back."""

    restore_state = None  # type: ignore[assignment]


def test_resuming_refuses_a_component_that_saved_its_state_but_cannot_restore_it(tmp_path: Path):
    output = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run(Runner(CURRICULUM, addition(Fragile(fail_at=5)), seed=7, output=output))
    with pytest.raises(TypeError, match="'model' saved its state, but has no restore_state"):
        run(Runner.resume(output, components=addition(Amnesiac())))


def test_the_running_time_carries_on_across_a_resumption():
    context = RunContext({}, seed=1)
    context.restore_state({"counters": context.counters, "seconds": 3600.0, "values": {}})
    assert 3600 <= context.seconds < 3700


def test_a_checkpoint_needs_a_store():
    procedure = {"procedure": [{"stage": "checkpoint"}], "stop": {"stages": 1}}
    with pytest.raises(ConfigError, match="requires 'store'"):
        run(Runner(procedure, seed=1))


def test_components_given_as_they_are_are_not_closed(tmp_path: Path):
    given = addition()
    run(Runner(CURRICULUM, given, seed=7, output=tmp_path))
    assert not given["environment"].closed


def test_derive_seed_is_public_and_is_the_runs_rule():
    """Given the root seed 7, the exported derive_seed gives the run context's seed for a name,
    by the documented rule. Catches: the exported function drifting from the run's own rule,
    so a project's named sub-seeds would differ from the run's."""
    import random

    from ssapling.train import derive_seed as exported

    expected = random.Random("7:component-model").getrandbits(32)
    assert exported(7, "component-model") == expected
    assert RunContext({}, seed=7).seed("component-model") == expected
