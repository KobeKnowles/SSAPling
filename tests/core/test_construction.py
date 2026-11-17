"""Construction: create(), create_from(), validate(), and the catalogue."""

import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_type

import jsonschema
import pytest

from ssapling.core import (
    ComponentSpec,
    ConfigError,
    Registry,
    UnknownComponentError,
    load_config,
    setting,
)


@dataclass(frozen=True)
class StepwiseSettings:
    explain: bool = setting(False, description="Explain each correction.")
    max_interventions: int = setting(10, ge=0)


class StepwiseTeacher:
    """Reviews each step of the student's work.

    This second paragraph is left out of the catalogue.
    """

    Settings = StepwiseSettings

    def __init__(self, config: StepwiseSettings, task: object) -> None:
        self.config = config
        self.task = task


class EchoEnvironment:
    """Echoes actions back as observations."""

    def __init__(self, size: int = 1) -> None:
        self.size = size


@dataclass(frozen=True)
class Scripted:
    script: tuple[str, ...]


@dataclass(frozen=True)
class ScriptedSettings:
    script: tuple[str, ...] = ("hint",)


def make_scripted(config: ScriptedSettings) -> Scripted:
    return Scripted(config.script)


class Failing:
    def __init__(self) -> None:
        raise RuntimeError("cannot connect")


@pytest.fixture
def teachers() -> Registry[type[StepwiseTeacher]]:
    registry: Registry[type[StepwiseTeacher]] = Registry("teacher")
    registry.register("stepwise", StepwiseTeacher)
    return registry


@pytest.fixture
def environments() -> Registry[type[EchoEnvironment]]:
    registry: Registry[type[EchoEnvironment]] = Registry("environment")
    registry.register("echo", EchoEnvironment)
    return registry


# create()


def test_create_validates_settings_and_passes_live_objects(
    teachers: Registry[type[StepwiseTeacher]],
):
    task = object()
    teacher = teachers.create("stepwise", {"explain": True}, task=task)
    assert_type(teacher, StepwiseTeacher)
    assert teacher.config == StepwiseSettings(explain=True, max_interventions=10)
    assert teacher.task is task


def test_create_accepts_keyword_arguments_as_in_the_design_doc(teachers):
    teacher = teachers.create(name="stepwise", settings={"max_interventions": 3}, task=None)
    assert teacher.config.max_interventions == 3


def test_create_without_settings_uses_defaults(teachers):
    assert teachers.create("stepwise", task=None).config == StepwiseSettings()


def test_create_accepts_a_settings_instance(teachers):
    settings = StepwiseSettings(explain=True)
    assert teachers.create("stepwise", settings, task=None).config == settings


def test_component_without_settings_gets_live_objects_only(environments):
    assert environments.create("echo", size=3).size == 3
    assert environments.create("echo", {}).size == 1


def test_settings_for_a_component_that_takes_none_are_rejected(environments):
    with pytest.raises(ConfigError) as caught:
        environments.create("echo", {"size": 3})
    assert str(caught.value) == (
        "Invalid settings for environment 'echo':\n"
        "  - environment 'echo' takes no settings, got dict {'size': 3}"
    )


def test_invalid_settings_name_the_component(teachers):
    with pytest.raises(ConfigError) as caught:
        teachers.create("stepwise", {"max_interventions": -1}, task=None)
    assert str(caught.value) == (
        "Invalid settings for teacher 'stepwise':\n  - max_interventions: must be >= 0, got -1"
    )


def test_unknown_component_is_reported(teachers):
    with pytest.raises(UnknownComponentError, match=r"^Unknown teacher 'stepwize'\. Did you mean"):
        teachers.create("stepwize", task=None)


def test_config_is_reserved_for_the_validated_settings(teachers):
    with pytest.raises(TypeError, match="'config' cannot be passed as a live object"):
        teachers.create("stepwise", config=StepwiseSettings(), task=None)


def test_errors_from_the_implementation_carry_a_note(teachers):
    failing: Registry[type[Failing]] = Registry("environment")
    failing.register("remote", Failing)
    with pytest.raises(RuntimeError, match="cannot connect") as caught:
        failing.create("remote")
    assert caught.value.__notes__ == [
        "While creating environment 'remote' (tests.core.test_construction.Failing)."
    ]

    with pytest.raises(TypeError) as missing_live:
        teachers.create("stepwise")  # no task
    assert "While creating teacher 'stepwise'" in missing_live.value.__notes__[0]


def test_functions_can_build_components():
    factories: Registry[Callable[..., Scripted]] = Registry("teacher")
    factories.register("scripted", make_scripted, settings=ScriptedSettings)
    built = factories.create("scripted", {"script": ["a", "b"]})
    assert_type(built, Scripted)
    assert built == Scripted(("a", "b"))


# validate()


def test_validate_checks_settings_without_creating(teachers):
    assert teachers.validate("stepwise", {"explain": True}) == StepwiseSettings(explain=True)
    with pytest.raises(ConfigError):
        teachers.validate("stepwise", {"explain": "yes"})


def test_validate_returns_none_for_a_component_without_settings(environments):
    assert environments.validate("echo") is None


# create_from(): entries from files


@pytest.fixture
def run_yaml(tmp_path):
    def write(text: str) -> str:
        path = tmp_path / "run.yaml"
        path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
        return str(path)

    return write


def test_create_from_builds_an_entry_of_a_loaded_file(teachers, run_yaml):
    path = run_yaml(
        """
        teacher:
          name: stepwise
          settings: {explain: true}
        """
    )
    teacher = teachers.create_from(load_config(path).component("teacher"), task=None)
    assert teacher.config == StepwiseSettings(explain=True)


def test_invalid_settings_from_a_file_name_the_file_and_the_full_path(teachers, run_yaml):
    path = run_yaml(
        """
        components:
          teacher:
            name: stepwise
            settings: {max_interventions: -1, explian: true}
        """
    )
    spec = load_config(path).component("components.teacher")
    with pytest.raises(ConfigError) as caught:
        teachers.create_from(spec, task=None)
    assert str(caught.value) == (
        f"Invalid settings for teacher 'stepwise' in {path}:\n"
        "  - components.teacher.settings.explian: unknown setting; did you mean 'explain'? "
        "(known: explain, max_interventions)\n"
        "  - components.teacher.settings.max_interventions: must be >= 0, got -1"
    )


def test_unknown_name_from_a_file_names_the_file_and_entry(teachers, run_yaml):
    path = run_yaml("teacher: {name: stepwize}")
    with pytest.raises(UnknownComponentError) as caught:
        teachers.create_from(load_config(path).component("teacher"), task=None)
    assert caught.value.location == f"{path}: teacher.name"
    assert str(caught.value).startswith(
        f"{path}: teacher.name: Unknown teacher 'stepwize'. Did you mean 'stepwise'?"
    )


def test_implementation_errors_from_a_file_name_the_entry(run_yaml):
    failing: Registry[type[Failing]] = Registry("environment")
    failing.register("remote", Failing)
    path = run_yaml("env: {name: remote}")
    with pytest.raises(RuntimeError) as caught:
        failing.create_from(load_config(path).component("env"))
    assert caught.value.__notes__ == [
        f"While creating environment 'remote' (tests.core.test_construction.Failing) "
        f"for {path}: env."
    ]


def test_specs_built_in_python_report_plain_paths(teachers):
    with pytest.raises(ConfigError) as caught:
        teachers.validate_from(ComponentSpec("stepwise", {"max_interventions": -1}))
    assert [str(p) for p in caught.value.problems] == [
        "settings.max_interventions: must be >= 0, got -1"
    ]
    with pytest.raises(UnknownComponentError) as unknown:
        teachers.validate_from(ComponentSpec("stepwize"))
    assert unknown.value.location is None


@dataclass(frozen=True)
class RangeSettings:
    low: int = 0
    high: int = 1

    def validate(self) -> None:
        if self.low > self.high:
            raise ValueError("low must not exceed high")


def test_a_rule_over_all_the_settings_is_reported_at_the_settings_path():
    ranges: Registry[Callable[..., RangeSettings]] = Registry("sampler")
    ranges.register("range", lambda config: config, settings=RangeSettings)
    with pytest.raises(ConfigError) as caught:
        ranges.validate_from(ComponentSpec("range", {"low": 2}, path="sampler"))
    assert [str(p) for p in caught.value.problems] == ["sampler.settings: low must not exceed high"]


# Catalogue


def test_catalogue_describes_each_component(teachers, environments):
    teachers.register("echo_teacher", EchoEnvironment, description="Repeats the student.")
    catalogue = teachers.catalogue()
    assert [entry["name"] for entry in catalogue] == ["echo_teacher", "stepwise"]
    stepwise = catalogue[1]
    assert stepwise["description"] == "Reviews each step of the student's work."
    assert stepwise["settings"]["properties"]["explain"]["description"] == (
        "Explain each correction."
    )
    assert catalogue[0]["description"] == "Repeats the student."
    assert catalogue[0]["settings"]["properties"] == {}


def test_catalogue_is_json_ready_and_its_schemas_are_valid(teachers, environments):
    for entry in [*teachers.catalogue(), *environments.catalogue()]:
        json.dumps(entry)
        jsonschema.Draft202012Validator.check_schema(entry["settings"])
    no_settings = environments.catalogue()[0]["settings"]
    validator = jsonschema.Draft202012Validator(no_settings)
    assert validator.is_valid({})
    assert not validator.is_valid({"size": 3})


def test_registration_records_the_description(teachers):
    assert teachers.entry("stepwise").description == "Reviews each step of the student's work."


def test_registering_again_keeps_the_first_description_unless_replaced(teachers):
    teachers.register("stepwise", StepwiseTeacher, description="Something else.")
    assert teachers.entry("stepwise").description == "Reviews each step of the student's work."
    teachers.register("stepwise", StepwiseTeacher, description="Something else.", replace=True)
    assert teachers.entry("stepwise").description == "Something else."
