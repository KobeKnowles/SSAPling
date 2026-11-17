"""Registry: registration, lookup, unknown names, and duplicate names."""

from dataclasses import dataclass, field

import pytest

from ssapling import SSAPlingError
from ssapling.core import (
    DuplicateRegistrationError,
    Registry,
    SettingsDeclarationError,
    UnknownComponentError,
)


class StepwiseTeacher:
    pass


class ScriptedTeacher:
    pass


def count_steps(record):
    return len(record)


@pytest.fixture
def teachers():
    registry: Registry[type] = Registry("teacher")
    registry.register("stepwise", StepwiseTeacher)
    registry.register("scripted", ScriptedTeacher)
    return registry


# Registration and lookup


def test_get_returns_the_registered_class_itself_not_an_instance(teachers):
    assert teachers.get("stepwise") is StepwiseTeacher


def test_functions_can_be_registered():
    measurements: Registry[object] = Registry("measurement")
    measurements.register("steps", count_steps)
    assert measurements.get("steps") is count_steps


def test_names_lists_every_registered_name_sorted(teachers):
    assert teachers.names() == ("scripted", "stepwise")


def test_contains_and_len(teachers):
    assert "stepwise" in teachers
    assert "unknown" not in teachers
    assert len(teachers) == 2


def test_each_kind_has_its_own_registry(teachers):
    samplers: Registry[type] = Registry("sampler")
    assert "stepwise" not in samplers
    assert samplers.kind == "sampler"
    assert teachers.kind == "teacher"


def test_one_implementation_may_be_registered_under_several_names(teachers):
    teachers.register("stepwise_v1", StepwiseTeacher)
    assert teachers.get("stepwise_v1") is teachers.get("stepwise")


# Unknown names


def test_unknown_name_suggests_close_matches(teachers):
    with pytest.raises(UnknownComponentError) as caught:
        teachers.get("stepwize")
    error = caught.value
    assert error.suggestions == ("stepwise",)
    assert str(error) == (
        "Unknown teacher 'stepwize'. Did you mean 'stepwise'? "
        "Registered teacher names: 'scripted' and 'stepwise'."
    )


def test_unknown_name_without_close_match_lists_registered_names(teachers):
    with pytest.raises(UnknownComponentError) as caught:
        teachers.get("llm_judge")
    assert caught.value.suggestions == ()
    assert str(caught.value) == (
        "Unknown teacher 'llm_judge'. Registered teacher names: 'scripted' and 'stepwise'."
    )


def test_unknown_name_in_empty_registry_points_to_registration():
    with pytest.raises(UnknownComponentError, match=r"register_components\(\)"):
        Registry("teacher").get("stepwise")


def test_unknown_name_error_carries_structured_details(teachers):
    with pytest.raises(UnknownComponentError) as caught:
        teachers.get("stepwize")
    error = caught.value
    assert (error.kind, error.name, error.available) == (
        "teacher",
        "stepwize",
        ("scripted", "stepwise"),
    )


def test_unknown_name_error_is_an_ssapling_error_and_a_lookup_error(teachers):
    with pytest.raises(SSAPlingError):
        teachers.get("missing")
    with pytest.raises(LookupError):
        teachers.get("missing")


# Duplicate names


def test_registering_a_different_implementation_under_a_taken_name_is_rejected(teachers):
    with pytest.raises(DuplicateRegistrationError) as caught:
        teachers.register("stepwise", ScriptedTeacher)
    message = str(caught.value)
    assert "StepwiseTeacher" in message
    assert "ScriptedTeacher" in message
    assert "replace=True" in message
    assert teachers.get("stepwise") is StepwiseTeacher


def test_duplicate_error_is_an_ssapling_error_and_a_value_error(teachers):
    with pytest.raises(SSAPlingError):
        teachers.register("stepwise", ScriptedTeacher)
    with pytest.raises(ValueError, match="already registered"):
        teachers.register("stepwise", ScriptedTeacher)


def test_replace_true_replaces_the_registered_implementation(teachers):
    teachers.register("stepwise", ScriptedTeacher, replace=True)
    assert teachers.get("stepwise") is ScriptedTeacher


def test_registering_the_same_implementation_again_is_a_no_op(teachers):
    def register_components() -> None:
        teachers.register("stepwise", StepwiseTeacher)
        teachers.register("scripted", ScriptedTeacher)

    register_components()
    register_components()
    assert teachers.names() == ("scripted", "stepwise")


# Invalid names


@pytest.mark.parametrize("name", ["", " stepwise", "stepwise\n"])
def test_empty_or_padded_names_are_rejected(name):
    registry: Registry[type] = Registry("teacher")
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(name, StepwiseTeacher)


def test_non_string_names_are_rejected():
    registry: Registry[type] = Registry("teacher")
    with pytest.raises(TypeError, match="must be a string"):
        registry.register(3, StepwiseTeacher)  # type: ignore[arg-type]


def test_repr_shows_kind_and_names(teachers):
    assert repr(teachers) == "Registry('teacher', names=['scripted', 'stepwise'])"


# Settings classes


@dataclass(frozen=True)
class StepwiseSettings:
    explain: bool = False


@dataclass(frozen=True)
class OtherSettings:
    explain: bool = True


class TeacherWithSettings:
    Settings = StepwiseSettings


def test_settings_class_is_taken_from_the_settings_attribute():
    registry: Registry[type] = Registry("teacher")
    registry.register("stepwise", TeacherWithSettings)
    entry = registry.entry("stepwise")
    assert (entry.name, entry.implementation, entry.settings) == (
        "stepwise",
        TeacherWithSettings,
        StepwiseSettings,
    )


def test_explicit_settings_class_takes_precedence():
    registry: Registry[type] = Registry("teacher")
    registry.register("stepwise", TeacherWithSettings, settings=OtherSettings)
    assert registry.entry("stepwise").settings is OtherSettings


def test_implementation_without_settings_has_none(teachers):
    assert teachers.entry("stepwise").settings is None


def test_settings_class_is_checked_when_registered():
    @dataclass
    class BrokenSettings:
        tags: set[str] = field(default_factory=set)

    registry: Registry[type] = Registry("teacher")
    with pytest.raises(SettingsDeclarationError, match="tags: unsupported setting type"):
        registry.register("stepwise", StepwiseTeacher, settings=BrokenSettings)
    assert "stepwise" not in registry


def test_same_implementation_and_settings_again_is_a_no_op():
    registry: Registry[type] = Registry("teacher")
    registry.register("stepwise", TeacherWithSettings)
    registry.register("stepwise", TeacherWithSettings)
    registry.register("stepwise", TeacherWithSettings, settings=StepwiseSettings)
    assert len(registry) == 1


def test_same_implementation_with_different_settings_is_rejected():
    registry: Registry[type] = Registry("teacher")
    registry.register("stepwise", TeacherWithSettings)
    with pytest.raises(DuplicateRegistrationError) as caught:
        registry.register("stepwise", TeacherWithSettings, settings=OtherSettings)
    message = str(caught.value)
    assert "settings tests.core.test_registry.StepwiseSettings" in message
    assert "settings tests.core.test_registry.OtherSettings" in message
    assert "replace=True" in message
    assert registry.entry("stepwise").settings is StepwiseSettings

    registry.register("stepwise", TeacherWithSettings, settings=OtherSettings, replace=True)
    assert registry.entry("stepwise").settings is OtherSettings


def test_entry_for_unknown_name_raises(teachers):
    with pytest.raises(UnknownComponentError):
        teachers.entry("stepwize")
