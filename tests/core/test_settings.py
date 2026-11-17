"""Settings: defaults, type checks, unknown fields, constraints, validate(), schema."""

from dataclasses import dataclass, field
from typing import Any, Literal

import jsonschema
import pytest

from ssapling import SSAPlingError
from ssapling.core import (
    ConfigError,
    Problem,
    SettingsDeclarationError,
    check_settings_class,
    setting,
    settings_schema,
    settings_to_dict,
    validate_settings,
)


@dataclass(frozen=True)
class StepwiseSettings:
    """Settings for a teacher that reviews each step."""

    explain: bool = setting(False, description="Explain each correction.")
    max_interventions: int = setting(10, ge=0, description="Interventions allowed per episode.")


@dataclass(frozen=True)
class Schedule:
    warmup: int = setting(0, ge=0)
    weights: tuple[float, ...] = (1.0,)


@dataclass(frozen=True)
class SamplerSettings:
    name: str = setting(min_length=1, pattern=r"[a-z_]+", description="Required.")
    mode: Literal["uniform", "weighted"] = "uniform"
    temperature: float = setting(1.0, gt=0, le=10)
    limit: int | None = None
    sizes: list[int] = field(default_factory=lambda: [1, 2])
    span: tuple[int, str] = (0, "a")
    weights: dict[str, float] = field(default_factory=dict)
    schedule: Schedule = field(default_factory=Schedule)
    fallback: Schedule | None = None
    params: Any = None

    def validate(self) -> None:
        if self.mode == "weighted" and not self.weights:
            raise ValueError("mode 'weighted' needs at least one entry in weights")


def problems_of(cls: type, values: object) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        validate_settings(cls, values)
    return [str(p) for p in caught.value.problems]


# Defaults and required settings


def test_omitted_settings_take_their_defaults():
    assert validate_settings(StepwiseSettings, {}) == StepwiseSettings()
    assert validate_settings(StepwiseSettings) == StepwiseSettings()


def test_supplied_settings_override_defaults():
    result = validate_settings(StepwiseSettings, {"explain": True, "max_interventions": 5})
    assert result == StepwiseSettings(explain=True, max_interventions=5)


def test_mutable_defaults_are_fresh_each_time():
    first = validate_settings(SamplerSettings, {"name": "a"})
    second = validate_settings(SamplerSettings, {"name": "a"})
    assert first.sizes == second.sizes
    assert first.sizes is not second.sizes


def test_missing_required_setting_is_reported():
    assert problems_of(SamplerSettings, {}) == ["name: missing required setting"]


# Types


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        ({"explain": "yes"}, "explain: expected bool, got str 'yes'"),
        ({"explain": 1}, "explain: expected bool, got int 1"),
        ({"max_interventions": True}, "max_interventions: expected int, got bool True"),
        ({"max_interventions": 2.5}, "max_interventions: expected int, got float 2.5"),
        ({"max_interventions": "3"}, "max_interventions: expected int, got str '3'"),
    ],
)
def test_wrong_types_are_rejected_without_coercion(values, problem):
    assert problems_of(StepwiseSettings, values) == [problem]


def test_int_is_accepted_for_float_and_converted():
    result = validate_settings(SamplerSettings, {"name": "a", "temperature": 2})
    assert result.temperature == 2.0
    assert type(result.temperature) is float


def test_literal_accepts_only_its_values():
    assert validate_settings(SamplerSettings, {"name": "a", "mode": "uniform"}).mode == "uniform"
    assert problems_of(SamplerSettings, {"name": "a", "mode": "greedy"}) == [
        "mode: expected one of 'uniform', 'weighted', got str 'greedy'"
    ]


def test_literal_distinguishes_true_from_one():
    @dataclass
    class Flags:
        level: Literal[1, 2] = 1

    assert problems_of(Flags, {"level": True}) == ["level: expected one of 1, 2, got bool True"]


def test_optional_accepts_none_and_its_type():
    assert validate_settings(SamplerSettings, {"name": "a", "limit": None}).limit is None
    assert validate_settings(SamplerSettings, {"name": "a", "limit": 3}).limit == 3
    assert problems_of(SamplerSettings, {"name": "a", "limit": "3"}) == [
        "limit: expected int, got str '3'"
    ]


def test_sequences_are_converted_to_the_declared_kind():
    result = validate_settings(SamplerSettings, {"name": "a", "sizes": (3, 4), "span": [1, "b"]})
    assert result.sizes == [3, 4]
    assert result.span == (1, "b")


def test_items_are_checked_with_their_position():
    assert problems_of(SamplerSettings, {"name": "a", "sizes": [1, "x", 3]}) == [
        "sizes[1]: expected int, got str 'x'"
    ]
    assert problems_of(SamplerSettings, {"name": "a", "span": [1]}) == [
        "span: expected 2 items, got 1"
    ]


def test_dict_values_and_keys_are_checked():
    assert problems_of(SamplerSettings, {"name": "a", "weights": {"x": "heavy"}}) == [
        "weights.x: expected float, got str 'heavy'"
    ]
    assert problems_of(SamplerSettings, {"name": "a", "weights": {1: 1.0}}) == [
        "weights: keys must be strings, got int 1"
    ]


def test_nested_settings_are_built_from_mappings():
    result = validate_settings(
        SamplerSettings, {"name": "a", "schedule": {"warmup": 5, "weights": [0.5, 0.5]}}
    )
    assert result.schedule == Schedule(warmup=5, weights=(0.5, 0.5))


def test_nested_problems_have_dotted_paths():
    assert problems_of(
        SamplerSettings, {"name": "a", "schedule": {"warmup": -1, "wieghts": []}}
    ) == [
        "schedule.wieghts: unknown setting; did you mean 'weights'? (known: warmup, weights)",
        "schedule.warmup: must be >= 0, got -1",
    ]


def test_optional_nested_settings_report_their_own_problems():
    assert problems_of(SamplerSettings, {"name": "a", "fallback": {"warmup": "soon"}}) == [
        "fallback.warmup: expected int, got str 'soon'"
    ]


def test_any_accepts_plain_values_but_not_live_objects():
    plain = {"depth": [1, 2.5, None, {"x": True}]}
    assert validate_settings(SamplerSettings, {"name": "a", "params": plain}).params == plain

    problems = problems_of(SamplerSettings, {"name": "a", "params": {"env": object()}})
    assert len(problems) == 1
    assert problems[0].startswith("params.env: expected a plain value")


def test_string_annotations_are_resolved():
    @dataclass
    class Deferred:
        count: "int" = 1

    assert problems_of(Deferred, {"count": "x"}) == ["count: expected int, got str 'x'"]


# Unknown settings


def test_unknown_setting_is_reported_with_a_suggestion():
    assert problems_of(StepwiseSettings, {"max_intervention": 3}) == [
        "max_intervention: unknown setting; did you mean 'max_interventions'? "
        "(known: explain, max_interventions)"
    ]


def test_non_mapping_values_are_rejected():
    assert problems_of(StepwiseSettings, [("explain", True)]) == [
        "expected a mapping of StepwiseSettings settings, got list [('explain', True)]"
    ]


# Constraints


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        ({"temperature": 0}, "temperature: must be > 0, got 0.0"),
        ({"temperature": 11}, "temperature: must be <= 10, got 11.0"),
        ({"temperature": float("nan")}, "temperature: must be > 0, got nan"),
        ({"name": ""}, "name: must have at least 1 characters, got 0"),
        ({"name": "Big Name"}, "name: must match the pattern '[a-z_]+', got 'Big Name'"),
    ],
)
def test_constraints_are_enforced(values, problem):
    assert problem in problems_of(SamplerSettings, {"name": "a", **values})


def test_constraints_are_skipped_for_none():
    @dataclass
    class Bounded:
        limit: int | None = setting(None, ge=1)

    assert validate_settings(Bounded, {"limit": None}).limit is None
    assert problems_of(Bounded, {"limit": 0}) == ["limit: must be >= 1, got 0"]


def test_a_maximum_length_allows_its_bound_and_refuses_one_more():
    @dataclass
    class Short:
        code: str = setting("", max_length=3)
        tags: tuple[str, ...] = setting((), max_length=2)
        labels: dict[str, str] = setting(default_factory=dict, max_length=1)

    at_bound = {"code": "abc", "tags": ["a", "b"], "labels": {"x": "y"}}
    assert validate_settings(Short, at_bound).code == "abc"
    assert problems_of(
        Short, {"code": "abcd", "tags": ["a", "b", "c"], "labels": {"x": "", "y": ""}}
    ) == [
        "code: must have at most 3 characters, got 4",
        "tags: must have at most 2 items, got 3",
        "labels: must have at most 1 items, got 2",
    ]


# Every problem at once


def test_every_problem_is_reported_together():
    with pytest.raises(ConfigError) as caught:
        validate_settings(StepwiseSettings, {"explain": "yes", "max_interventions": -1, "x": 1})
    assert str(caught.value) == (
        "Invalid settings for StepwiseSettings:\n"
        "  - x: unknown setting (known: explain, max_interventions)\n"
        "  - explain: expected bool, got str 'yes'\n"
        "  - max_interventions: must be >= 0, got -1"
    )
    assert caught.value.problems[0] == Problem(
        "x", "unknown setting (known: explain, max_interventions)"
    )


def test_config_error_is_an_ssapling_error_and_a_value_error():
    with pytest.raises(SSAPlingError):
        validate_settings(StepwiseSettings, {"explain": "yes"})
    with pytest.raises(ValueError, match="Invalid settings"):
        validate_settings(StepwiseSettings, {"explain": "yes"})


# The class's own validate()


def test_validate_method_reports_rules_across_settings():
    assert problems_of(SamplerSettings, {"name": "a", "mode": "weighted"}) == [
        "mode 'weighted' needs at least one entry in weights"
    ]
    result = validate_settings(
        SamplerSettings, {"name": "a", "mode": "weighted", "weights": {"a": 1}}
    )
    assert result.weights == {"a": 1.0}


def test_validate_method_runs_only_after_field_checks_pass():
    # The field problem is reported; validate() would have failed too, but is not run.
    assert problems_of(SamplerSettings, {"name": "a", "mode": "weighted", "limit": "x"}) == [
        "limit: expected int, got str 'x'"
    ]


def test_validate_method_of_nested_settings_reports_its_path():
    @dataclass
    class Window:
        start: int = 0
        end: int = 1

        def validate(self) -> None:
            if self.end <= self.start:
                raise ValueError("end must be after start")

    @dataclass
    class Outer:
        window: Window = field(default_factory=Window)

    assert problems_of(Outer, {"window": {"start": 5, "end": 2}}) == [
        "window: end must be after start"
    ]


def test_an_existing_instance_is_checked_again():
    unchecked = StepwiseSettings(max_interventions=-3)
    assert problems_of(StepwiseSettings, unchecked) == ["max_interventions: must be >= 0, got -3"]


# Converting back to plain values


def test_settings_to_dict_gives_plain_values_that_round_trip():
    settings = validate_settings(
        SamplerSettings,
        {"name": "a", "span": (2, "c"), "schedule": {"weights": [0.2, 0.8]}, "limit": 4},
    )
    plain = settings_to_dict(settings)
    assert plain["span"] == [2, "c"]
    assert plain["schedule"] == {"warmup": 0, "weights": [0.2, 0.8]}
    assert validate_settings(SamplerSettings, plain) == settings


def test_settings_to_dict_rejects_non_dataclasses():
    with pytest.raises(TypeError, match="settings dataclass instance"):
        settings_to_dict({"explain": True})


# Declaration checks


def test_valid_settings_classes_pass_the_declaration_check():
    check_settings_class(StepwiseSettings)
    check_settings_class(SamplerSettings)


def test_settings_must_be_a_dataclass():
    class NotADataclass:
        explain: bool = False

    with pytest.raises(SettingsDeclarationError, match="must be a dataclass"):
        check_settings_class(NotADataclass)


def test_declaration_problems_are_reported_together():
    @dataclass
    class Broken:
        tags: set[str] = field(default_factory=set)
        items: list = field(default_factory=list)  # type: ignore[type-arg]
        label: str = setting("x", ge=0)
        count: int = setting(-1, ge=0)
        code: str = setting("a", pattern="(")
        lookup: dict[int, str] = field(default_factory=dict)

    with pytest.raises(SettingsDeclarationError) as caught:
        check_settings_class(Broken)
    messages = [str(p) for p in caught.value.problems]
    assert caught.value.settings_class.endswith("Broken")
    assert len(messages) == 6
    assert messages[0].startswith("tags: unsupported setting type set[str]")
    assert messages[1] == "items: give the item types, as in list[int], not bare list"
    assert messages[2] == "label: ge= applies to numbers, not str"
    assert messages[3] == "count: invalid default: must be >= 0, got -1"
    assert messages[4].startswith("code: pattern '(' is not valid")
    assert messages[5] == "lookup: dict keys must be str, as in dict[str, int]"


def test_constraints_and_literal_values_that_do_not_fit_their_type_are_declaration_errors():
    @dataclass
    class Misfit:
        ratio: float = setting(0.5, min_length=1)
        count: int = setting(1, max_length=3)
        flag: bool = setting(False, pattern="x")
        level: Literal[1.5, 2.5] = 1.5  # type: ignore[valid-type]

    with pytest.raises(SettingsDeclarationError) as caught:
        check_settings_class(Misfit)
    assert [str(p) for p in caught.value.problems] == [
        "ratio: min_length= applies to strings, lists, tuples, and dicts, not float",
        "count: max_length= applies to strings, lists, tuples, and dicts, not int",
        "flag: pattern= applies to strings, not bool",
        "level: Literal values must be str, int, bool, or None",
    ]


def test_annotations_that_cannot_be_resolved_are_a_declaration_error():
    @dataclass
    class Unresolved:
        widget: "Missing | None" = None  # type: ignore[name-defined]  # noqa: F821

    with pytest.raises(SettingsDeclarationError) as caught:
        check_settings_class(Unresolved)
    assert str(caught.value.problems[0]) == (
        "annotations could not be resolved: name 'Missing' is not defined"
    )


def test_problems_in_nested_settings_classes_carry_the_field_path():
    @dataclass
    class Inner:
        tags: set[str] = field(default_factory=set)

    @dataclass
    class Outer:
        inner: Inner = field(default_factory=Inner)

    with pytest.raises(SettingsDeclarationError) as caught:
        check_settings_class(Outer)
    assert str(caught.value.problems[0]).startswith("inner.tags: unsupported setting type")


def test_validate_settings_reports_declaration_problems_first():
    @dataclass
    class Broken:
        tags: set[str] = field(default_factory=set)

    with pytest.raises(SettingsDeclarationError):
        validate_settings(Broken, {})


# JSON Schema


def test_schema_describes_each_setting():
    schema = settings_schema(StepwiseSettings)
    assert schema == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "StepwiseSettings",
        "description": "Settings for a teacher that reviews each step.",
        "type": "object",
        "properties": {
            "explain": {
                "type": "boolean",
                "description": "Explain each correction.",
                "default": False,
            },
            "max_interventions": {
                "type": "integer",
                "description": "Interventions allowed per episode.",
                "default": 10,
                "minimum": 0,
            },
        },
        "additionalProperties": False,
    }


def test_schema_lists_required_settings_and_omits_generated_docstrings():
    schema = settings_schema(SamplerSettings)
    assert schema["required"] == ["name"]
    assert "description" not in settings_schema(Schedule)


def test_schema_covers_every_supported_type():
    properties = settings_schema(SamplerSettings)["properties"]
    assert properties["mode"]["enum"] == ["uniform", "weighted"]
    assert properties["temperature"] == {
        "type": "number",
        "default": 1.0,
        "exclusiveMinimum": 0,
        "maximum": 10,
    }
    assert properties["limit"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert properties["sizes"]["items"] == {"type": "integer"}
    assert properties["span"]["prefixItems"] == [{"type": "integer"}, {"type": "string"}]
    assert properties["weights"]["additionalProperties"] == {"type": "number"}
    assert properties["schedule"]["properties"]["weights"]["default"] == [1.0]
    assert properties["name"]["pattern"] == "^(?:[a-z_]+)$"
    assert properties["params"] == {"default": None}


def test_the_schema_carries_length_limits_for_strings_lists_and_dicts():
    @dataclass
    class Sized:
        code: str = setting("ab", min_length=1, max_length=3)
        sizes: list[int] = setting(default_factory=lambda: [1], min_length=1, max_length=4)
        labels: dict[str, str] = setting(default_factory=dict, max_length=2)

    properties = settings_schema(Sized)["properties"]
    assert properties["code"] == {"type": "string", "default": "ab", "minLength": 1, "maxLength": 3}
    assert (properties["sizes"]["minItems"], properties["sizes"]["maxItems"]) == (1, 4)
    assert properties["labels"]["maxProperties"] == 2
    assert "minLength" not in properties["sizes"]  # each limit only for the kind it fits
    assert "maxItems" not in properties["code"]


def test_schema_is_valid_json_schema_and_agrees_with_validation():
    schema = settings_schema(SamplerSettings)
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    good = settings_to_dict(validate_settings(SamplerSettings, {"name": "abc"}))
    assert validator.is_valid(good)
    for bad in [
        {"name": "abc", "temperature": 0},
        {"name": "abc", "mode": "greedy"},
        {"name": "abc", "unknown": 1},
        {"name": "abc", "span": [1, "a", 2]},
        {"name": "Big"},
        {},
    ]:
        assert not validator.is_valid(bad), bad
        with pytest.raises(ConfigError):
            validate_settings(SamplerSettings, bad)
