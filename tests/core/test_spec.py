"""ComponentSpec: reading and writing one {name, settings} entry."""

import pytest

from ssapling.core import ComponentSpec, ConfigError


def problems_of(entry: object, **where: str) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        ComponentSpec.from_mapping(entry, **where)
    return [str(p) for p in caught.value.problems]


def test_reads_name_and_settings():
    spec = ComponentSpec.from_mapping({"name": "stepwise", "settings": {"explain": True}})
    assert spec == ComponentSpec("stepwise", {"explain": True})


def test_settings_are_optional_and_may_be_null():
    assert ComponentSpec.from_mapping({"name": "tiny"}).settings is None
    assert ComponentSpec.from_mapping({"name": "tiny", "settings": None}).settings is None


def test_every_layout_problem_is_reported_with_its_path():
    assert problems_of({"nmae": "x", "setting": {}}, path="teacher") == [
        "teacher.nmae: unknown key in a component entry; did you mean 'name'? "
        "(allowed: name, settings)",
        "teacher.setting: unknown key in a component entry; did you mean 'settings'? "
        "(allowed: name, settings)",
        "teacher.name: missing component name",
    ]


@pytest.mark.parametrize(
    ("name", "found"), [("", "str ''"), ("  ", "str '  '"), (3, "int 3"), (None, "None")]
)
def test_name_must_be_a_non_empty_string(name, found):
    assert problems_of({"name": name}, path="model") == [
        f"model.name: expected a non-empty string, got {found}"
    ]


def test_settings_must_be_a_mapping():
    assert problems_of({"name": "tiny", "settings": [1, 2]}, path="model") == [
        "model.settings: expected a mapping, got list [1, 2]"
    ]


def test_entry_must_be_a_mapping():
    assert problems_of("stepwise", path="teacher") == [
        "teacher: expected a component entry {name, settings}, got str 'stepwise'"
    ]


def test_error_names_the_source():
    with pytest.raises(ConfigError, match=r"^Invalid component entry in run\.yaml:"):
        ComponentSpec.from_mapping({}, source="run.yaml", path="teacher")


def test_source_and_path_are_not_part_of_equality():
    from_file = ComponentSpec("tiny", {"a": 1}, source="run.yaml", path="model")
    assert from_file == ComponentSpec("tiny", {"a": 1})


def test_to_mapping_round_trips():
    spec = ComponentSpec("stepwise", {"explain": True})
    assert spec.to_mapping() == {"name": "stepwise", "settings": {"explain": True}}
    assert ComponentSpec.from_mapping(spec.to_mapping()) == spec
    assert ComponentSpec("tiny").to_mapping() == {"name": "tiny"}
