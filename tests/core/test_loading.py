"""Loading configuration files: YAML and JSON, errors with file and line, component entries."""

import json
import sys
import textwrap

import pytest

from ssapling import SSAPlingError
from ssapling.core import ComponentSpec, Config, ConfigError, MissingDependencyError, load_config


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def problems_of(path) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    return [str(p) for p in caught.value.problems]


RUN_YAML = """
    teacher:
      name: stepwise
      settings:
        explain: true
        max_interventions: 5
    """


# Loading


def test_yaml_loads_into_plain_mappings(tmp_path):
    path = write(tmp_path, "run.yaml", RUN_YAML)
    config = load_config(path)
    assert config == {
        "teacher": {"name": "stepwise", "settings": {"explain": True, "max_interventions": 5}}
    }
    assert isinstance(config, Config)
    assert isinstance(config, dict)
    assert type(config["teacher"]) is dict
    assert config.source == str(path)


def test_json_loads_the_same_values(tmp_path):
    yaml_config = load_config(write(tmp_path, "run.yaml", RUN_YAML))
    json_config = load_config(
        write(
            tmp_path,
            "run.json",
            '{"teacher": {"name": "stepwise",'
            ' "settings": {"explain": true, "max_interventions": 5}}}',
        )
    )
    assert json_config == yaml_config


@pytest.mark.parametrize("name", ["run.yml", "RUN.YAML"])
def test_other_yaml_suffixes_are_accepted(tmp_path, name):
    assert load_config(write(tmp_path, name, "a: 1")) == {"a": 1}


def test_exponent_numbers_are_floats_in_yaml(tmp_path):
    path = write(
        tmp_path,
        "numbers.yaml",
        """
        lr: 1e-4
        scale: 1.0e5
        negative: -2E3
        plain_float: 1.5
        integer: 10
        quoted: "1e-4"
        """,
    )
    config = load_config(path)
    assert config == {
        "lr": 0.0001,
        "scale": 100000.0,
        "negative": -2000.0,
        "plain_float": 1.5,
        "integer": 10,
        "quoted": "1e-4",
    }
    assert type(config["integer"]) is int


def test_file_that_does_not_exist_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


# Errors that name the file and the place in it


def test_yaml_duplicate_keys_are_rejected_with_line_numbers(tmp_path):
    path = write(
        tmp_path,
        "run.yaml",
        """
        teacher:
          name: stepwise
          name: scripted
        """,
    )
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert str(caught.value) == (
        f"Invalid configuration file {path}:\n"
        "  - line 3, column 3: duplicate key 'name' (first on line 2)"
    )


def test_yaml_merge_keys_may_be_overridden(tmp_path):
    path = write(
        tmp_path,
        "run.yaml",
        """
        defaults: &defaults
          lr: 0.1
          steps: 5
        run:
          <<: *defaults
          lr: 0.2
        """,
    )
    assert load_config(path)["run"] == {"lr": 0.2, "steps": 5}


def test_json_duplicate_keys_are_rejected(tmp_path):
    path = write(tmp_path, "run.json", '{"a": 1, "b": {"c": 1, "c": 2}}')
    assert problems_of(path) == ["duplicate key 'c'"]


def test_yaml_syntax_errors_report_line_and_column(tmp_path):
    path = write(tmp_path, "run.yaml", "teacher:\n  name: [stepwise\n")
    problems = problems_of(path)
    assert len(problems) == 1
    assert problems[0].startswith("line 3, column 1: ")


def test_json_syntax_errors_report_line_and_column(tmp_path):
    text = '{\n  "a": 1,\n}'
    path = write(tmp_path, "run.json", text)
    # Python's json module decides where the error is and how to word it, and
    # both changed in Python 3.13 for this input. SSAPling reports them as given.
    with pytest.raises(json.JSONDecodeError) as python_error:
        json.loads(text)
    expected = f"line {python_error.value.lineno}, column {python_error.value.colno}"
    assert problems_of(path) == [f"{expected}: {python_error.value.msg}"]


def test_yaml_cannot_construct_python_objects(tmp_path):
    path = write(tmp_path, "run.yaml", 'x: !!python/object/apply:os.system ["echo hi"]')
    problems = problems_of(path)
    assert len(problems) == 1
    assert "could not determine a constructor" in problems[0]


def test_unsupported_suffix_is_rejected(tmp_path):
    path = write(tmp_path, "run.toml", "a = 1")
    assert problems_of(path) == ["unsupported file type '.toml'; use .yaml, .yml, or .json"]


@pytest.mark.parametrize(
    ("text", "found"),
    [("- a\n- b", "list ['a', 'b']"), ("", "an empty file"), ("3", "int 3")],
)
def test_top_level_must_be_a_mapping(tmp_path, text, found):
    path = write(tmp_path, "run.yaml", text)
    assert problems_of(path) == [f"expected a mapping at the top level, got {found}"]


# Without the yaml extra


def test_yaml_without_pyyaml_names_the_extra(tmp_path, monkeypatch):
    path = write(tmp_path, "run.yaml", RUN_YAML)
    monkeypatch.setitem(sys.modules, "yaml", None)  # makes "import yaml" fail
    with pytest.raises(MissingDependencyError, match=r"pip install 'ssapling\[yaml\]'"):
        load_config(path)
    with pytest.raises(ImportError):
        load_config(path)
    with pytest.raises(SSAPlingError):
        load_config(path)


def test_json_needs_no_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert load_config(write(tmp_path, "run.json", '{"a": 1}')) == {"a": 1}


# Reading component entries


def test_component_reads_an_entry_and_remembers_where_it_came_from(tmp_path):
    path = write(tmp_path, "run.yaml", RUN_YAML)
    spec = load_config(path).component("teacher")
    assert spec == ComponentSpec("stepwise", {"explain": True, "max_interventions": 5})
    assert (spec.source, spec.path) == (str(path), "teacher")


def test_component_follows_dotted_paths(tmp_path):
    path = write(
        tmp_path,
        "run.yaml",
        """
        components:
          model: {name: tiny}
        """,
    )
    spec = load_config(path).component("components.model")
    assert spec == ComponentSpec("tiny")
    assert spec.path == "components.model"


def test_component_reports_a_missing_entry_with_its_path(tmp_path):
    path = write(tmp_path, "run.yaml", "components: {}")
    with pytest.raises(ConfigError) as caught:
        load_config(path).component("components.teacher")
    assert str(caught.value) == (
        f"Invalid configuration file {path}:\n  - components.teacher: missing entry"
    )


def test_component_reports_a_malformed_entry_with_file_and_path(tmp_path):
    path = write(tmp_path, "run.yaml", "teacher: {nmae: stepwise}")
    with pytest.raises(ConfigError) as caught:
        load_config(path).component("teacher")
    assert str(caught.value) == (
        f"Invalid component entry in {path}:\n"
        "  - teacher.nmae: unknown key in a component entry; did you mean 'name'? "
        "(allowed: name, settings)\n"
        "  - teacher.name: missing component name"
    )
