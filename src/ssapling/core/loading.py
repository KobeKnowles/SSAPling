"""Loading configuration files (YAML or JSON) into plain mappings, and loading what a
``MODULE:NAME`` in them names.

Core loads files and reads individual component entries from them. How a whole
experiment file is organised (which components it combines, sweeps, overrides,
includes, presets) is up to the downstream project.
"""

import functools
import importlib
import json
import os
import re
import sys
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

from ._text import join_path, show
from .errors import ConfigError, MissingDependencyError, Problem
from .spec import ComponentSpec

_YAML_SUFFIXES = (".yaml", ".yml")
_JSON_SUFFIXES = (".json",)


class Config(dict[str, Any]):
    """The top-level mapping of a loaded configuration file.

    It is an ordinary ``dict`` of plain values, and also remembers the file it
    came from, so component entries read from it report that file in errors.

    Attributes:
        source: The path of the file, as given to ``load_config``.
    """

    def __init__(self, values: Mapping[str, Any], *, source: str | None = None) -> None:
        super().__init__(values)
        self.source = source

    def component(self, path: str) -> ComponentSpec:
        """Read the component entry at ``path``: a key, or a dotted path like ``"run.teacher"``.

        Raises:
            ConfigError: Nothing is at ``path``, or the entry is not a valid
                component entry. Errors name the file and the path.
        """
        value: Any = self
        walked = ""
        for key in path.split("."):
            walked = join_path(walked, key)
            if not isinstance(value, Mapping) or key not in value:
                raise ConfigError(self._subject(), [Problem(walked, "missing entry")])
            value = value[key]
        return ComponentSpec.from_mapping(value, source=self.source, path=path)

    def _subject(self) -> str:
        return f"configuration file {self.source}" if self.source else "configuration"


def load_config(path: str | os.PathLike[str]) -> Config:
    """Load a YAML (``.yaml``, ``.yml``) or JSON (``.json``) file into plain mappings.

    The top level must be a mapping. Duplicate keys are rejected rather than
    silently keeping the last one. In YAML, numbers in exponent form such as
    ``1e-4`` are read as floats, as in JSON and YAML 1.2. YAML needs the
    optional PyYAML dependency: ``pip install "ssapling[yaml]"``.

    Raises:
        ConfigError: The file cannot be parsed, has duplicate keys, has an
            unsupported suffix, or is not a mapping at the top level.
        MissingDependencyError: A YAML file was given and PyYAML is not
            installed.
        OSError: The file cannot be read.
    """
    file = Path(path)
    source = str(path)
    suffix = file.suffix.lower()
    if suffix in _YAML_SUFFIXES:
        values = _load_yaml(file.read_text(encoding="utf-8"), source)
    elif suffix in _JSON_SUFFIXES:
        values = _load_json(file.read_text(encoding="utf-8"), source)
    else:
        raise ConfigError(
            f"configuration file {source}",
            [Problem("", f"unsupported file type {suffix!r}; use .yaml, .yml, or .json")],
        )
    if not isinstance(values, Mapping):
        found = "an empty file" if values is None else show(values)
        raise ConfigError(
            f"configuration file {source}",
            [Problem("", f"expected a mapping at the top level, got {found}")],
        )
    return Config(values, source=source)


def load_named(spec: str, what: str) -> Any:
    """What ``spec`` names, as ``MODULE:NAME``: the module imported, then its attribute NAME.

    A configuration file names code this way, such as a task class or a
    factory. The current directory is searched too (it is put on ``sys.path``
    if it isn't there), so a command run from a project's root finds the
    project's own modules. ``what`` says what is named, for the errors.

    Raises:
        ValueError: ``spec`` isn't ``MODULE:NAME``, its module can't be
            imported, or the module has no NAME. The error names ``what``.
    """
    module_name, colon, name = spec.partition(":")
    if not (colon and module_name and name):
        raise ValueError(f"give the {what} as MODULE:NAME, such as package.module:Name: {spec!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"can't import {module_name!r} for the {what}: {error}") from None
    if not hasattr(module, name):
        raise ValueError(f"{module_name} has no {name!r} for the {what}")
    return getattr(module, name)


# JSON


def _load_json(text: str, source: str) -> Any:
    duplicates: list[str] = []

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        values = json.loads(text, object_pairs_hook=unique_keys)
    except json.JSONDecodeError as error:
        where = f"line {error.lineno}, column {error.colno}"
        raise ConfigError(f"configuration file {source}", [Problem(where, error.msg)]) from None
    if duplicates:
        raise ConfigError(
            f"configuration file {source}",
            [Problem("", f"duplicate key {key!r}") for key in duplicates],
        )
    return values


# YAML


def _load_yaml(text: str, source: str) -> Any:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise MissingDependencyError(
            f"Loading the YAML file {source} needs PyYAML, which is optional. "
            "Install it with: pip install 'ssapling[yaml]'"
        ) from error
    try:
        return yaml.load(text, Loader=_yaml_loader())  # A SafeLoader subclass.
    except yaml.MarkedYAMLError as error:
        mark = error.problem_mark
        where = f"line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        message = error.problem or str(error)
        if error.context:
            message += f" ({error.context})"
        raise ConfigError(f"configuration file {source}", [Problem(where, message)]) from None


# YAML 1.1, which PyYAML implements, reads 1e-4 and 1.0e5 as strings: its float
# pattern needs a dot and a signed exponent. This adds the exponent forms that
# YAML 1.2 and JSON read as numbers.
_EXPONENT_FLOAT = re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$")


@functools.cache
def _yaml_loader() -> Any:
    import yaml

    class Loader(yaml.SafeLoader):
        """PyYAML's safe loader, with exponent floats and duplicate keys rejected."""

        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            first_line: dict[Any, int] = {}
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue  # "<<: *defaults" merges; later keys may override it.
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, Hashable):
                    continue  # PyYAML itself reports unhashable keys.
                line = key_node.start_mark.line + 1
                if key in first_line:
                    raise yaml.constructor.ConstructorError(
                        None,
                        None,
                        f"duplicate key {key!r} (first on line {first_line[key]})",
                        key_node.start_mark,
                    )
                first_line[key] = line
            return super().construct_mapping(node, deep=deep)

    Loader.add_implicit_resolver("tag:yaml.org,2002:float", _EXPONENT_FLOAT, list("-+0123456789."))
    return Loader
