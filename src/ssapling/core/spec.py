"""Component specs: one ``{name, settings}`` entry, in Python or from a file."""

import difflib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._text import join_path, show
from .errors import ConfigError, Problem

_KEYS = ("name", "settings")


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Which registered component to build, and the settings to build it with.

    This is SSAPling's format for one component entry. In YAML::

        teacher:
          name: stepwise
          settings:
            explain: true

    ``settings`` holds plain values as supplied. They are checked against the
    component's settings class when the component is validated or created,
    not here.

    Attributes:
        name: The registered name of the component.
        settings: The supplied settings, or ``None`` for all defaults.
        source: Where the entry came from, such as ``"run.yaml"``, for error
            messages. Not part of equality.
        path: The entry's path within the source, such as ``"teacher"``, for
            error messages. Not part of equality.
    """

    name: str
    settings: Mapping[str, Any] | None = None
    source: str | None = field(default=None, kw_only=True, compare=False)
    path: str = field(default="", kw_only=True, compare=False)

    @classmethod
    def from_mapping(
        cls, entry: object, *, source: str | None = None, path: str = ""
    ) -> "ComponentSpec":
        """Read a component entry: a mapping with ``name`` and, optionally, ``settings``.

        Raises:
            ConfigError: The entry is not a mapping, lacks a name, has a
                ``settings`` value that is not a mapping, or has other keys.
                Every problem is reported, with its path.
        """
        problems: list[Problem] = []
        if not isinstance(entry, Mapping):
            problems.append(
                Problem(path, f"expected a component entry {{name, settings}}, got {show(entry)}")
            )
            raise ConfigError(_subject("component entry", source), problems)

        for key in entry:
            if key not in _KEYS:
                message = "unknown key in a component entry"
                close = difflib.get_close_matches(str(key), _KEYS, n=1)
                if close:
                    message += f"; did you mean {close[0]!r}?"
                problems.append(
                    Problem(join_path(path, str(key)), f"{message} (allowed: name, settings)")
                )

        name = entry.get("name")
        if "name" not in entry:
            problems.append(Problem(join_path(path, "name"), "missing component name"))
        elif not isinstance(name, str) or not name.strip():
            problems.append(
                Problem(join_path(path, "name"), f"expected a non-empty string, got {show(name)}")
            )

        settings = entry.get("settings")
        if settings is not None and not isinstance(settings, Mapping):
            problems.append(
                Problem(join_path(path, "settings"), f"expected a mapping, got {show(settings)}")
            )

        if problems:
            raise ConfigError(_subject("component entry", source), problems)
        return cls(str(name), settings, source=source, path=path)

    def to_mapping(self) -> dict[str, Any]:
        """The entry as plain values, ``{"name": ..., "settings": ...}``, ready to save."""
        entry: dict[str, Any] = {"name": self.name}
        if self.settings is not None:
            entry["settings"] = dict(self.settings)
        return entry


def _subject(what: str, source: str | None) -> str:
    return f"{what} in {source}" if source else what
