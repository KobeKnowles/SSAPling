"""Exceptions raised by SSAPling.

Every error SSAPling raises about components, settings, or compositions derives
from ``SSAPlingError``, so callers can catch them all at once. Each error also
derives from the closest built-in exception (``LookupError``, ``ValueError``),
so code that expects those keeps working.
"""

import difflib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ._text import describe


class SSAPlingError(Exception):
    """Base class for errors raised by SSAPling.

    Every SSAPling error can be pickled, so it survives being sent from a
    worker process to the caller with its message and attributes, whatever
    arguments its ``__init__`` takes.
    """

    def __reduce__(self) -> tuple[Any, ...]:
        # Pickle would call __init__ with the message alone, which most
        # errors do not accept, so rebuild without calling __init__.
        return (_rebuild, (type(self), self.args, self.__dict__))


def _rebuild(cls: type[SSAPlingError], args: tuple[Any, ...], state: dict[str, Any]) -> Any:
    error = cls.__new__(cls)
    error.args = args
    error.__dict__.update(state)
    return error


@dataclass(frozen=True, slots=True)
class Problem:
    """One problem found while checking settings.

    Attributes:
        path: Where the problem is, such as ``"schedule.weights[2]"``; empty
            for a problem with the settings as a whole.
        message: What is wrong.
    """

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message


class ConfigError(SSAPlingError, ValueError):
    """Settings values are invalid. Every problem found is listed, not just the first.

    Attributes:
        subject: What was being checked, such as ``"settings for StepwiseSettings"``.
        problems: Every problem found.
    """

    def __init__(self, subject: str, problems: Iterable[Problem]) -> None:
        self.subject = subject
        self.problems = tuple(problems)
        super().__init__(_listing(f"Invalid {subject}:", self.problems))


class SettingsDeclarationError(SSAPlingError, TypeError):
    """A settings class is declared in a way SSAPling cannot check.

    Raised when the class is registered (or first used), so a downstream
    author sees the mistake before any settings values are supplied.

    Attributes:
        settings_class: The name of the settings class.
        problems: Every problem found.
    """

    def __init__(self, settings_class: str, problems: Iterable[Problem]) -> None:
        self.settings_class = settings_class
        self.problems = tuple(problems)
        super().__init__(_listing(f"Invalid settings class {settings_class}:", self.problems))


class UnknownComponentError(SSAPlingError, LookupError):
    """A name was looked up in a registry that has nothing registered under it.

    Attributes:
        kind: The registry's kind of component, such as ``"teacher"``.
        name: The name that was looked up.
        available: Every registered name, sorted.
        suggestions: Registered names close to ``name``, best first.
        location: Where the name was written, such as ``"run.yaml: teacher.name"``,
            or ``None``.
    """

    def __init__(
        self, kind: str, name: str, available: Iterable[str], *, location: str | None = None
    ) -> None:
        self.kind = kind
        self.name = name
        self.available = tuple(sorted(available))
        self.suggestions = tuple(difflib.get_close_matches(name, self.available, n=3, cutoff=0.6))
        self.location = location
        message = self._message()
        super().__init__(f"{location}: {message}" if location else message)

    def _message(self) -> str:
        if not self.available:
            return (
                f"Unknown {self.kind} {self.name!r}: no {self.kind} has been registered. "
                "Components from a downstream package are available only after that "
                "package registers them, for example by calling its register_components()."
            )
        message = f"Unknown {self.kind} {self.name!r}."
        if self.suggestions:
            message += f" Did you mean {_quoted(self.suggestions, 'or')}?"
        return message + f" Registered {self.kind} names: {_quoted(self.available, 'and')}."


class DuplicateRegistrationError(SSAPlingError, ValueError):
    """A name was registered twice, with a different implementation or settings class.

    Attributes:
        kind: The registry's kind of component, such as ``"teacher"``.
        name: The name registered twice.
        existing: The implementation already registered under ``name``.
        new: The implementation that was rejected.
    """

    def __init__(
        self,
        kind: str,
        name: str,
        existing: object,
        new: object,
        *,
        existing_settings: type | None = None,
        new_settings: type | None = None,
    ) -> None:
        self.kind = kind
        self.name = name
        self.existing = existing
        self.new = new
        super().__init__(
            f"A {kind} named {name!r} is already registered "
            f"({_registered(existing, existing_settings)}). Registering "
            f"{_registered(new, new_settings)} under the same name would replace it; "
            "pass replace=True if that is intended."
        )


class MissingDependencyError(SSAPlingError, ImportError):
    """An optional feature was used without the extra that provides it.

    The message names the extra to install, such as ``ssapling[yaml]``.
    """


def _listing(title: str, problems: Iterable[Problem]) -> str:
    return "\n  - ".join([title, *(str(problem) for problem in problems)])


def _quoted(names: Iterable[str], conjunction: str) -> str:
    quoted = [repr(name) for name in names]
    if len(quoted) <= 2:
        return f" {conjunction} ".join(quoted)
    return f"{', '.join(quoted[:-1])}, {conjunction} {quoted[-1]}"


def _registered(implementation: object, settings: type | None) -> str:
    if settings is None:
        return describe(implementation)
    return f"{describe(implementation)} with settings {describe(settings)}"
