"""Reading what a component declares about itself, the same way in every area.

Internal: the areas' contracts use these, and users never import them. A
component declares what it offers in a few standard ways: by having methods
(a task's features, an environment's capabilities), by listing names (a
teacher's operations, a model's learning operations and signals), and by the
parameters its methods take (a model's ``key``, an environment's ``seed``).
These helpers read such declarations and report mistakes in them alike.

The plain checks every area makes of its own values, that an id is a
non-empty string and a count an integer, are here too, so their messages
read alike as well.
"""

import inspect
from collections.abc import Callable, Collection, Mapping
from typing import Any


def name_of(component: object) -> str:
    """The component's class name, for messages; ``component`` may be the class itself."""
    return component.__name__ if isinstance(component, type) else type(component).__name__


def names(
    value: object, what: str, *, of: str = "names", example: str = "('a', 'b')"
) -> tuple[str, ...]:
    """``value`` as a tuple of names; it must be a collection of non-empty strings.

    A single string is rejected, because ``"review" in "review_action"``
    would be true.

    Raises:
        TypeError: ``value`` is a string or not a collection, or holds
            something other than non-empty strings. The message names
            ``what``, such as ``"MyTeacher.operations"``.
    """
    if isinstance(value, str | bytes) or not isinstance(value, Collection):
        raise TypeError(f"{what} must be a collection of {of}, such as {example}, got {value!r}")
    found = tuple(value)
    if any(not isinstance(name, str) or not name for name in found):
        raise TypeError(f"{what} must contain only non-empty strings, got {value!r}")
    return found


def keyed(value: object, what: str, keys: str) -> dict[str, Any]:
    """A copy of ``value``, which must be a mapping with string keys.

    Raises:
        TypeError: ``value`` is not a mapping, or a key is not a string. The
            message names ``what`` and what the keys are, such as ``"agent ids"``.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping keyed by {keys}, got {value!r}")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"the keys of {what} must be strings ({keys})")
    return dict(value)


def check_identifier(value: object, what: str) -> None:
    """Check that ``value`` is a non-empty string, as every id and name must be.

    Raises:
        ValueError: It is not. The message names ``what``, such as ``"a record id"``.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")


def check_int(value: object, what: str, *, minimum: int | None = None) -> None:
    """Check that ``value`` is an integer, and at least ``minimum`` if one is given.

    A bool is not an integer here, as in Core's settings.

    Raises:
        TypeError: ``value`` is not an integer. The message names ``what``.
        ValueError: ``value`` is less than ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be {minimum} or more, got {value!r}")


def takes(method: object, parameter: str) -> bool:
    """Whether ``method`` accepts ``parameter`` as a keyword argument (or any keyword)."""
    if not callable(method):
        return False
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.kind is p.VAR_KEYWORD or (p.name == parameter and p.kind is not p.POSITIONAL_ONLY)
        for p in parameters
    )


def has_feature(
    component: object,
    feature: str,
    features: Mapping[str, tuple[str, ...]],
    *,
    word: str,
    words: str,
    requirement: Callable[[object], bool] | None = None,
) -> bool:
    """Whether ``component`` offers ``feature``, one of ``features``, declared by methods.

    It offers a feature when it has every method the feature needs, meets
    ``requirement`` if there is one, and, if it defines a
    ``supports(feature)`` method of its own, that method agrees. That method
    can only switch a feature off; it cannot claim one.

    ``word`` and ``words`` name the kind of feature in messages, such as
    ``"capability"`` and ``"capabilities"``.

    Raises:
        ValueError: ``feature`` is not one of ``features``: a mistake in the
            calling code.
        TypeError: The component has a ``supports`` attribute that is not a
            method.
    """
    try:
        methods = features[feature]
    except KeyError:
        known = ", ".join(repr(name) for name in features)
        raise ValueError(f"unknown {word} {feature!r}; the {words} are {known}") from None
    declared = getattr(component, "supports", None)
    if declared is not None and not callable(declared):
        raise TypeError(
            f"{name_of(component)}.supports must be a method that takes a {word} name, "
            f"got {declared!r}"
        )
    if not all(callable(getattr(component, method, None)) for method in methods):
        return False
    if requirement is not None and not requirement(component):
        return False
    return declared is None or bool(declared(feature))
