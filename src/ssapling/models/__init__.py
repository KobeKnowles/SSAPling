"""SSAPling models.

Responsibility: Interfaces and lifecycle for user-defined models, their inputs
and outputs, and internal state.

Not its responsibility: Where data comes from or when learning happens.

May import: core. Enforced by tests/test_layering.py.
"""

from collections.abc import Callable

from ..core import Registry
from .buffering import as_buffered
from .contracts import (
    ActModel,
    BufferedModel,
    Model,
    check_descriptions,
    filter_signals,
    supports,
)
from .errors import DescriptionMismatchError
from .types import NOTHING, DescriptionKind, DescriptionMismatch, Nothing, Refusal

registry: Registry[Callable[..., Model]] = Registry("model")
"""Registered models. Also available as ``ssapling.registries.models``."""

__all__ = [
    "NOTHING",
    "ActModel",
    "BufferedModel",
    "DescriptionKind",
    "DescriptionMismatch",
    "DescriptionMismatchError",
    "Model",
    "Nothing",
    "Refusal",
    "as_buffered",
    "check_descriptions",
    "filter_signals",
    "registry",
    "supports",
]
