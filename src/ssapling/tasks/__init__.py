"""SSAPling tasks.

Responsibility: Problem definitions: instances, references, scoring, and
optional teaching resources.

Not its responsibility: Sampling, collections, streaming, storage, running
episodes.

May import: core. Enforced by tests/test_layering.py.
"""

from collections.abc import Callable

from ..core import Registry
from .contracts import (
    FEATURES,
    EnumeratedTask,
    GeneratedTask,
    PartialChecker,
    Task,
    TeachingResourceProvider,
    supports,
    teaching_resource,
    teaching_resources,
)
from .errors import MissingTeachingResourceError, UnknownInstanceError
from .types import Instance, PartialReport, Score, ScoreStatus

registry: Registry[Callable[..., Task]] = Registry("task")
"""Registered tasks. Also available as ``ssapling.registries.tasks``."""

__all__ = [
    "FEATURES",
    "EnumeratedTask",
    "GeneratedTask",
    "Instance",
    "MissingTeachingResourceError",
    "PartialChecker",
    "PartialReport",
    "Score",
    "ScoreStatus",
    "Task",
    "TeachingResourceProvider",
    "UnknownInstanceError",
    "registry",
    "supports",
    "teaching_resource",
    "teaching_resources",
]
