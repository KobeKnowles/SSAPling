"""SSAPling environments.

Responsibility: The contract for agents, actions, observations, transitions,
and turn order.

Not its responsibility: Choosing instances, scoring, teaching strategies.

May import: core, and the response types of teachers. Enforced by
tests/test_layering.py.
"""

from collections.abc import Callable

from ..core import Registry
from .contracts import (
    CAPABILITIES,
    Environment,
    active_agents,
    agents,
    next_agents,
    supports,
)
from .errors import EnvironmentCommunicationError, UnsupportedCapabilityError
from .runner import EnvironmentRunner
from .types import ResetResult, StepResult

registry: Registry[Callable[..., Environment]] = Registry("environment")
"""Registered environments. Also available as ``ssapling.registries.environments``."""

__all__ = [
    "CAPABILITIES",
    "Environment",
    "EnvironmentCommunicationError",
    "EnvironmentRunner",
    "ResetResult",
    "StepResult",
    "UnsupportedCapabilityError",
    "active_agents",
    "agents",
    "next_agents",
    "registry",
    "supports",
]
