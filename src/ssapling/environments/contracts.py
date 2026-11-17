"""The environment contract, its optional capabilities, and helpers for consumers.

An environment is any object with ``reset()`` and ``step(actions)``, written
as ``def`` or ``async def``. Everything else is optional: who takes part and
who acts next, descriptions of observations and actions, ``close()``, and the
capabilities in ``CAPABILITIES``. This is a structural protocol: a native
environment or a downstream adapter implements it and inherits from nothing.

Consumers check a capability before using it::

    from ssapling import environments

    if environments.supports(env, "snapshot"):
        saved = env.snapshot()
"""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from ..core._declarations import has_feature, name_of, names, takes
from .types import ResetResult, StepResult

__all__ = [
    "CAPABILITIES",
    "Environment",
    "active_agents",
    "agents",
    "next_agents",
    "supports",
]


class Environment(Protocol):
    """What every environment provides: starting an episode, and taking a step.

    ``reset`` may also take ``seed=`` (making ``"seeded_reset"`` a
    capability) and ``options=``, what the episode is set up from, such as
    an instance's inputs. SSAPling passes each only when there is one.

    Optional members, read without waiting:

    - ``agents``: every agent id that can take part. By default, the agents
      observed at reset.
    - ``active_agents``: the agents currently taking part. By default, those
      just given an observation whose part has not ended.
    - ``next_agents``: the agents that act in the next step. By default, every
      active agent, all at once. A sequential environment says whose turn it
      is.
    - ``observation_descriptions`` and ``action_descriptions``: each agent's
      descriptions, keyed by agent id, in the environment's own form (such as
      a library's space objects), passed through unchanged.

    Optional methods, ``def`` or ``async def``: ``close()``, which releases
    what the environment holds, and the capabilities' methods (see
    ``CAPABILITIES``).
    """

    def reset(self) -> ResetResult | Awaitable[ResetResult]:
        """Start a new episode and return each agent's initial observation."""
        ...

    def step(self, actions: Mapping[str, Any]) -> StepResult | Awaitable[StepResult]:
        """Apply ``actions``, keyed by agent id, and return the transition result."""
        ...


CAPABILITIES: Mapping[str, tuple[str, ...]] = {
    "seeded_reset": ("reset",),
    "render": ("render",),
    "state": ("state",),
    "snapshot": ("snapshot", "restore"),
    "batched": ("reset_lanes", "step_lanes"),
}
"""The optional capabilities of an environment, each with the methods it needs.

- ``seeded_reset``: ``reset(seed=...)`` makes the episode reproducible.
  ``reset`` must take a ``seed`` parameter.
- ``render``: ``render()`` returns a view of the current state, such as an
  image or text.
- ``state``: ``state()`` returns the environment's full state, including what
  agents cannot observe, for inspection.
- ``snapshot``: ``snapshot()`` returns a saved state and ``restore(saved)``
  returns to it.
- ``batched``: several independent copies, called lanes, stepped together.
  ``lanes`` is how many; ``reset_lanes(lanes)`` resets the listed lanes
  (optionally with ``seeds=`` and ``options=``, keyed by lane) and
  ``step_lanes(actions)`` steps those with actions, keyed by lane, each
  returning a result per lane. A lane that ends waits to be reset; nothing
  resets it automatically. A batch of independent environments is not a
  multi-agent world.
"""


def supports(env: object, capability: str) -> bool:
    """Whether ``env`` offers ``capability``: one of ``CAPABILITIES``.

    An environment offers a capability when it has every method the
    capability needs and, if it defines a ``supports(capability)`` method of
    its own, that method agrees. So an environment declares a capability just
    by implementing it, and an adapter can switch off a capability its class
    implements when the environment it wraps cannot honour it, for example
    seeding that is not reproducible.

    An environment's own ``supports`` only switches capabilities off: it
    answers ``False`` for those and ``True`` for every other capability. It
    must not call this function on itself, because this function calls it.
    A wrapper may call it on the environment it wraps.

    Raises:
        ValueError: ``capability`` is not an environment capability. That is
            a mistake in the calling code.
        TypeError: The environment has a ``supports`` attribute that is not a
            method.
    """
    return has_feature(
        env,
        capability,
        CAPABILITIES,
        word="capability",
        words="capabilities",
        requirement=_reset_takes_a_seed if capability == "seeded_reset" else None,
    )


def agents(env: object, reset: ResetResult) -> tuple[str, ...]:
    """Every agent id that can take part: the environment's ``agents``, or those reset observed."""
    return _participants(env, "agents", lambda: tuple(reset.observations))


def active_agents(env: object, last: ResetResult | StepResult) -> tuple[str, ...]:
    """The agents taking part now, after ``last``, the latest reset or step result.

    The environment's ``active_agents`` if it has them. Otherwise, the agents
    ``last`` gave an observation to, except those whose part has ended.
    """

    def default() -> tuple[str, ...]:
        ended = last.ended_agents if isinstance(last, StepResult) else frozenset()
        return tuple(agent for agent in last.observations if agent not in ended)

    return _participants(env, "active_agents", default)


def next_agents(env: object, last: ResetResult | StepResult) -> tuple[str, ...]:
    """The agents that act in the next step, after ``last``, the latest reset or step result.

    The environment's ``next_agents`` if it has them. Otherwise, every active
    agent (see ``active_agents``), all at once.
    """
    return _participants(env, "next_agents", lambda: active_agents(env, last))


def _participants(
    env: object, member: str, default: Callable[[], tuple[str, ...]]
) -> tuple[str, ...]:
    declared = getattr(env, member, None)
    if declared is None:
        return default()
    return names(
        declared,
        f"{name_of(env)}.{member}",
        of="agent ids",
        example="('player_1',), read without calling it",
    )


def _reset_takes_a_seed(env: object) -> bool:
    return takes(getattr(env, "reset", None), "seed")
