"""The result types of the Environments area: what reset and step return.

Data may import this module, and only this one, from ``environments``: a
record step holds the environment's transition result. So it holds plain data
and no environment behaviour.

Every per-agent value is keyed by agent id, even with one agent, so records
from single-agent and multi-agent environments have the same shape.
Episode-level signals sit beside the per-agent ones.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core._declarations import keyed
from ..teachers.types import TeacherResponse

__all__ = ["ResetResult", "StepResult"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResetResult:
    """What an environment returns when a new episode starts.

    Attributes:
        observations: Each agent's initial observation, keyed by agent id.
        info: Metadata about the episode, keyed by name. SSAPling passes it
            through uninterpreted.
        action_restrictions: Optional. The actions each agent may currently
            take, keyed by agent id, in the environment's own form (such as
            a list of available actions or a mask). Whether an action fits
            the interface is a separate question, answered by the action
            descriptions.
    """

    observations: Mapping[str, Any]
    info: Mapping[str, Any] = field(default_factory=dict)
    action_restrictions: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", _per_agent(self.observations, "observations"))
        object.__setattr__(self, "info", keyed(self.info, "info", "names"))
        if self.action_restrictions is not None:
            restrictions = _per_agent(self.action_restrictions, "action_restrictions")
            object.__setattr__(self, "action_restrictions", restrictions)


@dataclass(frozen=True, slots=True, kw_only=True)
class StepResult:
    """What an environment returns after a step.

    Attributes:
        observations: Each agent's observation after the step, keyed by agent
            id. For an agent whose part has just ended, this is its final
            observation.
        rewards: Optional. Each agent's reward, keyed by agent id. ``None``
            means the environment has no rewards, which is not the same as a
            reward of 0.
        terminated: Optional. For each agent, whether its part ended because
            the environment reached a terminal state (it won, lost, or
            solved the problem). ``None`` means not reported.
        truncated: Optional. For each agent, whether its part was cut off
            before a terminal state, for example by a time limit. ``None``
            means not reported.
        episode_terminated: Whether the episode as a whole reached a terminal
            state.
        episode_truncated: Whether the episode as a whole was cut off.
        info: Diagnostics, keyed by name, passed through uninterpreted.
        executed_actions: Optional. The actions the environment actually
            executed, keyed by agent id, when they may differ from those
            submitted (for example after an intervention by a teacher the
            environment owns). ``None`` means not reported.
        teacher_responses: Optional. The responses of a teacher that the
            environment owns, made during this step. ``None`` means the
            environment reports no teacher.
        action_restrictions: Optional. The actions each agent may take next,
            keyed by agent id, in the environment's own form.

    An agent finishing need not end the episode. When the episode ends is
    the environment's rule: if it gives either episode signal, that is used
    as given, and the other counts as ``False``. Only when it gives neither
    does SSAPling fill the gap: the episode ends once every agent the result
    reports on has ended, terminated if all of them terminated and truncated
    otherwise. An environment that reports on only some of its agents
    should give the episode signals itself.
    """

    observations: Mapping[str, Any]
    rewards: Mapping[str, Any] | None = None
    terminated: Mapping[str, bool] | None = None
    truncated: Mapping[str, bool] | None = None
    episode_terminated: bool | None = None
    episode_truncated: bool | None = None
    info: Mapping[str, Any] = field(default_factory=dict)
    executed_actions: Mapping[str, Any] | None = None
    teacher_responses: Sequence[TeacherResponse] | None = None
    action_restrictions: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "observations", _per_agent(self.observations, "observations"))
        set_(self, "info", keyed(self.info, "info", "names"))
        for name in ("rewards", "executed_actions", "action_restrictions"):
            value = getattr(self, name)
            if value is not None:
                set_(self, name, _per_agent(value, name))
        for name in ("terminated", "truncated"):
            value = getattr(self, name)
            if value is not None:
                set_(self, name, _flags(value, name))
        if self.teacher_responses is not None:
            set_(self, "teacher_responses", _responses(self.teacher_responses))

        for name in ("episode_terminated", "episode_truncated"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be True or False, got {value!r}")
        if self.episode_terminated is None and self.episode_truncated is None:
            terminated, truncated = self._episode_end()
        else:
            terminated, truncated = bool(self.episode_terminated), bool(self.episode_truncated)
        set_(self, "episode_terminated", terminated)
        set_(self, "episode_truncated", truncated)

    @property
    def episode_over(self) -> bool:
        """Whether the episode has ended, by termination or truncation."""
        return bool(self.episode_terminated or self.episode_truncated)

    @property
    def ended_agents(self) -> frozenset[str]:
        """The agents this result reports as terminated or truncated."""
        terminated, truncated = self.terminated or {}, self.truncated or {}
        return frozenset(
            a for a in {*terminated, *truncated} if terminated.get(a) or truncated.get(a)
        )

    def _episode_end(self) -> tuple[bool, bool]:
        """The episode signals when the environment gives neither: from the agents' own."""
        agents = {*self.observations, *(self.terminated or {}), *(self.truncated or {})}
        if not agents or not agents <= self.ended_agents:
            return False, False
        all_terminated = all((self.terminated or {}).get(agent) for agent in agents)
        return all_terminated, not all_terminated


def _per_agent(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{what} must be keyed by agent id, even with one agent, such as "
            f"{{'agent': ...}}, got {value!r}"
        )
    return keyed(value, what, "agent ids")


def _flags(value: object, what: str) -> dict[str, bool]:
    flags = _per_agent(value, what)
    for agent, flag in flags.items():
        if not isinstance(flag, bool):
            raise TypeError(
                f"{what}[{agent!r}] must be True or False, got {flag!r}; "
                "an adapter converts it, for example with bool()"
            )
    return flags


def _responses(value: object) -> tuple[TeacherResponse, ...]:
    if isinstance(value, Mapping) or not isinstance(value, Sequence):
        raise TypeError(f"teacher_responses must be a sequence of TeacherResponse, got {value!r}")
    if any(not isinstance(response, TeacherResponse) for response in value):
        raise TypeError("teacher_responses must contain only TeacherResponse objects")
    return tuple(value)
