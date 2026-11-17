"""The environment contract: capabilities, participants and turn order, and descriptions."""

from typing import TYPE_CHECKING, Any

import pytest

from ssapling import environments, models
from ssapling.environments import (
    CAPABILITIES,
    Environment,
    ResetResult,
    StepResult,
    active_agents,
    agents,
    next_agents,
    supports,
)
from tests.fixtures.environments import (
    ArcEnv,
    AsyncProxy,
    DigitPadEnv,
    LaneBatch,
    NimEnv,
    PenniesEnv,
)
from tests.fixtures.models import CountingLearner

# The fixtures implement the protocol structurally, inheriting from nothing.
# mypy checks these assignments; they never run. The last one must stay an
# error: strict mypy reports an unneeded "type: ignore", so this fails if the
# protocol ever stops telling environments apart from other components.
if TYPE_CHECKING:
    _single: Environment = DigitPadEnv()
    _sequential: Environment = NimEnv()
    _simultaneous: Environment = PenniesEnv()
    _remote: Environment = ArcEnv()
    _proxy: Environment = AsyncProxy(PenniesEnv())
    _not_an_environment: Environment = CountingLearner()  # type: ignore[assignment]


class Minimal:
    """The least an environment needs: reset and step."""

    def reset(self) -> ResetResult:
        return ResetResult(observations={"agent": 0})

    def step(self, actions: Any) -> StepResult:
        return StepResult(observations={"agent": 0}, terminated={"agent": True})


# Capabilities


def test_capabilities_follow_from_the_methods_an_environment_has():
    assert {c: supports(DigitPadEnv(), c) for c in CAPABILITIES} == {
        "seeded_reset": True,
        "render": True,
        "state": True,
        "snapshot": True,
        "batched": False,
    }
    assert {c: supports(Minimal(), c) for c in CAPABILITIES} == dict.fromkeys(CAPABILITIES, False)
    assert supports(LaneBatch(DigitPadEnv, lanes=2), "batched")


def test_seeded_reset_needs_reset_to_take_a_seed():
    assert not supports(PenniesEnv(), "seeded_reset")  # reset() takes no seed
    assert not supports(ArcEnv(), "seeded_reset")  # reset(options=) only


def test_an_adapter_switches_off_what_the_wrapped_environment_cannot_honour():
    """NimEnv.reset takes a seed, but the native game ignores it: no false guarantee."""
    assert not supports(NimEnv(), "seeded_reset")


def test_an_environment_cannot_claim_a_capability_it_lacks():
    class Boastful(Minimal):
        def supports(self, capability: str) -> bool:
            return True

    assert not supports(Boastful(), "snapshot")


def test_a_snapshot_needs_both_snapshot_and_restore():
    class HalfSnapshot(Minimal):
        def snapshot(self) -> None:
            pass

    assert not supports(HalfSnapshot(), "snapshot")


def test_a_wrapper_reports_what_the_environment_it_wraps_supports():
    assert supports(AsyncProxy(DigitPadEnv()), "snapshot")
    assert not supports(AsyncProxy(NimEnv()), "seeded_reset")
    assert not supports(AsyncProxy(PenniesEnv()), "seeded_reset")


def test_unknown_capabilities_are_a_programming_error():
    with pytest.raises(ValueError, match="unknown capability 'teleport'; the capabilities are"):
        supports(Minimal(), "teleport")


def test_a_supports_attribute_must_be_a_method():
    class ListsCapabilities(Minimal):
        supports = ("render",)

    with pytest.raises(TypeError, match=r"ListsCapabilities.supports must be a method"):
        supports(ListsCapabilities(), "render")


def test_capabilities_work_as_the_contract_says():
    env = DigitPadEnv()
    env.reset(options={"a": 45, "b": 57})
    env.step({"student": "2"})
    saved = env.snapshot()
    env.step({"student": "9"})  # wrong: ends the game
    assert env.render() == "45 + 57 = 92"
    env.restore(saved)
    assert env.render() == "45 + 57 = 2"
    assert env.state()["answer"] == 102  # what the agent cannot observe


# Participants and turn order


def test_a_single_agent_environment_needs_no_turn_logic():
    first = Minimal().reset()
    assert agents(Minimal(), first) == ("agent",)
    assert next_agents(Minimal(), first) == ("agent",)


def test_by_default_every_active_agent_acts_at_once():
    env = PenniesEnv()
    first = env.reset()
    assert next_agents(env, first) == ("player_1", "player_2")
    last = env.step({"player_1": "heads", "player_2": "tails"})
    assert next_agents(env, last) == ("player_1", "player_2")


def test_an_agent_whose_part_has_ended_is_no_longer_active():
    last = StepResult(
        observations={"p1": 0, "p2": 0, "p3": 0}, terminated={"p1": False, "p2": True}
    )
    assert active_agents(Minimal(), last) == ("p1", "p3")
    assert next_agents(Minimal(), last) == ("p1", "p3")


def test_a_sequential_environment_says_whose_turn_it_is():
    env = NimEnv(stones=5)
    first = env.reset()
    assert agents(env, first) == ("player_1", "player_2")
    assert next_agents(env, first) == ("player_1",)
    last = env.step({"player_1": 2})
    assert next_agents(env, last) == ("player_2",)


def test_participants_must_be_a_collection_of_agent_ids():
    class OneName(Minimal):
        next_agents = "agent"

    with pytest.raises(TypeError, match=r"OneName.next_agents must be a collection of agent ids"):
        next_agents(OneName(), OneName().reset())


# Descriptions


def test_a_models_descriptions_are_checked_against_the_environments():
    env = DigitPadEnv()
    models.check_descriptions(
        CountingLearner(),
        observations=env.observation_descriptions["student"],
        actions=env.action_descriptions["student"],
        participant="student",
    )
    with pytest.raises(models.DescriptionMismatchError, match="the environment gives"):
        models.check_descriptions(
            CountingLearner(), actions=ArcEnv.action_descriptions["agent"], participant="agent"
        )


def test_the_environments_area_exports_its_helpers():
    assert environments.supports is supports
    assert environments.next_agents is next_agents
