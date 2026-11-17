"""ResetResult and StepResult: per-agent values, and when an episode ends."""

import pickle

import pytest

from ssapling.environments import ResetResult, StepResult
from ssapling.teachers import TeacherRequest, TeacherResponse

# Reset results


def test_a_reset_result_holds_initial_observations_and_metadata():
    first = ResetResult(
        observations={"agent": [[0]]},
        info={"level": 1},
        action_restrictions={"agent": ["ACTION1"]},
    )
    assert first.observations == {"agent": [[0]]}
    assert first.info == {"level": 1}
    assert first.action_restrictions == {"agent": ["ACTION1"]}
    assert ResetResult(observations={"agent": 0}).info == {}


def test_observations_are_keyed_by_agent_even_with_one_agent():
    with pytest.raises(TypeError, match=r"keyed by agent id, even with one agent"):
        ResetResult(observations=[[0]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="the keys of observations must be strings"):
        ResetResult(observations={0: "x"})  # type: ignore[dict-item]


# Step results


def test_a_step_result_holds_every_signal_per_agent():
    response = TeacherResponse.answered(TeacherRequest(id="r", operation="review_action"))
    result = StepResult(
        observations={"student": "4"},
        rewards={"student": 1.0},
        terminated={"student": False},
        truncated={"student": False},
        info={"correct": True},
        executed_actions={"student": "4"},
        teacher_responses=[response],
        action_restrictions={"student": "0123456789"},
    )
    assert result.rewards == {"student": 1.0}
    assert result.executed_actions == {"student": "4"}
    assert result.teacher_responses == (response,)
    assert result.info == {"correct": True}


def test_no_rewards_is_not_a_reward_of_zero():
    assert StepResult(observations={"agent": 0}).rewards is None
    assert StepResult(observations={"agent": 0}, rewards={"agent": 0}).rewards == {"agent": 0}


def test_per_agent_values_are_copied_when_the_result_is_made():
    rewards = {"player_1": 1}
    result = StepResult(observations={"player_1": 0}, rewards=rewards)
    rewards["player_1"] = 99
    assert result.rewards == {"player_1": 1}


def test_every_mapping_a_result_is_given_is_copied_when_it_is_made():
    observations, info, restrictions = {"agent": [[0]]}, {"level": 1}, {"agent": ["ACTION1"]}
    first = ResetResult(observations=observations, info=info, action_restrictions=restrictions)
    step_observations, step_info = {"agent": [[1]]}, {"level": 2}
    later = StepResult(observations=step_observations, info=step_info)
    for given in (observations, info, restrictions, step_observations, step_info):
        given.clear()
    assert (first.observations, first.info) == ({"agent": [[0]]}, {"level": 1})
    assert first.action_restrictions == {"agent": ["ACTION1"]}
    assert (later.observations, later.info) == ({"agent": [[1]]}, {"level": 2})


def test_termination_and_truncation_flags_must_be_booleans():
    with pytest.raises(TypeError, match=r"terminated\['agent'\] must be True or False.*bool\(\)"):
        StepResult(observations={"agent": 0}, terminated={"agent": 1})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="episode_truncated must be True or False"):
        StepResult(observations={"agent": 0}, episode_truncated="no")  # type: ignore[arg-type]


def test_teacher_responses_must_be_responses():
    with pytest.raises(TypeError, match="teacher_responses must be a sequence"):
        StepResult(observations={}, teacher_responses={"student": "ok"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must contain only TeacherResponse"):
        StepResult(observations={}, teacher_responses=["ok"])  # type: ignore[list-item]


# When the episode ends


def test_one_agents_end_is_the_episodes_end():
    """The design doc's adapter gives only per-agent flags; the episode signals follow."""
    done = StepResult(observations={"agent": 0}, terminated={"agent": True}, truncated=None)
    going = StepResult(observations={"agent": 0}, terminated={"agent": False})
    assert (done.episode_terminated, done.episode_truncated, done.episode_over) == (
        True,
        False,
        True,
    )
    assert (going.episode_terminated, going.episode_truncated, going.episode_over) == (
        False,
        False,
        False,
    )


def test_a_time_limit_truncates_rather_than_terminates():
    result = StepResult(
        observations={"agent": 0}, terminated={"agent": False}, truncated={"agent": True}
    )
    assert (result.episode_terminated, result.episode_truncated) == (False, True)


def test_one_agent_finishing_need_not_end_the_episode():
    result = StepResult(
        observations={"player_1": 0, "player_2": 0},
        terminated={"player_1": True, "player_2": False},
    )
    assert result.ended_agents == {"player_1"}
    assert not result.episode_over


def test_the_episode_ends_when_every_agent_has_ended():
    all_terminated = StepResult(
        observations={"p1": 0, "p2": 0}, terminated={"p1": True, "p2": True}
    )
    mixed = StepResult(
        observations={"p1": 0, "p2": 0},
        terminated={"p1": True, "p2": False},
        truncated={"p1": False, "p2": True},
    )
    assert (all_terminated.episode_terminated, all_terminated.episode_truncated) == (True, False)
    assert (mixed.episode_terminated, mixed.episode_truncated) == (False, True)


def test_the_environments_own_episode_signals_win():
    result = StepResult(
        observations={"p1": 0, "p2": 0},
        terminated={"p1": True, "p2": False},
        episode_terminated=True,
    )
    assert (result.episode_terminated, result.episode_truncated) == (True, False)
    still_going = StepResult(
        observations={"agent": 0}, terminated={"agent": True}, episode_terminated=False
    )
    assert not still_going.episode_over


@pytest.mark.parametrize(
    ("rule", "result", "expected"),
    [
        (
            "the game ends as soon as anyone wins",
            StepResult(
                observations={"p1": 0, "p2": 0, "p3": 0},
                terminated={"p1": True, "p2": False, "p3": False},
                episode_terminated=True,
            ),
            (True, False),
        ),
        (
            "an episode-wide time limit, with no per-agent flags at all",
            StepResult(observations={"p1": 0, "p2": 0}, episode_truncated=True),
            (False, True),
        ),
        (
            "the last player standing wins: one still playing, yet the episode is over",
            StepResult(
                observations={"p1": 0, "p2": 0},
                terminated={"p1": True, "p2": False},
                episode_terminated=True,
            ),
            (True, False),
        ),
        (
            "everyone has ended, but the environment keeps the episode going",
            StepResult(observations={"p1": 0}, terminated={"p1": True}, episode_terminated=False),
            (False, False),
        ),
        (
            "both at once, as some environments report",
            StepResult(observations={"p1": 0}, episode_terminated=True, episode_truncated=True),
            (True, True),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_any_downstream_rule_for_the_episodes_end_is_expressed_by_its_own_signals(
    rule, result, expected
):
    """The rule for when an episode ends is the environment's; SSAPling only fills in a gap."""
    assert (result.episode_terminated, result.episode_truncated) == expected


def test_an_agent_that_has_not_reported_has_not_ended():
    """An agent that only received an observation keeps the episode going."""
    result = StepResult(observations={"p1": 0, "p2": 0}, terminated={"p1": True})
    assert not result.episode_over


def test_results_pickle():
    """So they can be sent from an environment in another process (Stage 7)."""
    result = StepResult(observations={"agent": 0}, terminated={"agent": True}, rewards=None)
    assert pickle.loads(pickle.dumps(result)) == result
    first = ResetResult(observations={"agent": 0})
    assert pickle.loads(pickle.dumps(first)) == first
