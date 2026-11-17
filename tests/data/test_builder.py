"""Building records step by step, from the fixture environments driven by hand.

The tests play the coordinating code's part themselves, so they exercise
the record structure on real transitions before the episode loop exists.
"""

import asyncio

import pytest

from ssapling.data import (
    Record,
    RecordBuilder,
    RecordHeader,
    RecordStep,
    RecordSummary,
)
from ssapling.environments import EnvironmentCommunicationError, ResetResult, StepResult
from ssapling.tasks import Instance
from ssapling.teachers import TeacherRequest, TeacherResponse
from tests.fixtures.environments import ArcEnv, DigitPadEnv, NimEnv, PenniesEnv
from tests.fixtures.tasks import AdditionTask, PuzzleBook
from tests.fixtures.teachers import DigitTutor, DigitTutorSettings

PROBLEM = {"a": 45, "b": 57}  # 102: the student enters "2", "0", "1", units first
PROPOSALS = ("3", "0", "1")  # the first digit is wrong


def tutor() -> DigitTutor:
    return DigitTutor(DigitTutorSettings(intervene=True), task=AdditionTask())


def start(env, record_id, **header) -> RecordBuilder:
    reset = env.reset(options=PROBLEM) if isinstance(env, DigitPadEnv) else env.reset()
    participants = header.pop("participants", tuple(reset.observations))
    return RecordBuilder(
        RecordHeader(
            id=record_id,
            instance_id="add-45-57",
            participants=participants,
            reset=reset,
            **header,
        )
    )


# One agent


def test_a_record_of_one_agent_is_built_step_by_step_and_valid_before_its_summary():
    env = DigitPadEnv()
    builder = start(env, "pad-1")
    for index, digit in enumerate("201"):
        result = env.step({"student": digit})
        builder.append(
            RecordStep(
                index=index,
                acting=("student",),
                proposals={"student": digit},
                executed_actions={"student": digit},
                transition=result,
            )
        )
        if index == 0:
            early = builder.record
            assert not early.finished  # valid, and can be streamed or learned from now
            assert len(early.steps) == 1

    record = builder.finish(RecordSummary.terminated())
    assert record.finished
    transitions = [step.transition for step in record.steps if step.transition is not None]
    assert [t.observations["student"]["entered"] for t in transitions] == ["2", "20", "201"]
    assert transitions[-1].episode_terminated
    assert record.header.initial_observations == {"student": {"a": 45, "b": 57, "entered": ""}}
    assert len(early.steps) == 1  # an earlier snapshot does not change


def test_the_snapshot_is_reused_until_the_record_changes():
    builder = start(DigitPadEnv(), "pad-1")
    assert builder.record is builder.record
    before = builder.record
    builder.append(RecordStep(index=0, acting=("student",), proposals={"student": "2"}))
    assert builder.record is not before
    assert builder.next_index == 1


# Teacher placement: the record reads the same wherever the teacher sits


def record_with_teacher_in_environment() -> Record:
    env = DigitPadEnv(teacher=tutor())
    builder = start(env, "pad-env", teacher_placement="environment")
    for index, digit in enumerate(PROPOSALS):
        result = env.step({"student": digit})  # the environment consults its teacher
        builder.append(
            RecordStep(
                index=index,
                acting=("student",),
                proposals={"student": digit},
                # The environment's report, passed on unchanged.
                executed_actions=result.executed_actions,
                teacher_responses=result.teacher_responses or (),
                transition=result,
            )
        )
    return builder.finish(RecordSummary.terminated())


def record_with_teacher_in_coordinator() -> Record:
    env, teacher = DigitPadEnv(), tutor()
    builder = start(env, "pad-coordinator", teacher_placement="coordinator")
    entered = ""
    for index, digit in enumerate(PROPOSALS):
        request = TeacherRequest(
            id=f"pad-review-{index + 1}",
            operation="review_action",
            context={
                "instance": Instance(id="digit-pad", inputs=PROBLEM, reference=102),
                "state": entered,
            },
            proposals={"student": digit},
            participant="student",
            step=index,
        )
        response = teacher.respond(request)
        intervention = response.contributions.get("intervention", {})
        executed = intervention.get("student", digit)
        result = env.step({"student": executed})
        entered = result.observations["student"]["entered"]
        builder.append(
            RecordStep(
                index=index,
                acting=("student",),
                proposals={"student": digit},
                teacher_responses=[response],
                executed_actions={"student": executed},
                transition=result,
            )
        )
    return builder.finish(RecordSummary.terminated())


def test_each_teacher_placement_records_proposals_executed_actions_and_responses_alike():
    inside = record_with_teacher_in_environment()
    outside = record_with_teacher_in_coordinator()

    def seen(record: Record):
        return [
            (
                step.proposals,
                step.executed_actions,
                [(r.request_id, r.status, r.contributions) for r in step.teacher_responses],
                step.transition.observations,  # type: ignore[union-attr]
            )
            for step in record.steps
        ]

    assert seen(inside) == seen(outside)
    first = inside.steps[0]
    assert first.proposals == {"student": "3"}
    assert first.executed_actions == {"student": "2"}  # only an intervention changes it
    assert first.teacher_responses[0].contributions["intervention"] == {"student": "2"}
    assert inside.steps[0].transition.teacher_responses is not None  # type: ignore[union-attr]
    assert outside.steps[0].transition.teacher_responses is None  # type: ignore[union-attr]
    assert (inside.header.teacher_placement, outside.header.teacher_placement) == (
        "environment",
        "coordinator",
    )


def test_teacher_responses_of_every_status_are_recorded():
    env = PenniesEnv(rounds=5)
    builder = start(env, "pennies-teacher", teacher_placement="coordinator")
    request = TeacherRequest(id="r", operation="review_action")
    answer = TeacherResponse.answered(request, {"feedback": {"player_1": "Mix it up."}})
    responses = [
        answer,
        TeacherResponse.abstained(request, "nothing to say"),
        TeacherResponse.failed(request, "teacher raised RuntimeError"),
        TeacherResponse.cancelled(request, "no answer in time"),
        answer.as_stale(),
    ]
    for index, response in enumerate(responses):
        actions = {"player_1": "heads", "player_2": "tails"}
        builder.append(
            RecordStep(
                index=index,
                acting=("player_1", "player_2"),
                proposals=actions,
                teacher_responses=[response],
                executed_actions=actions,
                transition=env.step(actions),
            )
        )
    record = builder.finish(RecordSummary.truncated("round limit"))

    statuses = [step.teacher_responses[0].status for step in record.steps]
    assert statuses == ["answered", "abstained", "failed", "cancelled", "stale"]
    stale = record.steps[4].teacher_responses[0]
    assert stale.contributions == {"feedback": {"player_1": "Mix it up."}}  # kept, never applied
    assert record.steps[4].executed_actions == record.steps[4].proposals


# Several agents


def test_a_record_of_players_taking_turns():
    env = NimEnv(stones=5)
    builder = start(env, "nim-1", participants=NimEnv.agents)
    for index, take in enumerate((2, 2, 1)):
        (player,) = env.next_agents
        builder.append(
            RecordStep(
                index=index,
                acting=(player,),
                proposals={player: take},
                executed_actions={player: take},
                transition=env.step({player: take}),
            )
        )
    record = builder.finish(RecordSummary.terminated(detail={"winner": "player_1"}))

    assert [step.acting for step in record.steps] == [("player_1",), ("player_2",), ("player_1",)]
    last = record.steps[-1].transition
    assert last is not None
    assert last.observations == {"player_1": 0, "player_2": 0}  # both final observations
    assert last.rewards == {"player_1": 1, "player_2": -1}


def test_a_record_of_players_acting_at_once():
    env = PenniesEnv(rounds=3)
    builder = start(env, "pennies-1")
    for index, (one, two) in enumerate(
        [("heads", "heads"), ("heads", "tails"), ("tails", "tails")]
    ):
        actions = {"player_1": one, "player_2": two}
        builder.append(
            RecordStep(
                index=index,
                acting=("player_1", "player_2"),
                proposals=actions,
                executed_actions=actions,
                transition=env.step(actions),
            )
        )
    record = builder.finish(RecordSummary.truncated("round limit"))
    assert all(step.acting == ("player_1", "player_2") for step in record.steps)
    assert record.steps[-1].transition.episode_truncated  # type: ignore[union-attr]


def test_a_step_that_did_not_complete_is_kept_and_the_attempt_marked_failed():
    env = ArcEnv()
    reset = asyncio.run(env.reset())
    builder = RecordBuilder(
        RecordHeader(
            id="arc-1", instance_id="game-a", participants=("agent",), reset=reset, clock="wall"
        )
    )
    env.service.drop_next = True
    action = {"agent": {"action": "ACTION6", "x": 1, "y": 0}}
    with pytest.raises(EnvironmentCommunicationError) as caught:
        asyncio.run(env.step(action))
    builder.append(
        RecordStep(
            index=0,
            acting=("agent",),
            proposals=action,
            executed_actions=action,
            transition=None,
        )
    )
    record = builder.finish(RecordSummary.failed(str(caught.value)))
    assert record.summary is not None
    assert record.summary.status == "failed"
    assert record.steps[0].transition is None


# Static solving


def test_static_and_interactive_records_share_one_structure():
    puzzle = PuzzleBook().get("p1")
    builder = RecordBuilder(
        RecordHeader(id="static-1", instance_id=puzzle.id, participants=("solver",))
    )
    attempts = [{"grid": [[1, 1], [1, 1]]}, puzzle.reference]  # several attempts, several steps
    for index, prediction in enumerate(attempts):
        builder.append(
            RecordStep(
                index=index,
                acting=("solver",),
                presented={"solver": puzzle.inputs},
                predictions={"solver": prediction},
            )
        )
    static = builder.finish(RecordSummary.terminated())
    interactive = record_with_teacher_in_coordinator()

    assert type(static) is type(interactive)
    assert {type(step) for step in static.steps + interactive.steps} == {RecordStep}
    assert static.header.static
    assert not interactive.header.static
    assert all(step.transition is None for step in static.steps)
    assert static.steps[-1].predictions == {"solver": puzzle.reference}


def test_static_solving_can_present_an_instance_in_parts():
    puzzle = PuzzleBook().get("p2")
    builder = RecordBuilder(RecordHeader(id="parts-1", instance_id="p2", participants=("solver",)))
    for index, row in enumerate(puzzle.inputs["grid"]):
        builder.append(RecordStep(index=index, acting=("solver",), presented={"solver": row}))
    builder.append(
        RecordStep(index=2, acting=("solver",), predictions={"solver": puzzle.reference})
    )
    assert [step.presented for step in builder.record.steps] == [
        {"solver": [2, 2]},
        {"solver": [2, 2]},
        None,
    ]


# Mistakes, reported where they are made


def interactive(**header) -> RecordBuilder:
    reset = ResetResult(observations={"student": 0})
    return RecordBuilder(
        RecordHeader(id="ep-1", instance_id="i", participants=("student",), reset=reset, **header)
    )


def test_steps_are_appended_in_order():
    builder = interactive()
    with pytest.raises(ValueError, match="step index 1 is out of order: the next step is 0"):
        builder.append(RecordStep(index=1, acting=()))


def test_only_participants_act():
    with pytest.raises(ValueError, match="'teacher' acted but not among the participants"):
        interactive().append(RecordStep(index=0, acting=("teacher",)))


def test_a_finished_record_takes_no_more_steps():
    builder = interactive()
    builder.finish(RecordSummary.truncated())
    with pytest.raises(ValueError, match="record 'ep-1' is finished"):
        builder.append(RecordStep(index=0, acting=()))
    with pytest.raises(ValueError, match="already finished"):
        builder.finish(RecordSummary.terminated())


def test_a_record_holds_one_episode():
    builder = interactive()
    over = StepResult(observations={"student": 1}, terminated={"student": True})
    builder.append(RecordStep(index=0, acting=("student",), transition=over))
    with pytest.raises(ValueError, match=r"ended the episode at step 0; .* start a new record"):
        builder.append(RecordStep(index=1, acting=("student",)))


def test_a_record_refuses_a_step_after_whichever_step_ended_the_episode():
    builder = interactive()
    going = StepResult(observations={"student": 1}, terminated={"student": False})
    over = StepResult(observations={"student": 2}, terminated={"student": True})
    builder.append(RecordStep(index=0, acting=("student",), transition=going))
    builder.append(RecordStep(index=1, acting=("student",), transition=over))
    with pytest.raises(ValueError, match="ended the episode at step 1"):
        builder.append(RecordStep(index=2, acting=("student",)))


def test_static_and_interactive_steps_are_not_mixed():
    static = RecordBuilder(RecordHeader(id="s", instance_id="i", participants=("solver",)))
    with pytest.raises(ValueError, match="has proposals, transition, but record 's' is static"):
        static.append(
            RecordStep(
                index=0,
                acting=("solver",),
                proposals={"solver": 1},
                transition=StepResult(observations={}),
            )
        )
    with pytest.raises(ValueError, match="has predictions, but record 'ep-1' is interactive"):
        interactive().append(RecordStep(index=0, acting=("student",), predictions={"student": 1}))


def test_the_builder_takes_only_record_types():
    with pytest.raises(TypeError, match="header must be a RecordHeader"):
        RecordBuilder({"id": "ep-1"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="steps must be RecordStep objects"):
        interactive().append({"index": 0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="summary must be a RecordSummary"):
        interactive().finish("terminated")  # type: ignore[arg-type]
