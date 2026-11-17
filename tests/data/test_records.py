"""The record types: header, step, summary, and the record as a whole."""

import dataclasses
import pickle

import pytest

from ssapling.data import (
    RECORD_SCHEMA_VERSION,
    AgentTiming,
    Provenance,
    Record,
    RecordHeader,
    RecordStep,
    RecordSummary,
)
from ssapling.environments import ResetResult, StepResult
from ssapling.teachers import TeacherRequest, TeacherResponse

REQUEST = TeacherRequest(id="r1", operation="review_action", proposals={"student": "3"})


def header(**changes):
    fields = {
        "id": "ep-1",
        "instance_id": "add-2d-41",
        "participants": ("student",),
        "reset": ResetResult(observations={"student": {"a": 45, "b": 57, "entered": ""}}),
    }
    return RecordHeader(**{**fields, **changes})


# Header


def test_a_header_holds_what_is_fixed_when_an_attempt_starts():
    origin = Provenance(source="addition", seed=41)
    first = header(
        provenance=origin,
        seeds={"reset": 7, "model": 3},
        clock="simulated",
        teacher_placement="coordinator",
        metadata={"run": "baseline"},
    )
    assert first.participants == ("student",)
    assert first.provenance is origin
    assert first.seeds == {"reset": 7, "model": 3}
    assert first.schema_version == RECORD_SCHEMA_VERSION == 1
    assert first.initial_observations == {"student": {"a": 45, "b": 57, "entered": ""}}
    assert not first.static


def test_a_header_without_a_reset_is_static_solving():
    static = header(reset=None)
    assert static.static
    assert static.initial_observations == {}


def test_participants_are_a_non_empty_collection_of_distinct_agent_ids():
    assert header(participants=["p1", "p2"]).participants == ("p1", "p2")
    with pytest.raises(TypeError, match="participants must be a collection of agent ids"):
        header(participants="student")
    with pytest.raises(ValueError, match="at least one agent id"):
        header(participants=())
    with pytest.raises(ValueError, match="names 'p1' more than once"):
        header(participants=("p1", "p1"))


def test_ids_are_non_empty_strings():
    with pytest.raises(ValueError, match="a record id must be a non-empty string"):
        header(id="")
    with pytest.raises(ValueError, match="an instance id must be a non-empty string"):
        header(instance_id=None)


def test_the_header_checks_its_parts():
    with pytest.raises(TypeError, match="reset must be a ResetResult"):
        header(reset={"student": 0})
    with pytest.raises(TypeError, match="provenance must be a Provenance"):
        header(provenance={"source": "addition"})
    with pytest.raises(TypeError, match=r"seeds\['reset'\] must be an integer"):
        header(seeds={"reset": "7"})
    with pytest.raises(ValueError, match="clock must be one of 'simulated', 'wall', or None"):
        header(clock="cpu")
    with pytest.raises(ValueError, match="teacher_placement must be one of"):
        header(teacher_placement="inside")


def test_static_solving_has_no_environment_to_hold_a_teacher():
    with pytest.raises(ValueError, match=r"static solving .* has no environment"):
        header(reset=None, teacher_placement="environment")


def test_a_record_from_a_newer_ssapling_is_refused_by_name():
    with pytest.raises(ValueError, match=r"record 'ep-1' has schema version 2.*upgrade SSAPling"):
        header(schema_version=RECORD_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="schema_version must be 1 or more"):
        header(schema_version=0)
    with pytest.raises(TypeError, match="schema_version must be an integer"):
        header(schema_version="1")


# Steps


def test_a_step_holds_what_happened_per_agent():
    transition = StepResult(observations={"student": "x"}, terminated={"student": False})
    step = RecordStep(
        index=0,
        acting=("student",),
        proposals={"student": "3"},
        executed_actions={"student": "2"},
        teacher_responses=[TeacherResponse.answered(REQUEST, {"feedback": {"student": "No."}})],
        transition=transition,
        metadata={"model_latency_ms": 4},
    )
    assert step.transition is transition  # held, not copied
    assert step.executed_actions == {"student": "2"}
    assert isinstance(step.teacher_responses, tuple)


def test_per_agent_values_are_keyed_by_agent_even_with_one_agent():
    with pytest.raises(TypeError, match="proposals must be keyed by agent id, even with one"):
        RecordStep(index=0, acting=("student",), proposals="3")  # type: ignore[arg-type]


def test_per_agent_values_name_only_agents_that_acted():
    with pytest.raises(ValueError, match=r"step 4: proposals names 'player_2', which did not act"):
        RecordStep(index=4, acting=("player_1",), proposals={"player_1": 1, "player_2": 2})
    with pytest.raises(ValueError, match="presented names 'other'"):
        RecordStep(index=0, acting=("student",), presented={"other": 1})
    with pytest.raises(ValueError, match=r"predictions names 'solver'.*\(acting: none\)"):
        RecordStep(index=0, acting=(), predictions={"solver": 1})


def test_step_values_are_checked():
    with pytest.raises(ValueError, match="index must be 0 or more"):
        RecordStep(index=-1, acting=())
    with pytest.raises(TypeError, match="transition must be a StepResult"):
        RecordStep(index=0, acting=(), transition={"student": 0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only TeacherResponse objects"):
        RecordStep(index=0, acting=(), teacher_responses=["ok"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="executed_actions must be keyed by agent id"):
        RecordStep(index=0, acting=("student",), executed_actions="2")  # type: ignore[arg-type]


def test_the_environments_report_is_passed_on_unchanged():
    """With the teacher in the environment, the coordinating code passes its report on."""
    response = TeacherResponse.answered(REQUEST, {"intervention": {"student": "2"}})
    reported = StepResult(
        observations={"student": "x"},
        executed_actions={"student": "2", "other": "noop"},
        teacher_responses=[response],
    )
    step = RecordStep(
        index=0,
        acting=("student",),
        proposals={"student": "3"},
        executed_actions=reported.executed_actions,
        teacher_responses=reported.teacher_responses or (),
        transition=reported,
    )
    # Executed actions are not checked against acting: they are the environment's word.
    assert step.executed_actions == {"student": "2", "other": "noop"}
    assert step.teacher_responses == (response,)


def test_a_step_can_be_copied_with_changes():
    step = RecordStep(index=0, acting=("student",), executed_actions={"student": "2"})
    changed = dataclasses.replace(step, metadata={"note": "kept"})
    assert changed.executed_actions == {"student": "2"}
    assert changed.metadata == {"note": "kept"}


def test_the_coordinating_codes_values_are_copied_when_the_step_is_made():
    proposals = {"student": "3"}
    step = RecordStep(index=0, acting=("student",), proposals=proposals)
    proposals["student"] = "9"
    assert step.proposals == {"student": "3"}


def test_values_given_to_a_header_step_summary_or_record_are_copied_when_it_is_made():
    seeds, metadata = {"reset": 7}, {"run": "baseline"}
    made = header(seeds=seeds, metadata=metadata)
    acting, diagnostics, notes = ["student"], {"student": {"tokens": 12}}, {"note": "first"}
    step = RecordStep(index=0, acting=acting, diagnostics=diagnostics, metadata=notes)  # type: ignore[arg-type]
    ending = {"steps": 1}
    summary = RecordSummary.terminated(metadata=ending)
    steps = [step]
    record = Record(header=made, steps=steps, summary=summary)  # type: ignore[arg-type]

    for given in (seeds, metadata, diagnostics, notes, ending):
        given.clear()
    acting.append("intruder")
    steps.append(step)
    assert (made.seeds, made.metadata) == ({"reset": 7}, {"run": "baseline"})
    assert step.acting == ("student",)
    assert (step.diagnostics, step.metadata) == ({"student": {"tokens": 12}}, {"note": "first"})
    assert summary.metadata == {"steps": 1}
    assert record.steps == (step,)


# Summary


def test_a_summary_says_how_the_attempt_ended():
    assert RecordSummary.terminated().status == "terminated"
    cut = RecordSummary.truncated("step limit", metadata={"steps": 50})
    assert (cut.status, cut.detail, cut.metadata) == ("truncated", "step limit", {"steps": 50})
    assert RecordSummary.failed("connection lost").detail == "connection lost"
    with pytest.raises(ValueError, match="status must be one of 'terminated'"):
        RecordSummary(status="interrupted")  # type: ignore[arg-type]


# The record as a whole


def test_a_record_is_valid_before_it_has_a_summary():
    record = Record(header=header(), steps=(RecordStep(index=0, acting=("student",)),))
    assert record.summary is None
    assert not record.finished
    assert isinstance(record.steps, tuple)
    finished = dataclasses.replace(record, summary=RecordSummary.terminated())
    assert finished.finished


def test_a_record_is_checked_as_a_whole_when_made():
    with pytest.raises(ValueError, match="step index 1 is out of order: the next step is 0"):
        Record(header=header(), steps=(RecordStep(index=1, acting=()),))
    with pytest.raises(TypeError, match="header must be a RecordHeader"):
        Record(header={"id": "ep-1"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="steps must be a sequence of RecordStep"):
        Record(header=header(), steps={0: None})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="steps must be RecordStep objects"):
        Record(header=header(), steps=({"index": 0},))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="summary must be a RecordSummary"):
        Record(header=header(), summary="terminated")  # type: ignore[arg-type]


def test_a_record_pickles():
    response = TeacherResponse.answered(REQUEST, {"feedback": {"student": "No."}})
    record = Record(
        header=header(teacher_placement="coordinator", clock="wall"),
        steps=(
            RecordStep(
                index=0,
                acting=("student",),
                proposals={"student": "3"},
                teacher_responses=[response],
                transition=StepResult(observations={"student": "x"}),
            ),
        ),
        summary=RecordSummary.truncated(),
    )
    assert pickle.loads(pickle.dumps(record)) == record


def test_timing_marks_name_only_acting_agents():
    step = RecordStep(index=0, acting=("a",), timing={"a": AgentTiming(elapsed=2, defaulted=True)})
    assert step.timing["a"].defaulted
    with pytest.raises(ValueError, match="timing names 'b', which did not act"):
        RecordStep(index=0, acting=("a",), timing={"b": AgentTiming(elapsed=1)})
    with pytest.raises(TypeError, match="only AgentTiming objects"):
        RecordStep(index=0, acting=("a",), timing={"a": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="elapsed must be a number, 0 or more"):
        AgentTiming(elapsed=-1)
    with pytest.raises(ValueError, match="late must be one of"):
        AgentTiming(elapsed=1, late="lost")  # type: ignore[arg-type]


def test_elapsed_time_may_be_zero():
    assert AgentTiming(elapsed=0).elapsed == 0  # an output ready before any advance
