"""Values SSAPling hands out cannot be changed once made, so what was checked stays as checked."""

import dataclasses
import typing
from typing import Any

import pytest

from ssapling import registries as r
from ssapling.core import ComponentSpec, Problem, Registry
from ssapling.data import (
    AgentTiming,
    Drawn,
    GeneratedSettings,
    Provenance,
    Record,
    RecordHeader,
    RecordStep,
    RecordSummary,
    Sample,
    StratifiedSettings,
)
from ssapling.environments import ResetResult, StepResult
from ssapling.evaluate import (
    AggregationSettings,
    AssessmentResults,
    AssessmentSettings,
    InstanceResult,
    MeasuredValue,
    Summary,
    Trial,
)
from ssapling.interaction import EpisodeSettings
from ssapling.models import DescriptionMismatch
from ssapling.tasks import Instance, PartialReport, Score
from ssapling.teachers import TeacherRequest, TeacherResponse
from ssapling.train import Condition, Entry, Every, Procedure, Published

INSTANCE = Instance(id="add-1", inputs={"a": 1, "b": 2}, reference=3)
HEADER = RecordHeader(id="ep-1", instance_id="add-1", participants=("student",))
RECORD = Record(header=HEADER)
REQUEST = TeacherRequest(id="r1", operation="review_action")
registry: Registry[Any] = Registry("thing")
registry.register("plain", dict)

VALUES = {
    "Problem": Problem("path", "message"),
    "Registration": registry.entry("plain"),
    "ComponentSpec": ComponentSpec("plain", {}),
    "Instance": INSTANCE,
    "Score": Score.local(1.0),
    "PartialReport": PartialReport(progress=0.5),
    "TeacherRequest": REQUEST,
    "TeacherResponse": TeacherResponse.abstained(REQUEST, "nothing to add"),
    "DescriptionMismatch": DescriptionMismatch("action", "a grid"),
    "ResetResult": ResetResult(observations={"student": 0}),
    "StepResult": StepResult(observations={"student": 1}),
    "Provenance": Provenance(source="addition", seed=1),
    "Sample": Sample(instance=INSTANCE),
    "Drawn": Drawn(INSTANCE, Provenance(source="addition", seed=1)),
    "RecordHeader": HEADER,
    "RecordStep": RecordStep(index=0, acting=("student",)),
    "AgentTiming": AgentTiming(elapsed=1),
    "RecordSummary": RecordSummary.terminated(),
    "Record": RECORD,
    "MeasuredValue": MeasuredValue.available(1.0),
    "Trial": Trial(INSTANCE, RECORD, 3, None),
    "InstanceResult": InstanceResult(
        instance_id="add-1",
        collection="validation",
        descriptors={},
        record_id="ep-1",
        status="terminated",
        values={},
    ),
    "Summary": Summary(value=1.0, contributed=1),
    "AssessmentResults": AssessmentResults(run_id="a1", instances=(), aggregates={}, provenance={}),
    "Published": Published(0.5, {"episodes": 1}),
}


@pytest.mark.parametrize("value", VALUES.values(), ids=VALUES.keys())
def test_a_value_cannot_be_changed_once_made(value: Any):
    first = dataclasses.fields(value)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, first, None)


def nested(cls: type) -> list[type]:
    """``cls`` and every settings class its settings hold, however deep."""
    found = [cls]
    for hint in typing.get_type_hints(cls).values():
        for part in [hint, *typing.get_args(hint)]:
            if dataclasses.is_dataclass(part) and isinstance(part, type):
                found += nested(part)
    return found


def settings_classes() -> dict[str, type]:
    """Every settings class SSAPling declares: exported, registered with a component, or nested."""
    exported = [
        EpisodeSettings,
        AssessmentSettings,
        AggregationSettings,
        StratifiedSettings,
        GeneratedSettings,
        Entry,
        Every,
        Condition,
        Procedure,
    ]
    registered = [
        entry.settings
        for each in (r.stages, r.samplers, r.assessments)
        for entry in map(each.entry, each.names())
        if entry.settings is not None and entry.settings.__module__.startswith("ssapling.")
    ]
    return {each.__qualname__: each for cls in [*exported, *registered] for each in nested(cls)}


@pytest.mark.parametrize("cls", settings_classes().values(), ids=settings_classes().keys())
def test_settings_cannot_be_changed_once_checked(cls: type):
    assert getattr(cls, "__dataclass_params__").frozen  # noqa: B009
