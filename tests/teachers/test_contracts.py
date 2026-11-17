"""The teacher contract: declaring operations, supports(), and checking teaching resources."""

from typing import TYPE_CHECKING

import pytest

from ssapling import SSAPlingError, tasks
from ssapling.teachers import MissingResourcesError, Teacher, check_resources, supports
from tests.fixtures.tasks import AdditionTask, PuzzleBook
from tests.fixtures.teachers import AsyncExplainer, DigitTutor, FailingTeacher, SlowTeacher

# The fixtures implement the protocol structurally, inheriting from nothing.
# mypy checks these assignments; they never run. The last one must stay an
# error: strict mypy reports an unneeded "type: ignore", so this fails if the
# protocol ever stops telling teachers apart from other components.
if TYPE_CHECKING:
    _sync: Teacher = DigitTutor(task=AdditionTask())
    _async: Teacher = AsyncExplainer()
    _slow: Teacher = SlowTeacher(delay=1.0)
    _failing: Teacher = FailingTeacher(RuntimeError())
    _not_a_teacher: Teacher = AdditionTask()  # type: ignore[assignment]


class Listed:
    """A minimal teacher whose operations are given when it is made."""

    def __init__(self, operations: object) -> None:
        self.operations = operations

    def respond(self, request: object) -> object:
        raise NotImplementedError


# Operations


def test_a_teacher_supports_the_operations_it_lists():
    tutor = DigitTutor(task=AdditionTask())
    assert supports(tutor, "review_action")
    assert not supports(tutor, "explain")
    assert supports(AsyncExplainer(), "explain")


def test_operations_are_an_open_set_of_names():
    """A downstream project can add an operation without changing SSAPling."""
    teacher = Listed({"review_action", "rate_explanation_clarity"})
    assert supports(teacher, "rate_explanation_clarity")
    assert not supports(teacher, "an operation nobody has invented")


def test_operations_are_the_only_declaration():
    class SaysNo:
        operations = frozenset({"review_action"})

        def supports(self, operation: str) -> bool:
            return False

        def respond(self, request: object) -> object:
            raise NotImplementedError

    assert supports(SaysNo(), "review_action")


def test_a_single_string_is_not_a_list_of_operations():
    with pytest.raises(TypeError, match=r"Listed.operations must be a collection of names"):
        supports(Listed("review_action"), "review")


@pytest.mark.parametrize("operations", [{"review_action", ""}, {"review_action", 3}])
def test_operation_names_must_be_non_empty_strings(operations):
    with pytest.raises(TypeError, match="must contain only non-empty strings"):
        supports(Listed(operations), "review_action")


def test_an_object_without_operations_is_not_a_teacher():
    with pytest.raises(TypeError, match="AdditionTask has no operations"):
        supports(AdditionTask(), "review_action")


def test_operation_names_are_strings():
    with pytest.raises(TypeError, match="an operation is a name"):
        supports(AsyncExplainer(), None)  # type: ignore[arg-type]


# Teaching resources


def test_a_teacher_whose_resources_the_task_offers_passes():
    check_resources(DigitTutor, tasks.teaching_resources(AdditionTask()))
    check_resources(DigitTutor(task=AdditionTask()), ["procedure", "worked_example"])


def test_a_teacher_that_requires_nothing_needs_nothing():
    check_resources(AsyncExplainer(), tasks.teaching_resources(PuzzleBook()))


def test_a_missing_resource_is_reported_before_the_teacher_is_built():
    with pytest.raises(MissingResourcesError) as caught:
        check_resources(DigitTutor, tasks.teaching_resources(PuzzleBook()))
    assert str(caught.value) == (
        "Teacher DigitTutor needs teaching resources that are not available: 'procedure'; "
        "none are available."
    )
    assert isinstance(caught.value, LookupError)
    assert isinstance(caught.value, SSAPlingError)


def test_every_missing_resource_is_listed_with_what_is_available():
    class Demanding:
        required_resources = ("procedure", "worked_example", "hints")

    with pytest.raises(MissingResourcesError) as caught:
        check_resources(Demanding, {"procedure": "...", "glossary": "..."})
    assert caught.value.missing == ("worked_example", "hints")
    assert caught.value.available == ("glossary", "procedure")
    assert str(caught.value).endswith("available: 'glossary', 'procedure'.")


def test_required_resources_must_be_a_collection_of_names():
    class OneString:
        required_resources = "procedure"

    with pytest.raises(TypeError, match=r"OneString.required_resources must be a collection"):
        check_resources(OneString, ["procedure"])


def test_available_resources_must_be_a_collection_of_names():
    with pytest.raises(TypeError, match="available must be a collection of resource names"):
        check_resources(DigitTutor, "procedure")
