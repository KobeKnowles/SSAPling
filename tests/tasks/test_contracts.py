"""The task contract: generated and enumerated tasks, features, scoring, resources."""

from typing import TYPE_CHECKING

import pytest

from ssapling import SSAPlingError
from ssapling.tasks import (
    FEATURES,
    EnumeratedTask,
    GeneratedTask,
    Instance,
    MissingTeachingResourceError,
    PartialChecker,
    PartialReport,
    Task,
    TeachingResourceProvider,
    UnknownInstanceError,
    supports,
    teaching_resource,
    teaching_resources,
)
from tests.fixtures.tasks import (
    AdditionSettings,
    AdditionTask,
    PuzzleBook,
    RemoteGameTask,
    SwitchableChecks,
)

# The fixtures implement the protocols structurally, inheriting from nothing.
# mypy checks these assignments; they never run. The last one must stay an
# error: strict mypy reports an unneeded "type: ignore", so this fails if the
# protocols ever stop telling tasks apart.
if TYPE_CHECKING:
    _task: Task = PuzzleBook()
    _generated: GeneratedTask = AdditionTask()
    _enumerated: EnumeratedTask = PuzzleBook()
    _remote: EnumeratedTask = RemoteGameTask()
    _checker: PartialChecker = AdditionTask()
    _provider: TeachingResourceProvider = AdditionTask()
    _not_enumerated: EnumeratedTask = AdditionTask()  # type: ignore[assignment]

# Generated tasks


def test_generation_is_reproducible_from_params_and_seed():
    task = AdditionTask()
    first = task.generate(params={"digits": 2}, seed=17)
    again = AdditionTask().generate(params={"digits": 2}, seed=17)
    other_seed = task.generate(params={"digits": 2}, seed=18)
    assert first == again
    assert first != other_seed
    assert first.id != other_seed.id


def test_generated_instances_carry_inputs_reference_and_descriptors():
    instance = AdditionTask().generate(params={"digits": 1}, seed=3)
    a, b = instance.inputs["a"], instance.inputs["b"]
    assert 0 <= a <= 9
    assert 0 <= b <= 9
    assert instance.reference == a + b
    assert instance.descriptors == {"digits": 1}
    assert instance.id == "add-1d-3"


def test_equivalent_generated_instances_share_a_problem_key():
    task = AdditionTask()
    instances = [task.generate(params={"digits": 1}, seed=seed) for seed in range(200)]
    by_key: dict[object, set[tuple[int, int]]] = {}
    for instance in instances:
        pair = (instance.inputs["a"], instance.inputs["b"])
        by_key.setdefault(instance.problem_key, set()).add(pair)
    assert len(by_key) < len(instances), "one-digit sums repeat across 200 seeds"
    assert all(len(pairs) == 1 for pairs in by_key.values()), "one key, one problem"


def test_params_default_to_the_task_settings():
    instance = AdditionTask(AdditionSettings(max_digits=3)).generate(seed=0)
    assert instance.descriptors == {"digits": 3}


# Enumerated tasks


def test_enumerated_instances_are_fetched_by_id():
    book = PuzzleBook()
    assert book.ids() == ("p1", "p1-mirror", "p2", "hidden")
    assert book.get("p2").inputs == {"grid": [[2, 2], [2, 2]]}


def test_unknown_id_is_reported():
    with pytest.raises(UnknownInstanceError, match="No instance 'p9' in task 'puzzle_book'"):
        PuzzleBook().get("p9")
    with pytest.raises(LookupError):
        PuzzleBook().get("p9")
    with pytest.raises(SSAPlingError):
        PuzzleBook().get("p9")


def test_equivalent_enumerated_instances_share_a_problem_key_and_others_use_their_id():
    book = PuzzleBook()
    assert book.get("p1").problem_key == book.get("p1-mirror").problem_key == "family-1"
    assert book.get("p2").problem_key == "p2"


# Scoring


def test_local_scoring_compares_the_outcome_with_the_reference():
    task = AdditionTask()
    instance = task.generate(params={"digits": 1}, seed=1)
    assert task.score(instance, instance.reference).value == 1.0
    assert task.score(instance, instance.reference + 1).value == 0.0
    assert task.score(instance, instance.reference).status == "local"


def test_a_hidden_reference_gives_an_unavailable_score_not_zero():
    book = PuzzleBook()
    score = book.score(book.get("hidden"), {"grid": [[5]]})
    assert score.status == "unavailable"
    assert score.value is None
    assert score.detail == "the reference output is hidden"


def test_external_scores_are_recorded_as_reported():
    task = RemoteGameTask(scorecard={"game-a": 0.75})
    reported = task.score(task.get("game-a"), outcome=None)
    pending = task.score(task.get("game-b"), outcome=None)
    assert (reported.status, reported.value) == ("external", 0.75)
    assert reported.detail == {"source": "scorecard"}
    assert pending.status == "unavailable"


def test_remote_instances_can_be_identifiers_only():
    instance = RemoteGameTask().get("game-a")
    assert instance == Instance(id="game-a", inputs=None)


# Features


def test_features_follow_from_the_methods_a_task_has():
    addition, book, remote = AdditionTask(), PuzzleBook(), RemoteGameTask()
    assert {f: supports(addition, f) for f in FEATURES} == {
        "generate": True,
        "enumerate": False,
        "check_partial": True,
        "teaching_resources": True,
    }
    assert {f: supports(book, f) for f in FEATURES} == {
        "generate": False,
        "enumerate": True,
        "check_partial": False,
        "teaching_resources": False,
    }
    assert supports(remote, "enumerate")
    assert not supports(remote, "check_partial")


def test_a_feature_needing_several_methods_needs_all_of_them():
    class HalfEnumerated:
        def ids(self):
            return ()

        def score(self, instance, outcome):
            raise NotImplementedError

    assert not supports(HalfEnumerated(), "enumerate")


def test_a_task_can_opt_out_of_a_feature_its_class_implements():
    assert supports(SwitchableChecks(checks=True), "check_partial")
    assert not supports(SwitchableChecks(checks=False), "check_partial")
    assert supports(SwitchableChecks(checks=False), "generate")


def test_a_task_cannot_claim_a_feature_it_lacks():
    class Boastful:
        def score(self, instance, outcome):
            raise NotImplementedError

        def supports(self, feature):
            return True

    assert not supports(Boastful(), "check_partial")


def test_a_supports_attribute_must_be_a_method():
    class ListsItsFeatures:
        supports = ("check_partial",)

        def score(self, instance, outcome):
            raise NotImplementedError

        def check_partial(self, instance, state):
            raise NotImplementedError

    # Rejected whether or not the task has the feature's methods.
    for feature in ("check_partial", "generate"):
        with pytest.raises(TypeError, match=r"ListsItsFeatures.supports must be a method"):
            supports(ListsItsFeatures(), feature)


def test_unknown_feature_names_are_a_programming_error():
    with pytest.raises(ValueError, match="unknown feature 'solve'; the features are"):
        supports(AdditionTask(), "solve")


# Partial checks


def test_partial_checks_judge_an_intermediate_state():
    task = AdditionTask()
    instance = task.generate(params={"digits": 2}, seed=5)
    answer = str(instance.reference)
    units_first = answer[::-1]
    assert task.check_partial(instance, "") == PartialReport(valid=True, progress=0.0)
    assert task.check_partial(instance, units_first[:1]) == PartialReport(
        valid=True, progress=1 / len(answer)
    )
    assert task.check_partial(instance, units_first) == PartialReport(valid=True, progress=1.0)

    wrong_units = str((int(units_first[0]) + 1) % 10)
    report = task.check_partial(instance, wrong_units)
    assert report.valid is False
    assert report.first_error == {"position_from_right": 0}
    assert task.check_partial(instance, "x").valid is False


# Teaching resources


def test_teaching_resources_by_name_or_empty_when_absent():
    assert teaching_resources(AdditionTask()) == {
        "procedure": "Add the units, carry, then add the tens."
    }
    assert teaching_resources(PuzzleBook()) == {}
    assert teaching_resource(AdditionTask(), "procedure").startswith("Add the units")


def test_a_missing_teaching_resource_is_reported_with_what_is_offered():
    with pytest.raises(MissingTeachingResourceError) as caught:
        teaching_resource(AdditionTask(), "worked_example")
    assert str(caught.value) == (
        "The task has no teaching resource 'worked_example'; it offers: 'procedure'."
    )
    assert (caught.value.name, caught.value.available) == ("worked_example", ("procedure",))

    with pytest.raises(MissingTeachingResourceError, match="offers no teaching resources"):
        teaching_resource(PuzzleBook(), "procedure")
