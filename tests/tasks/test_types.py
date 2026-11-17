"""Instance, Score, and PartialReport."""

import pickle
from fractions import Fraction

import pytest

from ssapling.tasks import Instance, PartialReport, Score

# Instance


def test_instance_holds_identity_inputs_reference_and_descriptors():
    instance = Instance(id="add-1", inputs={"a": 1, "b": 2}, reference=3, descriptors={"digits": 1})
    assert (instance.id, instance.inputs, instance.reference) == ("add-1", {"a": 1, "b": 2}, 3)
    assert instance.descriptors == {"digits": 1}


def test_reference_descriptors_and_equivalence_key_are_optional():
    remote = Instance(id="game-a", inputs=None)
    assert remote.inputs is None
    assert remote.reference is None
    assert remote.descriptors == {}
    assert remote.equivalence_key is None


def test_inputs_must_be_given_even_when_there_are_none():
    with pytest.raises(TypeError, match="inputs"):
        Instance(id="game-a")  # type: ignore[call-arg]


def test_problem_key_is_the_equivalence_key_or_else_the_id():
    plain = Instance(id="p2", inputs=None)
    keyed = Instance(id="p1", inputs=None, equivalence_key="family-1")
    variant = Instance(id="p1-mirror", inputs=None, equivalence_key="family-1")
    assert plain.problem_key == "p2"
    assert keyed.problem_key == variant.problem_key == "family-1"
    key = ("add", 3, 4)
    assert Instance(id="x", inputs=None, equivalence_key=key).problem_key == key


@pytest.mark.parametrize("unhashable", [["add", 3, 4], ("add", [3, 4]), {"a": 3}])
def test_equivalence_key_must_be_hashable(unhashable):
    with pytest.raises(TypeError, match="equivalence key must be hashable"):
        Instance(id="x", inputs=None, equivalence_key=unhashable)


def test_instances_compare_by_value_and_hash_by_id():
    a = Instance(id="p1", inputs={"grid": [[1]]})
    b = Instance(id="p1", inputs={"grid": [[1]]})
    c = Instance(id="p1", inputs={"grid": [[2]]})
    assert a == b
    assert a != c
    assert hash(a) == hash(b) == hash(c)  # unhashable payloads do not stop hashing
    assert len({a, b}) == 1


def test_instance_fields_cannot_be_reassigned():
    instance = Instance(id="p1", inputs=None)
    with pytest.raises(AttributeError):
        instance.id = "p2"  # type: ignore[misc]


def test_descriptors_are_copied():
    descriptors = {"difficulty": "easy"}
    instance = Instance(id="p1", inputs=None, descriptors=descriptors)
    descriptors["difficulty"] = "hard"
    assert instance.descriptors == {"difficulty": "easy"}


@pytest.mark.parametrize("bad_id", ["", 3, None])
def test_id_must_be_a_non_empty_string(bad_id):
    with pytest.raises(ValueError, match="non-empty string"):
        Instance(id=bad_id, inputs=None)


@pytest.mark.parametrize("bad", [["difficulty"], {1: "easy"}])
def test_descriptors_must_be_a_mapping_with_string_names(bad):
    with pytest.raises(TypeError):
        Instance(id="p1", inputs=None, descriptors=bad)


def test_instances_pickle_for_worker_processes():
    instance = Instance(id="p1", inputs={"grid": [[1]]}, equivalence_key=("f", 1))
    assert pickle.loads(pickle.dumps(instance)) == instance


# Score


def test_local_external_and_unavailable_scores():
    local = Score.local(0.5, detail={"cells_correct": 4})
    external = Score.external(1.0, detail={"source": "scorecard"})
    unavailable = Score.unavailable("the reference output is hidden")
    assert (local.status, local.value, local.available) == ("local", 0.5, True)
    assert (external.status, external.value, external.available) == ("external", 1.0, True)
    assert (unavailable.status, unavailable.value, unavailable.available) == (
        "unavailable",
        None,
        False,
    )
    assert unavailable.detail == "the reference output is hidden"


def test_a_score_is_never_invented():
    with pytest.raises(ValueError, match="needs a value"):
        Score(status="local")
    with pytest.raises(ValueError, match="needs a value"):
        Score.external(None)
    with pytest.raises(ValueError, match="has no value"):
        Score(status="unavailable", value=0.0)


def test_falsy_values_are_still_values():
    assert Score.local(0.0).value == 0.0
    assert Score.local(False).value is False


def test_score_status_must_be_known():
    with pytest.raises(ValueError, match="status must be"):
        Score(status="guess", value=1.0)  # type: ignore[arg-type]


# PartialReport


def test_partial_report_fields_are_all_optional():
    assert PartialReport() == PartialReport(valid=None, progress=None, first_error=None)
    report = PartialReport(valid=True, progress=0.5, first_error=None, detail="halfway")
    assert (report.valid, report.progress, report.detail) == (True, 0.5, "halfway")


@pytest.mark.parametrize("progress", [-0.1, 1.5, float("nan")])
def test_progress_is_a_fraction(progress):
    with pytest.raises(ValueError, match="between 0 and 1"):
        PartialReport(progress=progress)


@pytest.mark.parametrize("progress", [0, 1, Fraction(1, 3)])
def test_progress_may_be_any_real_number(progress):
    assert PartialReport(progress=progress).progress == progress


@pytest.mark.parametrize("progress", ["half", True])
def test_progress_must_be_a_number(progress):
    with pytest.raises(TypeError, match="progress must be a number"):
        PartialReport(progress=progress)
