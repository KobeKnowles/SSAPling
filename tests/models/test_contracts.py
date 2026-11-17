"""The model contract: learning operations, learning signals, and descriptions."""

from typing import TYPE_CHECKING, Any

import pytest

from ssapling import SSAPlingError
from ssapling.models import (
    ActModel,
    BufferedModel,
    DescriptionMismatch,
    DescriptionMismatchError,
    Refusal,
    check_descriptions,
    filter_signals,
    supports,
)
from tests.fixtures.models import AsyncEcho, CountingLearner, GridModel, LookupModel, SlowThinker

# The fixtures implement the protocols structurally, inheriting from nothing.
# mypy checks these assignments; they never run. The last two must stay
# errors: strict mypy reports an unneeded "type: ignore", so this fails if the
# protocols ever stop telling the two ways of acting apart.
if TYPE_CHECKING:
    _lookup: ActModel = LookupModel()
    _async: ActModel = AsyncEcho()
    _learner: ActModel = CountingLearner()
    _buffered: BufferedModel = SlowThinker()
    _not_buffered: BufferedModel = LookupModel()  # type: ignore[assignment]
    _not_act: ActModel = SlowThinker()  # type: ignore[assignment]


class Declares:
    """A model whose declarations are given when it is made."""

    def __init__(self, **declarations: Any) -> None:
        self.__dict__.update(declarations)

    def act(self, observations: object) -> object:
        return observations

    def learn(self, operation: str, **payload: Any) -> dict[str, Any]:
        return {}


# Learning operations


def test_a_model_supports_the_learning_operations_it_lists():
    learner = CountingLearner()
    assert supports(learner, "learn_from_batch")
    assert supports(learner, "consolidate")
    assert not supports(learner, "learn_from_experience")


def test_a_model_without_learning_operations_learns_nothing():
    """A fixed rule or a frozen network is a model too."""
    assert not supports(LookupModel(), "learn_from_batch")
    assert not supports(SlowThinker(), "consolidate")


def test_learning_operations_are_an_open_set_of_names():
    model = Declares(learning_operations={"sleep_replay"})
    assert supports(model, "sleep_replay")


def test_a_single_string_is_not_a_list_of_learning_operations():
    with pytest.raises(TypeError, match=r"Declares.learning_operations must be a collection"):
        supports(Declares(learning_operations="consolidate"), "consolidate")


def test_listing_learning_operations_needs_a_learn_method():
    class NoLearn:
        learning_operations = frozenset({"consolidate"})

        def act(self, observations: object) -> object:
            return observations

    with pytest.raises(TypeError, match="NoLearn lists learning operations but has no learn"):
        supports(NoLearn(), "consolidate")


def test_learn_returns_statistics():
    learner = CountingLearner()
    assert learner.learn("learn_from_batch", batch=["a", "b", "c"]) == {
        "updates": 1,
        "examples_seen": 3,
    }


# Learning signals


SIGNALS = {"reward": 1.0, "supervision": "7", "feedback": "Carry the 1.", "demonstration": []}


def test_only_declared_signals_are_passed():
    assert filter_signals(CountingLearner(), SIGNALS) == {"reward": 1.0, "supervision": "7"}


def test_teacher_feedback_is_not_turned_into_reward():
    passed = filter_signals(Declares(signals={"reward"}), SIGNALS)
    assert passed == {"reward": 1.0}
    assert "feedback" not in passed


def test_a_model_that_declares_no_signals_receives_none():
    assert filter_signals(LookupModel(), SIGNALS) == {}


def test_a_declared_signal_that_is_not_available_is_simply_absent():
    assert filter_signals(CountingLearner(), {"feedback": "..."}) == {}


def test_signals_are_checked():
    with pytest.raises(TypeError, match=r"Declares.signals must be a collection of names"):
        filter_signals(Declares(signals="reward"), SIGNALS)
    with pytest.raises(TypeError, match="signals must be a mapping"):
        filter_signals(CountingLearner(), ["reward"])  # type: ignore[arg-type]


def test_the_filtered_signals_reach_the_model():
    learner = CountingLearner()
    learner.learn("learn_from_batch", batch=[], signals=filter_signals(learner, SIGNALS))
    assert learner.last_signals == {"reward": 1.0, "supervision": "7"}


# Descriptions


def test_matching_descriptions_pass():
    check_descriptions(CountingLearner(), observations="a string of digits", actions="one digit")


def test_a_mismatch_is_reported_with_both_descriptions():
    with pytest.raises(DescriptionMismatchError) as caught:
        check_descriptions(CountingLearner(), observations="a string of digits", actions="a grid")
    assert str(caught.value) == (
        "Model CountingLearner cannot handle the environment's descriptions:\n"
        "  - actions: the model handles 'one digit', the environment gives 'a grid'"
    )
    assert caught.value.mismatches == (DescriptionMismatch("action", "a grid", "one digit"),)
    assert isinstance(caught.value, ValueError)
    assert isinstance(caught.value, SSAPlingError)


def test_every_mismatch_is_reported_at_once_with_the_participant():
    with pytest.raises(DescriptionMismatchError) as caught:
        check_descriptions(
            CountingLearner(), observations="a grid", actions="a grid", participant="player_1"
        )
    message = str(caught.value)
    assert message.startswith(
        "Model CountingLearner cannot handle the environment's descriptions "
        "as participant 'player_1':"
    )
    assert [m.kind for m in caught.value.mismatches] == ["observation", "action"]


def test_a_model_can_decide_for_itself_what_it_accepts():
    check_descriptions(GridModel(size=30), observations={"max_size": 30}, actions="a grid")
    with pytest.raises(DescriptionMismatchError) as caught:
        check_descriptions(GridModel(size=10), observations={"max_size": 30}, actions="a grid")
    assert str(caught.value).endswith("observations: the model does not accept {'max_size': 30}")


def test_nothing_is_compared_where_either_side_gives_no_description():
    check_descriptions(LookupModel(), observations="anything", actions="anything")
    check_descriptions(CountingLearner(), observations=None, actions=None)


def test_accepts_description_must_be_a_method():
    with pytest.raises(TypeError, match=r"Declares.accepts_description must be a method"):
        check_descriptions(Declares(accepts_description=True), observations="x")


def test_a_refusal_is_false_and_says_why():
    refusal = Refusal("it reads no utterance channel")
    assert not refusal
    assert refusal.reason == "it reads no utterance channel"


def test_a_models_reason_for_refusing_is_in_the_mismatch():
    """Given a model whose accepts_description refuses with a reason, when it is checked against
    an environment it can't handle, then the mismatch carries the reason, and the error says it.
    Catches: a refusal that only says no, so the user can't tell which field is missing."""
    model = Declares(
        accepts_description=lambda kind, description: Refusal(f"no {kind} field 'speaker'")
    )
    with pytest.raises(DescriptionMismatchError) as caught:
        check_descriptions(model, observations={"fields": ["cells"]})
    (mismatch,) = caught.value.mismatches
    assert mismatch.reason == "no observation field 'speaker'"
    assert str(caught.value).endswith(
        "observations: the model does not accept {'fields': ['cells']}: "
        "no observation field 'speaker'"
    )
