"""TeacherRequest, TeacherResponse, and the response statuses."""

import pickle
from typing import Any

import pytest

from ssapling.teachers import TeacherRequest, TeacherResponse

REVIEW = TeacherRequest(
    id="review-12",
    operation="review_action",
    context={"state": "4"},
    proposals={"student": "7"},
    episode="ep-3",
    participant="student",
    step=12,
)

# Requests


def test_a_request_holds_what_is_asked_and_where_it_comes_from():
    assert (REVIEW.id, REVIEW.operation) == ("review-12", "review_action")
    assert REVIEW.context == {"state": "4"}
    assert REVIEW.proposals == {"student": "7"}
    assert (REVIEW.episode, REVIEW.participant, REVIEW.step) == ("ep-3", "student", 12)


def test_context_proposals_and_identity_are_optional():
    request = TeacherRequest(id="propose-1", operation="propose_instances")
    assert request.context == {}
    assert request.proposals == {}
    assert (request.episode, request.participant, request.step) == (None, None, None)


def test_a_request_captures_its_context_when_it_is_made():
    context: dict[str, Any] = {"state": "4"}
    proposals = {"student": "7"}
    request = TeacherRequest(
        id="review-12", operation="review_action", context=context, proposals=proposals
    )
    context["state"] = "47"  # the episode moves on
    context["hint"] = "added later"
    proposals["student"] = "8"
    assert request.context == {"state": "4"}
    assert request.proposals == {"student": "7"}


@pytest.mark.parametrize("field", ["id", "operation"])
@pytest.mark.parametrize("value", ["", None, 12])
def test_id_and_operation_must_be_non_empty_strings(field, value):
    arguments: dict[str, Any] = {"id": "review-12", "operation": "review_action", field: value}
    with pytest.raises(ValueError, match="must be a non-empty string"):
        TeacherRequest(**arguments)


@pytest.mark.parametrize("field", ["context", "proposals"])
def test_context_and_proposals_must_be_mappings_with_string_keys(field):
    not_a_mapping: dict[str, Any] = {field: ["state"]}
    with pytest.raises(TypeError, match=f"{field} must be a mapping"):
        TeacherRequest(id="r", operation="op", **not_a_mapping)
    number_keys: dict[str, Any] = {field: {1: "x"}}
    with pytest.raises(TypeError, match=f"the keys of {field} must be strings"):
        TeacherRequest(id="r", operation="op", **number_keys)


def test_step_is_a_non_negative_integer():
    assert TeacherRequest(id="r", operation="op", step=0).step == 0
    with pytest.raises(ValueError, match="step must be 0 or more"):
        TeacherRequest(id="r", operation="op", step=-1)
    for bad in (True, 1.5, "3"):
        with pytest.raises(TypeError, match="step must be an integer"):
            TeacherRequest(id="r", operation="op", step=bad)  # type: ignore[arg-type]


def test_episode_and_participant_must_be_non_empty_strings_when_given():
    with pytest.raises(ValueError, match="an episode id"):
        TeacherRequest(id="r", operation="op", episode="")
    with pytest.raises(ValueError, match="a participant id"):
        TeacherRequest(id="r", operation="op", participant=3)  # type: ignore[arg-type]


# Responses


def test_an_answer_holds_contributions_by_kind_then_agent():
    response = TeacherResponse.answered(
        REVIEW,
        {"supervision": {"student": "8"}, "feedback": {"student": "Carry the 1."}},
        detail={"cost": 0.01},
    )
    assert response.request_id == "review-12"
    assert response.status == "answered"
    assert response.contributions["feedback"] == {"student": "Carry the 1."}
    assert response.detail == {"cost": 0.01}


def test_an_answer_may_contribute_nothing():
    assert TeacherResponse.answered(REVIEW).contributions == {}


def test_the_constructors_copy_the_request_id():
    for build in (
        TeacherResponse.answered,
        TeacherResponse.abstained,
        TeacherResponse.failed,
        TeacherResponse.cancelled,
    ):
        assert build(REVIEW).request_id == "review-12"


@pytest.mark.parametrize("status", ["abstained", "failed", "cancelled"])
def test_only_an_answer_or_a_stale_answer_has_contributions(status):
    """A response without an answer has nothing that could be applied, so it cannot approve."""
    with pytest.raises(ValueError, match=f"status '{status}' has no contributions"):
        TeacherResponse(
            request_id="review-12", status=status, contributions={"feedback": {"student": "ok"}}
        )
    response = TeacherResponse(request_id="review-12", status=status, detail="the reason")
    assert response.contributions == {}
    assert response.detail == "the reason"


def test_unknown_statuses_are_rejected():
    with pytest.raises(ValueError, match="status must be one of 'answered', 'abstained'"):
        TeacherResponse(request_id="review-12", status="approved")  # type: ignore[arg-type]


def test_contributions_must_be_keyed_by_kind_then_by_agent():
    with pytest.raises(TypeError, match=r"contributions\['feedback'\] must map agent ids"):
        TeacherResponse.answered(REVIEW, {"feedback": "Carry the 1."})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="the keys of contributions must be strings"):
        TeacherResponse.answered(REVIEW, {1: {"student": "x"}})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match=r"the keys of contributions\['feedback'\] must be strings"):
        TeacherResponse.answered(REVIEW, {"feedback": {0: "x"}})  # type: ignore[dict-item]


def test_contributions_are_copied_when_the_response_is_made():
    feedback = {"student": "Carry the 1."}
    contributions = {"feedback": feedback}
    response = TeacherResponse.answered(REVIEW, contributions)
    feedback["student"] = "changed"
    contributions["supervision"] = {"student": "8"}
    assert response.contributions == {"feedback": {"student": "Carry the 1."}}


def test_a_late_answer_is_marked_stale_and_keeps_its_contributions_for_the_record():
    answer = TeacherResponse.answered(REVIEW, {"intervention": {"student": "8"}}, detail="d")
    stale = answer.as_stale()
    assert stale.status == "stale"
    assert stale.contributions == answer.contributions
    assert (stale.request_id, stale.detail) == ("review-12", "d")
    assert answer.status == "answered"  # unchanged


@pytest.mark.parametrize("status", ["abstained", "failed", "cancelled", "stale"])
def test_only_an_answer_can_be_marked_stale(status):
    response = TeacherResponse(request_id="review-12", status=status)
    with pytest.raises(ValueError, match="only an answered response can be marked stale"):
        response.as_stale()


def test_requests_and_responses_pickle():
    """So they can be sent to and from a teacher in another process (Stage 7)."""
    answer = TeacherResponse.answered(REVIEW, {"feedback": {"student": "ok"}})
    assert pickle.loads(pickle.dumps(REVIEW)) == REVIEW
    assert pickle.loads(pickle.dumps(answer)) == answer


def test_requests_and_responses_are_immutable():
    with pytest.raises(AttributeError):
        REVIEW.operation = "explain"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        TeacherResponse.answered(REVIEW).status = "failed"  # type: ignore[misc]
