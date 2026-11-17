"""The internal plain-value helper, shared by settings and provenance."""

from ssapling.core._plain import plain_copy
from ssapling.core.errors import Problem


def test_plain_values_are_copied_with_tuples_as_lists():
    value = {"shape": (2, 3), "nested": {"weights": [0.5, None, True, "a"]}}
    problems: list[Problem] = []
    copy = plain_copy(value, "params", problems)
    assert copy == {"shape": [2, 3], "nested": {"weights": [0.5, None, True, "a"]}}
    assert copy["nested"] is not value["nested"]
    assert problems == []


def test_every_part_that_is_not_plain_is_reported_with_its_path():
    problems: list[Problem] = []
    plain_copy({"a": [1, {2}], "b": {3: "x"}, "c": b"raw"}, "params", problems)
    assert [p.path for p in problems] == ["params.a[1]", "params.b", "params.c"]
    assert problems[0].message.startswith("expected a plain value")
    assert problems[1].message == "keys must be strings, got int 3"
