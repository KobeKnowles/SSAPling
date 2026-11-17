"""Measured values and aggregation: missing values never count as zero."""

from typing import Any

import pytest

from ssapling.core import ConfigError, UnknownComponentError
from ssapling.evaluate import InstanceResult, MeasuredValue, Summary, aggregate
from ssapling.tasks import Score


def result(value: MeasuredValue, *, digits: int = 1, status: Any = "terminated") -> InstanceResult:
    return InstanceResult(
        instance_id=f"i{id(value)}",
        collection="validation",
        descriptors={"digits": digits},
        record_id="r",
        status=status,
        values={"score": value},
    )


def mean_of_score(**extra: Any) -> dict[str, Any]:
    return {"measurement": "score", "aggregator": "mean", **extra}


def test_what_a_measurement_returns_becomes_a_measured_value():
    assert MeasuredValue.of(0.5) == MeasuredValue.available(0.5)
    assert MeasuredValue.of(None) == MeasuredValue.unavailable("the measurement gave no value")
    assert MeasuredValue.of(Score.local(1.0)) == MeasuredValue.available(1.0)
    assert MeasuredValue.of(Score.external(0.7, {"source": "x"})) == MeasuredValue.external(
        0.7, {"source": "x"}
    )
    assert MeasuredValue.of(Score.unavailable("hidden")) == MeasuredValue.unavailable("hidden")
    kept = MeasuredValue.failed("boom")
    assert MeasuredValue.of(kept) is kept


def test_a_measured_value_has_a_value_exactly_when_it_should():
    with pytest.raises(ValueError, match="a value is required"):
        MeasuredValue(status="available")
    with pytest.raises(ValueError, match="not allowed otherwise"):
        MeasuredValue(status="failed", value=0)
    with pytest.raises(ValueError, match="status must be"):
        MeasuredValue(status="lost", value=1)  # type: ignore[arg-type]


def test_missing_values_are_excluded_and_counted_never_taken_as_zero():
    results = [
        result(MeasuredValue.available(1.0)),
        result(MeasuredValue.external(0.5)),  # reported by a service: it counts
        result(MeasuredValue.unavailable("hidden")),
        result(MeasuredValue.failed("boom")),
    ]
    summary = aggregate(results, {"accuracy": mean_of_score()})["accuracy"]
    assert summary == Summary(value=0.75, contributed=2, excluded={"unavailable": 1, "failed": 1})


def test_an_aggregate_over_no_values_is_undefined():
    results = [result(MeasuredValue.unavailable()), result(MeasuredValue.failed())]
    summary = aggregate(results, {"accuracy": mean_of_score()})["accuracy"]
    assert summary.value is None  # not 0
    assert summary.contributed == 0
    assert aggregate([], {"n": {"measurement": "score", "aggregator": "count"}})["n"].value is None


def test_count_gives_how_many_values_contributed():
    results = [
        result(MeasuredValue.available(1.0)),
        result(MeasuredValue.available(0.0)),
        result(MeasuredValue.unavailable("hidden")),
    ]
    summary = aggregate(results, {"n": {"measurement": "score", "aggregator": "count"}})["n"]
    assert summary == Summary(value=2, contributed=2, excluded={"unavailable": 1})


def test_truncated_attempts_count_unless_excluded():
    results = [
        result(MeasuredValue.available(1.0)),
        result(MeasuredValue.available(0.0), status="truncated"),
    ]
    assert aggregate(results, {"a": mean_of_score()})["a"].value == 0.5
    excluded = aggregate(results, {"a": mean_of_score(exclude_truncated=True)})["a"]
    assert excluded == Summary(value=1.0, contributed=1, excluded={"truncated": 1})


def test_grouping_summarises_each_descriptor_value_and_keeps_the_pooled_summary():
    results = [
        result(MeasuredValue.available(1.0), digits=1),
        result(MeasuredValue.available(0.0), digits=2),
        result(MeasuredValue.available(0.0), digits=2),
        result(MeasuredValue.unavailable(), digits=2),
    ]
    summary = aggregate(results, {"a": mean_of_score(group_by="digits")})["a"]
    assert summary.value == pytest.approx(1 / 3)  # pooled: every instance weighs the same
    assert summary.groups == {
        "1": Summary(value=1.0, contributed=1),
        "2": Summary(value=0.0, contributed=2, excluded={"unavailable": 1}),
    }


def test_aggregations_must_name_registered_aggregators_and_valid_settings():
    with pytest.raises(UnknownComponentError, match="aggregator"):
        aggregate([], {"a": {"measurement": "score", "aggregator": "median"}})
    with pytest.raises(ConfigError):
        aggregate([], {"a": {"measurement": "score"}})


# Restricting an aggregation to some instances (where), and weighting its groups equally.
#
#   instance   group      digits   score
#   a          broad      1        1
#   b          broad      1        0
#   c          broad      1        1
#   d          broad      2        0
#   e          broad      3        unavailable
#   f          targeted   1        0
#   g          (none)     2        failed

SCORES: dict[str, tuple[dict[str, Any], MeasuredValue]] = {
    "a": ({"group": "broad", "digits": 1}, MeasuredValue.available(1)),
    "b": ({"group": "broad", "digits": 1}, MeasuredValue.available(0)),
    "c": ({"group": "broad", "digits": 1}, MeasuredValue.available(1)),
    "d": ({"group": "broad", "digits": 2}, MeasuredValue.available(0)),
    "e": ({"group": "broad", "digits": 3}, MeasuredValue.unavailable("not this case")),
    "f": ({"group": "targeted", "digits": 1}, MeasuredValue.available(0)),
    "g": ({"digits": 2}, MeasuredValue.failed("boom")),
}


def scored(names: str) -> list[InstanceResult]:
    """The instances named, one letter each, from the table above."""
    return [
        InstanceResult(
            instance_id=name,
            collection="validation",
            descriptors=SCORES[name][0],
            record_id=f"r-{name}",
            status="terminated",
            values={"score": SCORES[name][1]},
        )
        for name in names
    ]


def test_where_keeps_only_the_instances_whose_descriptors_match():
    """Given the table, when the mean is restricted to group broad, then only a-e count: 2/4,
    with e's unavailable value counted as excluded, and each digits group holds only broad ones.
    Catches: f counted, which gives 2/5 pooled and 2/4 at digits 1."""
    broad = mean_of_score(where={"group": "broad"})
    assert aggregate(scored("abcdefg"), {"a": broad})["a"] == Summary(
        value=0.5, contributed=4, excluded={"unavailable": 1}
    )
    grouped = aggregate(scored("abcdefg"), {"a": {**broad, "group_by": "digits"}})["a"]
    assert grouped.groups["1"].value == pytest.approx(2 / 3)
    assert grouped.groups["2"] == Summary(value=0.0, contributed=1)
    assert grouped.groups["3"] == Summary(value=None, contributed=0, excluded={"unavailable": 1})


def test_instances_left_out_by_where_are_neither_contributed_nor_excluded():
    """Given g's failed value, when the aggregation keeps group broad, then g is in no count.
    Catches: a filtered instance reported as failed, which would hide real failures."""
    summary = aggregate(scored("abcdefg"), {"a": mean_of_score(where={"group": "broad"})})["a"]
    assert "failed" not in summary.excluded
    assert summary.contributed + sum(summary.excluded.values()) == 5  # a-e, not f or g


def test_an_instance_without_the_descriptor_never_matches_where():
    """Given g alone, which has no group, when the aggregation keeps group broad, then nothing
    contributes and nothing is excluded. Catches: a missing descriptor read as a match."""
    summary = aggregate(scored("g"), {"a": mean_of_score(where={"group": "broad"})})["a"]
    assert summary == Summary(value=None, contributed=0)


def test_where_matches_every_descriptor_it_names():
    """Given where with group broad and digits 1, then a-c alone count: 2/3.
    Catches: only the first descriptor compared, which gives 2/4."""
    where = {"group": "broad", "digits": 1}
    summary = aggregate(scored("abcdefg"), {"a": mean_of_score(where=where)})["a"]
    assert summary.value == pytest.approx(2 / 3)
    assert summary.contributed == 3


def test_equal_weighting_is_the_mean_of_the_groups_values():
    """Given a-d grouped by digits, when weighted equally, then the value is the mean of digits
    1's 2/3 and digits 2's 0, which is 1/3, though pooled it is 2/4; the groups are reported as
    pooled grouping reports them, and contributed is still the 4 values.
    Catches: the pooled mean reported as the equal-weight one."""
    equal = aggregate(scored("abcd"), {"a": mean_of_score(group_by="digits", weighting="equal")})
    pooled = aggregate(scored("abcd"), {"a": mean_of_score(group_by="digits")})
    assert equal["a"].value == pytest.approx(1 / 3)
    assert pooled["a"].value == 0.5
    assert equal["a"].groups == pooled["a"].groups
    assert equal["a"].contributed == 4
    assert equal["a"].excluded == {}


def test_equal_weighting_leaves_out_a_group_with_no_value():
    """Given a-e grouped by digits, where digits 3 has only e's unavailable value, when weighted
    equally, then digits 3 is reported as undefined and left out of the mean: 1/3, with the
    unavailable value counted as excluded. Catches: an undefined group counted as 0 (2/9)."""
    summary = aggregate(
        scored("abcde"), {"a": mean_of_score(group_by="digits", weighting="equal")}
    )["a"]
    assert summary.value == pytest.approx(1 / 3)
    assert summary.groups["3"].value is None
    assert summary.contributed == 4
    assert summary.excluded == {"unavailable": 1}


def test_equal_weighting_with_no_group_value_is_undefined():
    """Given e alone, when weighted equally by digits, then the value is None.
    Catches: 0 reported when nothing could be measured."""
    summary = aggregate(scored("e"), {"a": mean_of_score(group_by="digits", weighting="equal")})
    assert summary["a"].value is None
    assert summary["a"].contributed == 0


def test_equal_weighting_needs_a_grouping():
    """Given weighting equal with no group_by, then the settings are refused, naming group_by.
    Catches: an equal-weight value that is silently the pooled one."""
    with pytest.raises(ConfigError, match="weighting 'equal' needs group_by"):
        aggregate(scored("abcd"), {"a": mean_of_score(weighting="equal")})


def test_where_and_equal_weighting_together():
    """Given the table grouped by digits and weighted equally, then with where group broad the
    value is (2/3 + 0) / 2 = 1/3; without it, digits 1 holds a, b, c and f, so it is
    (2/4 + 0) / 2 = 1/4. Catches: the filter applied to the value but not the groups, or the
    reverse."""
    equal = mean_of_score(group_by="digits", weighting="equal")
    broad = {**equal, "where": {"group": "broad"}}
    summaries = aggregate(scored("abcdefg"), {"broad": broad, "all": equal})
    assert summaries["broad"].value == pytest.approx(1 / 3)
    assert summaries["all"].value == pytest.approx(1 / 4)


def test_aggregations_without_the_new_settings_give_what_they_gave():
    """Given the table grouped by digits, then leaving where and weighting out gives the same
    summary as where {} and weighting pooled: the defaults change nothing.
    Catches: a changed default."""
    plain = mean_of_score(group_by="digits")
    explicit = {**plain, "where": {}, "weighting": "pooled"}
    summaries = aggregate(scored("abcdefg"), {"plain": plain, "explicit": explicit})
    assert summaries["plain"] == summaries["explicit"]
    assert summaries["plain"].value == pytest.approx(2 / 5)


def test_where_matches_a_tuple_descriptor_given_as_a_list():
    """Given results whose `lengths` descriptor is a tuple, when where gives it as a list, as a
    YAML or JSON file does, then the matching results are aggregated: 1 of 2.
    Catches: [2, 1] never equal to (2, 1), so a filter on a length pair matched nothing."""
    results = [
        InstanceResult(
            instance_id=name,
            collection="validation",
            descriptors={"lengths": lengths},
            record_id=f"r-{name}",
            status="terminated",
            values={"score": MeasuredValue.available(value)},
        )
        for name, lengths, value in [("a", (2, 1), 1), ("b", (2, 1), 0), ("c", (1, 2), 1)]
    ]
    summary = aggregate(results, {"a": mean_of_score(where={"lengths": [2, 1]})})["a"]
    assert summary == Summary(value=0.5, contributed=2)
