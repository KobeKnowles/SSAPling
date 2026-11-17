"""Aggregation: per-instance values combined into summaries, with exclusions reported.

An **aggregator** is a function over a list of values, registered by name in
``aggregators``; ``"count"`` and ``"mean"`` are built in. An **aggregation**
names a measurement, an aggregator, and optionally a descriptor to group
by, the descriptor values an instance must have to count (``where``), and
how its groups are weighted. The machinery, not the aggregator, decides
which values contribute:

- Only instances that match ``where`` are aggregated. The rest are in no
  count, neither contributed nor excluded.
- Unavailable and failed values never count, as zero or otherwise. They are
  left out, and how many is reported by reason.
- Truncated attempts count, since the task scored whatever outcome was
  reached, unless an aggregation excludes them.
- An aggregate over no values is ``None``, undefined, never 0.
- With ``weighting="equal"``, the value is the mean of the groups' values,
  each group weighing the same however many instances it holds; a group
  with no value is left out of that mean.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core import Registry, setting, validate_settings
from ..data._matching import matches
from .measurement import InstanceResult

__all__ = ["AggregationSettings", "Summary", "aggregate", "aggregators"]


@dataclass(frozen=True)
class AggregationSettings:
    """One aggregation: which measurement's values, of which instances, combined how, grouped by
    what, and weighted how."""

    measurement: str = setting(description="The measurement whose values are combined.")
    aggregator: str = setting(description="The registered aggregator, such as 'mean'.")
    group_by: str | None = setting(
        None, description="A descriptor: also summarise each of its values separately."
    )
    exclude_truncated: bool = setting(
        False, description="Leave out the values of attempts that were truncated."
    )
    where: dict[str, Any] = setting(
        default_factory=dict,
        description=(
            "Descriptor values an instance must have to be aggregated, such as "
            "{'group': 'broad'}. Other instances are in no count."
        ),
    )
    weighting: Literal["pooled", "equal"] = setting(
        "pooled",
        description=(
            "'pooled': every instance weighs the same. 'equal': the value is the mean of the "
            "groups' values, every group weighing the same; needs group_by."
        ),
    )

    def validate(self) -> None:
        if self.weighting == "equal" and self.group_by is None:
            raise ValueError("weighting 'equal' needs group_by: the groups it weighs equally")


@dataclass(frozen=True, slots=True, kw_only=True)
class Summary:
    """The result of one aggregation.

    Attributes:
        value: What the aggregator gave, or ``None`` when no value
            contributed: undefined, not 0. With equal weighting, the mean
            of the groups' values, ``None`` when no group has one.
        contributed: How many values were combined (with equal weighting,
            in all the groups together).
        excluded: How many were left out, by reason: ``"unavailable"``,
            ``"failed"``, ``"truncated"``. Reasons with none are omitted.
            Instances that don't match ``where`` are not among them.
        groups: With ``group_by``: a summary for each value of the
            descriptor, keyed by the value as a string. Instances without
            the descriptor are under ``"None"``. Empty otherwise.
    """

    value: Any
    contributed: int
    excluded: Mapping[str, int] = field(default_factory=dict)
    groups: Mapping[str, "Summary"] = field(default_factory=dict)


def aggregate(
    instances: Sequence[InstanceResult],
    aggregations: Mapping[str, AggregationSettings | Mapping[str, Any]],
) -> dict[str, Summary]:
    """Combine per-instance values into a summary for each aggregation.

    It needs only the per-instance results, so aggregates can be recomputed
    from stored results without running the model again.

    Example::

        aggregate(results.instances, {"accuracy": {"measurement": "score", "aggregator": "mean"}})

    Raises:
        ConfigError: An aggregation's settings are invalid.
        UnknownComponentError: An aggregation names no registered aggregator.
    """
    summaries = {}
    for name, settings in aggregations.items():
        aggregation = validate_settings(AggregationSettings, settings)
        combine = aggregators.get(aggregation.aggregator)
        matching = [r for r in instances if matches(r.descriptors, aggregation.where)]
        summary = _summarise(matching, aggregation, combine)
        if aggregation.group_by is not None:
            groups: dict[str, list[InstanceResult]] = {}
            for result in matching:
                key = str(result.descriptors.get(aggregation.group_by))
                groups.setdefault(key, []).append(result)
            summary = dataclasses.replace(
                summary,
                groups={k: _summarise(rs, aggregation, combine) for k, rs in groups.items()},
            )
        if aggregation.weighting == "equal":  # validated: group_by is set
            values = [group.value for group in summary.groups.values() if group.value is not None]
            summary = dataclasses.replace(
                summary, value=sum(values) / len(values) if values else None
            )
        summaries[name] = summary
    return summaries


def _summarise(
    instances: Sequence[InstanceResult],
    aggregation: AggregationSettings,
    combine: Callable[[list[Any]], Any],
) -> Summary:
    values: list[Any] = []
    excluded: dict[str, int] = {}
    for result in instances:
        measured = result.values[aggregation.measurement]
        reason: str
        if measured.status in ("unavailable", "failed"):
            reason = measured.status
        elif aggregation.exclude_truncated and result.status == "truncated":
            reason = "truncated"
        else:
            values.append(measured.value)
            continue
        excluded[reason] = excluded.get(reason, 0) + 1
    return Summary(
        value=combine(values) if values else None, contributed=len(values), excluded=excluded
    )


def count(values: list[Any]) -> int:
    """How many values contributed."""
    return len(values)


def mean(values: list[Any]) -> Any:
    """The mean of the values, pooled: every instance weighs the same. True counts as 1."""
    return sum(values) / len(values)


aggregators: Registry[Callable[[list[Any]], Any]] = Registry("aggregator")
"""Registered aggregators. Also available as ``ssapling.registries.aggregators``.

An aggregator is registered as the function itself: it takes the list of
contributing values, never empty, and returns the aggregate. ``"count"``
and ``"mean"`` are built in.
"""
aggregators.register("count", count)
aggregators.register("mean", mean)
