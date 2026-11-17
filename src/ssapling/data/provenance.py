"""Provenance: where an instance or record came from, in plain values.

Data does not choose seeds; the coordinating code supplies them and Data
records them. Provenance holds enough to trace an item back to its source,
or to make it again: the source and its settings, the collection and
sampler, the parameters it was generated with, and its seed or position.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core._declarations import check_identifier, check_int, keyed
from ..core._plain import plain_copy
from ..core.errors import Problem

__all__ = ["Provenance"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Provenance:
    """Where an item came from: enough to trace it, or to make it again.

    Attributes:
        source: The name of the source, such as the task's registered name
            or the name of a published dataset.
        source_settings: The source's settings, such as the task's, as
            plain values: ``settings_to_dict(task.config)``.
        collection: Optional. The name of the collection the item belongs
            to.
        sampler: Optional. The name of the sampler that selected the item.
        sampler_settings: The sampler's settings, as plain values, given
            only with a sampler.
        params: The parameters a generated item was made with, as passed to
            the task's ``generate``. A stateful sampler may choose them
            differently later, so they are kept as they were.
        seed: The seed the item was drawn or generated with.
        position: The item's position: its index in an indexable source, or
            its place in a stream.

    At least one of ``seed`` and ``position`` is required. Every value is
    plain (``None``, booleans, numbers, strings, lists, and mappings with
    string keys), so provenance can be stored whatever codec the payloads
    use. As with ``settings_to_dict``, tuples become lists.

    For example, the instance a generated task makes with
    ``task.generate(params=params, seed=seed)`` is made again by building
    the same task from ``source`` and ``source_settings`` and calling it
    with the same ``params`` and ``seed``.
    """

    source: str
    source_settings: Mapping[str, Any] = field(default_factory=dict)
    collection: str | None = None
    sampler: str | None = None
    sampler_settings: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None
    position: int | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        check_identifier(self.source, "source")
        for name in ("collection", "sampler"):
            if getattr(self, name) is not None:
                check_identifier(getattr(self, name), name)
        for name in ("source_settings", "sampler_settings", "params"):
            set_(self, name, _plain_mapping(getattr(self, name), name))
        if self.sampler_settings and self.sampler is None:
            raise ValueError("sampler_settings were given without a sampler; name the sampler")

        if self.seed is None and self.position is None:
            raise ValueError(
                "provenance needs a seed or a position (or both), so the item can be "
                "traced or made again"
            )
        if self.seed is not None:
            check_int(self.seed, "seed")
        if self.position is not None:
            check_int(self.position, "position", minimum=0)


def _plain_mapping(value: object, what: str) -> dict[str, Any]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        raise TypeError(
            f"{what} must be a mapping of plain values; turn a settings instance into "
            f"one with settings_to_dict(), got {type(value).__name__}"
        )
    problems: list[Problem] = []
    plain: dict[str, Any] = plain_copy(keyed(value, what, "names"), what, problems)
    if problems:
        raise TypeError("provenance holds plain values only: " + "; ".join(map(str, problems)))
    return plain
