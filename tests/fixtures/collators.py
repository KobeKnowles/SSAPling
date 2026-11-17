"""Toy collators and codecs for tests, at module level so worker processes can import them."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ssapling.core import setting
from ssapling.data import JsonCodec, codecs, collators


def total(batch: Sequence[int]) -> int:
    return sum(batch)


def with_process(batch: Sequence[Any]) -> tuple[int, list[Any]]:
    """The batch, and the id of the process that collated it."""
    return os.getpid(), list(batch)


def refuse(batch: Sequence[Any]) -> Any:
    raise ValueError(f"cannot collate {list(batch)}")


@dataclass(frozen=True)
class PadSettings:
    width: int = setting(4, ge=1, description="Pad every row to this many items.")
    fill: int = setting(0, description="The value to pad with.")


class Pad:
    """Pads rows of numbers to one width, as a collator for grids might."""

    Settings = PadSettings

    def __init__(self, config: PadSettings) -> None:
        self.config = config

    def __call__(self, batch: Sequence[Sequence[int]]) -> list[list[int]]:
        width, fill = self.config.width, self.config.fill
        return [list(row) + [fill] * (width - len(row)) for row in batch]


class SetCodec:
    """Handles sets, and passes everything else to the json codec."""

    def __init__(self) -> None:
        self.json = JsonCodec()

    def encode(self, value: Any) -> Any:
        if isinstance(value, set | frozenset):
            return {"__set__": sorted(self.json.encode(item) for item in value)}
        if isinstance(value, dict):
            return {key: self.encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.encode(item) for item in value]
        return self.json.encode(value)

    def decode(self, value: Any) -> Any:
        return self.json.decode(self._sets(value))

    def _sets(self, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"__set__"}:
                return set(value["__set__"])
            return {key: self._sets(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sets(item) for item in value]
        return value


def register_components() -> None:
    """What a downstream project's register_components does: register its components."""
    collators.register("pad", Pad)
    codecs.register("sets", SetCodec)
