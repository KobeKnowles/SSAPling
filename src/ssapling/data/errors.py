"""Errors specific to data."""

from collections.abc import Iterable

from ..core.errors import SSAPlingError


class SourceExhaustedError(SSAPlingError, LookupError):
    """A sampler found no usable instance within the allowed number of draws.

    Raised instead of retrying forever, when every draw gave an instance of
    a problem already reserved by a collection (or, while building, one that
    did not match ``where``). The source may be used up, or the reserved
    collections may cover most of it.

    Attributes:
        sampler: The sampler's class name.
        reserved: The names of the collections whose problems were reserved.
        draws: How many draws were made for the one instance.
        building: The collection being built, or ``None`` for the remainder.
    """

    def __init__(
        self, sampler: str, reserved: Iterable[str], draws: int, *, building: str | None
    ) -> None:
        self.sampler = sampler
        self.reserved = tuple(reserved)
        self.draws = draws
        self.building = building
        target = f"for collection {building!r}" if building else "for the remainder"
        collections = ", ".join(repr(name) for name in self.reserved) or "none"
        super().__init__(
            f"{sampler} drew {draws} times {target} without finding an instance outside the "
            f"reserved collections ({collections}); the source may be used up. Raise "
            "max_draws, use a larger source, or reserve less."
        )


class UnknownCollectionError(SSAPlingError, LookupError):
    """A collection was asked for by a name that has not been built.

    Attributes:
        name: The name asked for.
        available: The names of the collections that exist, in the order built.
    """

    def __init__(self, name: str, available: Iterable[str]) -> None:
        self.name = name
        self.available = tuple(available)
        built = ", ".join(repr(n) for n in self.available) or "none"
        super().__init__(f"No collection {name!r}; the collections are: {built}.")


class HeldOutError(SSAPlingError, LookupError):
    """A held-out collection was asked for without an evaluation purpose.

    Attributes:
        name: The collection's name.
        purpose: The purpose that was declared, or ``None``.
    """

    def __init__(self, name: str, purpose: str | None) -> None:
        self.name = name
        self.purpose = purpose
        declared = f"purpose {purpose!r}" if purpose else "no purpose"
        super().__init__(
            f"Collection {name!r} is held out, so it is served only with "
            f"purpose='evaluation', not with {declared}."
        )


class UnknownRecordError(SSAPlingError, LookupError):
    """A record was asked for that the store does not hold.

    Attributes:
        record_id: The id asked for.
        store: Where the store is.
    """

    def __init__(self, record_id: str, store: str) -> None:
        self.record_id = record_id
        self.store = store
        super().__init__(f"The store at {store} holds no record {record_id!r}.")


class UnknownDocumentError(SSAPlingError, LookupError):
    """A document was asked for that the store does not hold.

    Attributes:
        name: The name asked for.
        store: Where the store is.
    """

    def __init__(self, name: str, store: str) -> None:
        self.name = name
        self.store = store
        super().__init__(f"The store at {store} holds no document {name!r}.")


class StoreError(SSAPlingError, ValueError):
    """A store's files are not what SSAPling writes: not a store, a newer format, or damaged.

    The message names the file, and the line where there is one.
    """
