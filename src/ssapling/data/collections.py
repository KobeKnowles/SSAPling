"""Collections: named sets of instances drawn from one source, which never overlap.

Each collection reserves the problems it holds, by their ``problem_key``
(the equivalence key, or the id when there is none), and every later draw
rejects reserved problems. So a validation set and a test set never share a
problem, and the **remainder**, a stream of everything else drawn on the
fly, never yields one either. Every item served carries its provenance.

A **dataset** is finite and re-readable: a built collection is a tuple. A
**stream** may be unbounded: the remainder is an iterator. Both are
iterated the same way; check ``isinstance(items, collections.abc.Sized)``
or ``Sequence`` before counting or indexing.
"""

import dataclasses
import random
from collections.abc import Hashable, Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any

from ..core import settings_to_dict
from ..core._declarations import check_identifier, check_int, keyed, name_of
from ..tasks import Instance
from ..tasks import supports as task_supports
from ._matching import matches
from .errors import HeldOutError, SourceExhaustedError, UnknownCollectionError
from .provenance import Provenance
from .sampling import Sample, Sampler

__all__ = ["Collections", "Drawn"]


@dataclass(frozen=True, slots=True)
class Drawn:
    """An instance as Data serves it: with where it came from.

    Attributes:
        instance: The instance.
        provenance: Its source, collection, sampler, parameters and seed.
    """

    instance: Instance
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class _Collection:
    items: tuple[Drawn, ...]
    held_out: bool


class Collections:
    """The collections drawn from one source.

    Example::

        collections = Collections(task, source_name="addition")
        collections.build("validation", sampler=uniform, size=1000, seed=1)
        collections.build("test", sampler=uniform, size=1000, seed=2, held_out=True)
        training = collections.remainder(sampler=curriculum, seed=3)   # drawn on the fly

        collections.get("validation")                     # a tuple of Drawn
        collections.get("test", purpose="evaluation")     # held out: needs this purpose

    Args:
        source: Where the instances come from: anything with the task
            feature ``"generate"`` or ``"enumerate"``.
        source_name: The source's name in provenance, such as its
            registered name. Defaults to its class name. Give the registered
            name to make every item rebuildable through the registry.
        source_settings: The source's settings in provenance. Defaults to
            its ``config`` (the validated settings a registry gives a
            component), if it has one.

    Sources that arrive already divided, such as separately published
    training and evaluation sets, are separate sources, each with its own
    ``Collections``.
    """

    def __init__(
        self,
        source: object,
        *,
        source_name: str | None = None,
        source_settings: Mapping[str, Any] | None = None,
    ) -> None:
        if not (task_supports(source, "generate") or task_supports(source, "enumerate")):
            raise TypeError(
                f"{name_of(source)} is not a source: it has neither generate(params, seed) "
                "nor ids() and get(id)"
            )
        self._source = source
        self._source_name = source_name or name_of(source)
        self._source_settings = (
            keyed(source_settings, "source_settings", "names")
            if source_settings is not None
            else _settings_of(source)
        )
        self._collections: dict[str, _Collection] = {}
        self._reserved: set[Hashable] = set()

    @property
    def source(self) -> object:
        """The source every collection is drawn from."""
        return self._source

    def names(self) -> tuple[str, ...]:
        """The names of the built collections, in the order they were built."""
        return tuple(self._collections)

    def held_out(self, name: str) -> bool:
        """Whether the collection ``name`` is held out."""
        return self._collection(name).held_out

    def build(
        self,
        name: str,
        *,
        sampler: Sampler,
        size: int,
        seed: int,
        held_out: bool = False,
        where: Mapping[str, Any] | None = None,
        max_draws: int = 1000,
    ) -> tuple[Drawn, ...]:
        """Draw a collection of ``size`` distinct problems and reserve them.

        Each draw gets its own seed, derived from ``seed``, so the same
        sampler, seed and earlier collections give the same collection.
        A draw is rejected when its problem is reserved, by an earlier
        collection or by this one, or when its descriptors do not match
        ``where``. If ``max_draws`` draws in a row are rejected, building
        stops with an error and reserves nothing.

        Args:
            name: The collection's name, new in these collections.
            sampler: Picks the instances; it must draw from this source.
            size: How many instances.
            seed: Seeds every draw.
            held_out: Serve the collection only with
                ``purpose="evaluation"`` (see ``get``).
            where: Optional. Descriptor values every instance must have,
                such as ``{"digits": 3}``.
            max_draws: How many draws in a row may be rejected before giving
                up.

        Returns:
            The collection: a tuple of ``Drawn``, each with its provenance.

        Raises:
            SourceExhaustedError: ``max_draws`` draws in a row were rejected.
            ValueError: ``name`` is taken or empty, ``size`` is negative,
                ``max_draws`` is less than 1, or the sampler draws from
                another source.
            TypeError: An argument of the wrong type, or the sampler did not
                return a ``Sample``.
        """
        self._check_new_name(name)
        check_int(size, "size", minimum=0)
        if where is not None:
            where = keyed(where, "where", "descriptor names")
        self._check_sampler(sampler, seed, max_draws)

        rng = random.Random(seed)
        taken: set[Hashable] = set()
        items: list[Drawn] = []
        while len(items) < size:
            sample, draw_seed = self._draw(
                sampler, rng, max_draws, taken=taken, where=where, building=name
            )
            taken.add(sample.instance.problem_key)
            items.append(self._drawn(sample, draw_seed, sampler, collection=name))

        self._reserved |= taken
        self._collections[name] = _Collection(tuple(items), held_out=bool(held_out))
        return self._collections[name].items

    def add(
        self, name: str, items: Iterable[Drawn], *, held_out: bool = False
    ) -> tuple[Drawn, ...]:
        """Add a collection drawn elsewhere, such as one loaded from storage, and reserve it.

        The same rules hold as for a built collection: at most one instance
        per problem, and no problem another collection has reserved.

        Raises:
            ValueError: ``name`` is taken or empty, two items are the same
                problem, or an item's problem is already reserved.
            TypeError: An item is not a ``Drawn``.
        """
        self._check_new_name(name)
        added = tuple(items)
        taken: set[Hashable] = set()
        for drawn in added:
            if not isinstance(drawn, Drawn):
                raise TypeError(f"a collection holds Drawn items, got {drawn!r}")
            key = drawn.instance.problem_key
            if key in taken or key in self._reserved:
                raise ValueError(
                    f"collection {name!r}: instance {drawn.instance.id!r} is problem {key!r}, "
                    "which is already in this collection or reserved by another"
                )
            taken.add(key)
        self._reserved |= taken
        self._collections[name] = _Collection(added, held_out=bool(held_out))
        return added

    def get(self, name: str, *, purpose: str | None = None) -> tuple[Drawn, ...]:
        """The collection ``name``.

        A held-out collection is served only to a consumer that declares
        ``purpose="evaluation"``. How strictly that is used is the
        downstream project's choice.

        Raises:
            UnknownCollectionError: No collection is named ``name``.
            HeldOutError: The collection is held out and ``purpose`` is not
                ``"evaluation"``.
        """
        collection = self._collection(name)
        if collection.held_out and purpose != "evaluation":
            raise HeldOutError(name, purpose)
        return collection.items

    def remainder(self, *, sampler: Sampler, seed: int, max_draws: int = 1000) -> Iterator[Drawn]:
        """A stream of instances outside every collection, drawn on the fly.

        It never ends by itself. Each draw is checked against the
        collections as they are at that moment, so a collection built later
        is excluded from then on. The same problem may be drawn more than
        once: the remainder reserves nothing. Its items' provenance has no
        collection.

        Raises:
            SourceExhaustedError: When iterated: ``max_draws`` draws in a row
                gave reserved problems.
            ValueError, TypeError: As for ``build``, when called.
        """
        self._check_sampler(sampler, seed, max_draws)
        return self._remainder(sampler, seed, max_draws)

    def _remainder(self, sampler: Sampler, seed: int, max_draws: int) -> Iterator[Drawn]:
        rng = random.Random(seed)
        while True:
            sample, draw_seed = self._draw(
                sampler, rng, max_draws, taken=frozenset(), where=None, building=None
            )
            yield self._drawn(sample, draw_seed, sampler, collection=None)

    def _draw(
        self,
        sampler: Sampler,
        rng: random.Random,
        max_draws: int,
        *,
        taken: AbstractSet[Hashable],
        where: Mapping[str, Any] | None,
        building: str | None,
    ) -> tuple[Sample, int]:
        """The first acceptable sample within ``max_draws`` draws, and its seed.

        A sample is rejected when its problem is reserved, or already
        ``taken`` by the collection being built.
        """
        for _ in range(max_draws):
            draw_seed = rng.getrandbits(63)
            sample = sampler.draw(seed=draw_seed)
            if not isinstance(sample, Sample):
                raise TypeError(
                    f"{name_of(sampler)}.draw must return a Sample, got {type(sample).__name__}"
                )
            key = sample.instance.problem_key
            if key in self._reserved or key in taken:
                continue
            if where is not None and not _matches(sample.instance, where):
                continue
            return sample, draw_seed
        raise SourceExhaustedError(name_of(sampler), self.names(), max_draws, building=building)

    def _drawn(
        self, sample: Sample, seed: int, sampler: Sampler, *, collection: str | None
    ) -> Drawn:
        provenance = Provenance(
            source=self._source_name,
            source_settings=self._source_settings,
            collection=collection,
            sampler=name_of(sampler),
            sampler_settings=_settings_of(sampler),
            params=sample.params,
            seed=seed,
            position=sample.position,
        )
        return Drawn(sample.instance, provenance)

    def _check_sampler(self, sampler: object, seed: object, max_draws: object) -> None:
        if not callable(getattr(sampler, "draw", None)):
            raise TypeError(f"{name_of(sampler)} has no draw method, so it is not a sampler")
        drawn_from = getattr(sampler, "source", self._source)
        if drawn_from is not self._source:
            raise ValueError(
                f"{name_of(sampler)} draws from {name_of(drawn_from)}, not from this source "
                f"({name_of(self._source)}), so its draws could not respect the reservations"
            )
        check_int(seed, "seed")
        check_int(max_draws, "max_draws", minimum=1)

    def _check_new_name(self, name: object) -> None:
        check_identifier(name, "a collection name")
        if name in self._collections:
            raise ValueError(f"collection {name!r} is already built")

    def _collection(self, name: str) -> _Collection:
        try:
            return self._collections[name]
        except KeyError:
            raise UnknownCollectionError(name, self._collections) from None

    def __contains__(self, name: object) -> bool:
        return name in self._collections

    def __repr__(self) -> str:
        return f"Collections({self._source_name!r}, collections={list(self._collections)})"


def _settings_of(component: object) -> dict[str, Any]:
    """A component's settings as plain values: its ``config``, if that is a settings dataclass."""
    config = getattr(component, "config", None)
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return settings_to_dict(config)
    return {}


def _matches(instance: Instance, where: Mapping[str, Any]) -> bool:
    return matches(instance.descriptors, where)
