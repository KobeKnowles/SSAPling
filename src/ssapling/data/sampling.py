"""Sources and samplers: where instances come from, and how they are picked.

A **source** is anything that offers instances the way a task does: by
generating them (the task feature ``"generate"``: ``generate(params,
seed)``) or by listing them (the task feature ``"enumerate"``: ``ids()``
and ``get(id)``). A task is a source; so is a stored collection, a teacher
proposing problems, or a service supplying identifiers, once it offers one
of those methods. ``tasks.supports(source, feature)`` says which.

A **sampler** picks instances from a source according to its settings.
``draw(seed=...)`` returns a ``Sample``: the instance, and what is needed to
make it again. The same seed and the same settings always give the same
sample. Samplers are components, registered in ``ssapling.data.samplers``
and built through Core::

    sampler = samplers.create("stratified", {"by": "difficulty"}, source=task)
    sample = sampler.draw(seed=7)
    sample.instance
"""

import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core import Registry, setting, settings_to_dict, validate_settings
from ..core._declarations import keyed, name_of
from ..tasks import EnumeratedTask, GeneratedTask, Instance, UnknownInstanceError
from ..tasks import supports as task_supports

__all__ = [
    "GeneratedSampler",
    "GeneratedSettings",
    "Instances",
    "Sample",
    "Sampler",
    "StratifiedSampler",
    "StratifiedSettings",
    "UniformSampler",
    "samplers",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    """One instance a sampler picked, and what is needed to make it again.

    Attributes:
        instance: The instance.
        params: For a generated source: the parameters ``generate`` was
            called with. Together with the draw's seed, they make the
            instance again even after a stateful sampler has changed.
        position: For an enumerated source: the instance's index in the
            source's ``ids()``.
    """

    instance: Instance
    params: Mapping[str, Any] = field(default_factory=dict)
    position: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instance, Instance):
            raise TypeError(f"a sample's instance must be an Instance, got {self.instance!r}")
        object.__setattr__(self, "params", keyed(self.params, "params", "names"))


class Sampler(Protocol):
    """What every sampler provides: one sample per seed.

    ``draw`` must be reproducible: the same seed, with the same settings and
    state, gives the same sample. A sampler may also have ``source``, the
    source it draws from, which ``Collections`` checks, and
    ``export_state()``/``restore_state(state)`` when it has state that
    changes over time, such as weights a training schedule adjusts.
    """

    def draw(self, *, seed: int) -> Sample:
        """Pick one instance, using ``seed`` for every random choice."""
        ...


# A listed source


class Instances:
    """A source listing the instances it is given, in their order: an
    enumerated source (``ids()`` and ``get(id)``) for a few instances made in
    code, such as one problem a lesson keeps coming back to.

    Raises:
        TypeError: Something given isn't an ``Instance``.
        ValueError: Two instances have the same id.
    """

    def __init__(self, instances: Iterable[Instance]) -> None:
        self._instances: dict[str, Instance] = {}
        for instance in instances:
            if not isinstance(instance, Instance):
                raise TypeError(f"Instances holds Instance objects, got {instance!r}")
            if instance.id in self._instances:
                raise ValueError(f"two instances have the id {instance.id!r}")
            self._instances[instance.id] = instance

    def ids(self) -> list[str]:
        """Every instance's id, in the order given."""
        return list(self._instances)

    def get(self, instance_id: str) -> Instance:
        """The instance with ``instance_id``.

        Raises:
            UnknownInstanceError: None has that id.
        """
        try:
            return self._instances[instance_id]
        except KeyError:
            raise UnknownInstanceError(instance_id, task="Instances") from None

    def __repr__(self) -> str:
        return f"Instances({self.ids()!r})"


# Built-in samplers


class UniformSampler:
    """Picks any instance of an enumerated source, each equally likely."""

    def __init__(self, *, source: object) -> None:
        self.source = source
        self._source = _enumerated(source, self)
        self._ids = tuple(self._source.ids())
        if not self._ids:
            raise ValueError(f"{name_of(source)} lists no instances to draw from")

    def draw(self, *, seed: int) -> Sample:
        """An instance chosen uniformly, by ``seed``."""
        position = random.Random(seed).randrange(len(self._ids))
        return Sample(instance=self._source.get(self._ids[position]), position=position)


@dataclass(frozen=True)
class StratifiedSettings:
    """Settings for the stratified sampler."""

    by: str = setting(description="The descriptor whose values are the strata.", min_length=1)
    weights: dict[str, float] = setting(
        default_factory=dict,
        description=(
            "Each stratum's weight, keyed by the descriptor value as a string, such as "
            '{"easy": 1, "hard": 3}. Empty: every stratum equally likely. A stratum left '
            "out is not drawn."
        ),
    )

    def validate(self) -> None:
        _check_weights(self.weights)


class StratifiedSampler:
    """Picks a stratum (a descriptor value) by weight, then an instance of it uniformly.

    With equal weights, each stratum is drawn equally often however many
    instances it has. Weights make some strata likelier, or leave some out;
    ``set_weights`` changes them over time, as a curriculum might.
    """

    Settings = StratifiedSettings

    def __init__(self, config: StratifiedSettings, *, source: object) -> None:
        self.source = source
        self.config = validate_settings(StratifiedSettings, config)  # also when built directly
        enumerated = _enumerated(source, self)
        self._instances = tuple(enumerated.get(instance_id) for instance_id in enumerated.ids())
        self._strata: dict[str, list[int]] = {}
        for position, instance in enumerate(self._instances):
            if config.by not in instance.descriptors:
                raise ValueError(
                    f"instance {instance.id!r} has no descriptor {config.by!r} to stratify by"
                )
            self._strata.setdefault(str(instance.descriptors[config.by]), []).append(position)
        if not self._strata:
            raise ValueError(f"{name_of(source)} lists no instances to draw from")
        _check_keys(config.weights, self._strata, f"values of the descriptor {config.by!r}")

    @property
    def strata(self) -> tuple[str, ...]:
        """The strata, as strings, in the order they first appear in the source."""
        return tuple(self._strata)

    def draw(self, *, seed: int) -> Sample:
        """A stratum chosen by weight, then one of its instances uniformly, both by ``seed``."""
        rng = random.Random(seed)
        stratum = _choose(rng, self.strata, self.config.weights)
        position = rng.choice(self._strata[stratum])
        return Sample(instance=self._instances[position], position=position)

    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Replace the strata's weights; ``config`` changes to match."""
        config = _with_weights(self.config, weights)
        _check_keys(config.weights, self._strata, f"values of the descriptor {config.by!r}")
        self.config = config

    def export_state(self) -> dict[str, Any]:
        """The current weights, for a checkpoint."""
        return {"weights": dict(self.config.weights)}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Return to weights saved by ``export_state``."""
        self.set_weights(state["weights"])


@dataclass(frozen=True)
class GeneratedSettings:
    """Settings for the generated sampler."""

    params: dict[str, Any] = setting(
        default_factory=dict, description="Parameters passed to generate on every draw."
    )
    vary: str | None = setting(None, description="A parameter whose value changes per draw.")
    values: list[Any] = setting(default_factory=list, description="The values `vary` takes.")
    weights: dict[str, float] = setting(
        default_factory=dict,
        description=(
            "Each value's weight, keyed by the value as a string, such as "
            '{"1": 1, "3": 2}. Empty: every value equally likely. A value left out is not drawn.'
        ),
    )

    def validate(self) -> None:
        if self.vary is None:
            if self.values or self.weights:
                raise ValueError("values and weights apply to a varied parameter; set vary")
            return
        if not self.values:
            raise ValueError(f"vary is {self.vary!r}, so values must list what it takes")
        if self.vary in self.params:
            raise ValueError(f"{self.vary!r} is both fixed in params and varied")
        keys = [str(value) for value in self.values]
        if len(set(keys)) != len(keys):
            raise ValueError("values must be distinct as strings, since weights are keyed by them")
        _check_weights(self.weights)
        _check_keys(self.weights, keys, "values")


class GeneratedSampler:
    """Generates instances: fixed parameters, plus one parameter that can vary by weight.

    For example, ``{"vary": "digits", "values": [1, 2, 3], "weights": {"3":
    2}}`` asks an addition task for 3-digit problems only, and
    ``set_weights`` can shift it towards longer problems over time.
    """

    Settings = GeneratedSettings

    def __init__(self, config: GeneratedSettings | None = None, *, source: object) -> None:
        if not task_supports(source, "generate"):
            raise TypeError(
                f"{name_of(source)} cannot generate instances: the generated sampler needs a "
                "source with generate(params, seed); for a listed source use 'uniform' or "
                "'stratified'"
            )
        self.source = source
        self.config = validate_settings(GeneratedSettings, config)  # also when built directly

    def draw(self, *, seed: int) -> Sample:
        """Generate an instance with ``seed``, choosing the varied parameter by ``seed`` too."""
        params = dict(self.config.params)
        if self.config.vary is not None:
            by_key = {str(value): value for value in self.config.values}
            key = _choose(random.Random(seed), tuple(by_key), self.config.weights)
            params[self.config.vary] = by_key[key]
        source: GeneratedTask = self.source  # type: ignore[assignment]
        return Sample(instance=source.generate(params=params, seed=seed), params=params)

    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Replace the values' weights; ``config`` changes to match."""
        self.config = _with_weights(self.config, weights)

    def export_state(self) -> dict[str, Any]:
        """The current weights, for a checkpoint."""
        return {"weights": dict(self.config.weights)}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Return to weights saved by ``export_state``."""
        self.set_weights(state["weights"])


samplers: Registry[Callable[..., Sampler]] = Registry("sampler")
"""Registered samplers. Also available as ``ssapling.registries.samplers``.

The built-ins are registered here: ``"uniform"``, ``"stratified"`` and
``"generated"``. A downstream sampler is registered alongside them.
"""
samplers.register("uniform", UniformSampler)
samplers.register("stratified", StratifiedSampler)
samplers.register("generated", GeneratedSampler)


# Helpers


def _enumerated(source: object, sampler: object) -> EnumeratedTask:
    if not task_supports(source, "enumerate"):
        raise TypeError(
            f"{name_of(source)} does not list its instances: {name_of(sampler)} needs a source "
            "with ids() and get(id); for a generated source use 'generated'"
        )
    return source  # type: ignore[return-value]


def _check_weights(weights: Mapping[str, float]) -> None:
    for key, weight in weights.items():
        if not weight >= 0 or weight == float("inf"):
            raise ValueError(f"weights[{key!r}] must be a finite number of 0 or more, got {weight}")
    if weights and not sum(weights.values()) > 0:
        raise ValueError("at least one weight must be more than 0")


def _check_keys(weights: Mapping[str, float], known: Any, what: str) -> None:
    unknown = [key for key in weights if key not in known]
    if unknown:
        raise ValueError(
            f"weights name {', '.join(repr(k) for k in unknown)}, which are not among the "
            f"{what}: {', '.join(repr(k) for k in known)}"
        )


def _choose(rng: random.Random, keys: tuple[str, ...], weights: Mapping[str, float]) -> str:
    """One of ``keys``, by ``weights``; equally likely when there are none."""
    if not weights:
        return rng.choice(keys)
    return rng.choices(keys, weights=[weights.get(key, 0.0) for key in keys])[0]


def _with_weights[S](config: S, weights: Mapping[str, float]) -> S:
    """``config`` with new weights, checked like any settings."""
    values = {**settings_to_dict(config), "weights": dict(weights)}
    return validate_settings(type(config), values)
