"""The components a run builds from its settings: each from a registry, or the run's collections.

A run file lists them under ``components``, by role, in the order they are
built::

    components:
      task: {kind: tasks, name: addition, settings: {max_digits: 3}}
      uniform: {kind: samplers, name: uniform, uses: {source: task}}
      collections:
        kind: collections
        settings:
          source: task
          build: {validation: {sampler: uniform, size: 100, seed: 1}}
      model: {kind: models, name: lookup}
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ..core import ConfigError, Problem, Registry, UnknownComponentError, setting, validate_settings
from ..core._declarations import takes
from ..core._text import join_path
from ..data import Collections, collators, samplers
from ..environments import registry as environments
from ..evaluate import assessments
from ..models import registry as models
from ..tasks import registry as tasks
from ..teachers import registry as teachers
from .context import derive_seed

__all__ = ["CollectionSettings", "CollectionsSettings", "ComponentEntry"]

Kind = Literal[
    "tasks",
    "teachers",
    "models",
    "environments",
    "samplers",
    "collators",
    "assessments",
    "collections",
]

KINDS: Mapping[str, Registry[Any]] = {
    "tasks": tasks,
    "teachers": teachers,
    "models": models,
    "environments": environments,
    "samplers": samplers,
    "collators": collators,
    "assessments": assessments,
}
"""The registry each kind of component is built from, by its name in ``ssapling.registries``."""


@dataclass(frozen=True)
class ComponentEntry:
    """One component a run builds: a registered component, or the run's collections."""

    kind: Kind = setting(
        description=(
            "Where it comes from: a registry, as named in ssapling.registries, or "
            "'collections', for collections drawn from a source."
        )
    )
    name: str | None = setting(
        None, min_length=1, description="Its registered name. Collections have none."
    )
    settings: dict[str, Any] = setting(default_factory=dict, description="Its settings.")
    uses: dict[str, str] = setting(
        default_factory=dict,
        description=(
            "What it is built with: for each parameter, the role of a component built before "
            "it, such as {source: task}."
        ),
    )

    def validate(self) -> None:
        if self.kind == "collections":
            if self.name is not None or self.uses:
                raise ValueError(
                    "collections have no name or uses: their settings name their source and "
                    "samplers"
                )
        elif self.name is None:
            raise ValueError(f"name the registered {self.kind[:-1]}")


@dataclass(frozen=True)
class CollectionSettings:
    """One collection to build, as for ``Collections.build``."""

    sampler: str = setting(min_length=1, description="The role of the sampler that draws it.")
    size: int = setting(ge=0, description="How many instances.")
    seed: int = setting(description="Seeds every draw.")
    held_out: bool = setting(False, description="Serve it only for evaluation.")
    where: dict[str, Any] = setting(
        default_factory=dict, description="Descriptor values every instance must have."
    )
    max_draws: int = setting(
        1000, ge=1, description="How many draws in a row may be rejected before giving up."
    )


@dataclass(frozen=True)
class CollectionsSettings:
    """Settings for a run's collections."""

    source: str = setting(
        min_length=1, description="The role of the source they are drawn from, such as the task."
    )
    build: dict[str, CollectionSettings] = setting(
        default_factory=dict, description="The collections, by name, built in order."
    )


def check_components(
    entries: Mapping[str, ComponentEntry], given: Mapping[str, Any]
) -> list[Problem]:
    """Every problem with ``entries``, which come after the components ``given``; builds nothing."""
    problems: list[Problem] = []
    roles = set(given)
    for role, entry in entries.items():
        path = f"components.{role}"
        if role in given:
            problems.append(Problem(path, f"{role!r} is given as a component already"))
        try:
            if entry.kind == "collections":
                settings = validate_settings(CollectionsSettings, entry.settings)
                uses = {"settings.source": settings.source} | {
                    f"settings.build.{name}.sampler": collection.sampler
                    for name, collection in settings.build.items()
                }
            else:
                KINDS[entry.kind].validate(str(entry.name), entry.settings)
                uses = {f"uses.{parameter}": used for parameter, used in entry.uses.items()}
        except UnknownComponentError as error:
            problems.append(Problem(f"{path}.name", str(error)))
        except ConfigError as error:
            where = f"{path}.settings"
            problems += [Problem(join_path(where, p.path), p.message) for p in error.problems]
        else:
            problems += [
                Problem(f"{path}.{where}", f"{used!r} is not a component built before it")
                for where, used in uses.items()
                if used not in roles
            ]
        roles.add(role)
    return problems


def build_components(
    entries: Mapping[str, ComponentEntry], components: dict[str, Any], *, seed: int
) -> dict[str, int]:
    """Build ``entries`` in order, adding each to ``components``; return the seeds given.

    A component whose constructor takes ``seed`` is given one derived from
    the root seed and its role (``component-<role>``).
    """
    seeds = {}
    for role, entry in entries.items():
        if entry.kind == "collections":
            settings = validate_settings(CollectionsSettings, entry.settings)
            components[role] = _collections(settings, components, entries)
            continue
        registry, name = KINDS[entry.kind], str(entry.name)
        arguments = {parameter: components[used] for parameter, used in entry.uses.items()}
        if takes(registry.get(name), "seed"):
            arguments["seed"] = seeds[role] = derive_seed(seed, f"component-{role}")
        components[role] = registry.create(name, entry.settings, **arguments)
    return seeds


def _collections(
    settings: CollectionsSettings,
    components: Mapping[str, Any],
    entries: Mapping[str, ComponentEntry],
) -> Collections:
    source = entries.get(settings.source)  # its registered name makes provenance rebuildable
    collections = Collections(
        components[settings.source], source_name=source.name if source is not None else None
    )
    for name, collection in settings.build.items():
        collections.build(
            name,
            sampler=components[collection.sampler],
            size=collection.size,
            seed=collection.seed,
            held_out=collection.held_out,
            where=collection.where or None,
            max_draws=collection.max_draws,
        )
    return collections
