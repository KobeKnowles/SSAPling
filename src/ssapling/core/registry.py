"""Registries: one per kind of component, mapping names to implementations.

A registry also builds components: it validates settings against the
implementation's settings class and calls the implementation with them and
with the live objects the caller supplies.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast, overload

from ._text import describe, join_path, show, summary
from .errors import ConfigError, DuplicateRegistrationError, Problem, UnknownComponentError
from .settings import check_settings_class, settings_schema, validate_settings
from .spec import ComponentSpec


@dataclass(frozen=True, slots=True, eq=False)
class Registration[T]:
    """One registry entry.

    Attributes:
        name: The registered name.
        implementation: The registered class or function.
        settings: The settings dataclass the implementation accepts, or
            ``None`` if it takes no settings.
        description: A one-line description: the one given at registration,
            or else the first paragraph of the implementation's docstring.
    """

    name: str
    implementation: T
    settings: type | None
    description: str | None = None


class CatalogueEntry(TypedDict):
    """One component in a registry's catalogue, as plain JSON-ready values."""

    name: str
    description: str | None
    settings: dict[str, Any]
    """JSON Schema of the component's settings; an empty object if it takes none."""


class Registry[T]:
    """Maps names to the implementations (classes or functions) of one kind of component.

    A registry stores the implementation itself, never a constructed instance.
    Each kind of component (teachers, samplers, stages, ...) has its own
    registry, and ``kind`` names it in error messages.

    Example::

        teachers = Registry("teacher")
        teachers.register("stepwise", StepwiseTeacher)
        teachers.get("stepwise")  # -> StepwiseTeacher, the class itself
        teachers.create("stepwise", {"explain": True}, task=task)  # -> a StepwiseTeacher

    An implementation's settings class is taken from its ``Settings``
    attribute, or passed explicitly with ``settings=``; either way it is
    checked when registered (see ``ssapling.core.settings``).

    Registering the same implementation with the same settings class under
    the same name again does nothing, so a downstream ``register_components()``
    can safely run more than once. Registering anything different under a
    taken name is an error unless ``replace=True`` is passed.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, Registration[T]] = {}

    @property
    def kind(self) -> str:
        """The kind of component this registry holds, such as ``"teacher"``."""
        return self._kind

    # Registering and looking up

    def register(
        self,
        name: str,
        implementation: T,
        *,
        settings: type | None = None,
        description: str | None = None,
        replace: bool = False,
    ) -> None:
        """Register ``implementation`` under ``name``.

        Args:
            name: The name to register under.
            implementation: The class or function.
            settings: The settings dataclass the implementation accepts.
                Defaults to the implementation's ``Settings`` attribute, if any.
            description: A one-line description for the catalogue. Defaults
                to the first paragraph of the implementation's docstring.
            replace: Allow replacing a different registration under ``name``.

        Raises:
            DuplicateRegistrationError: ``name`` is taken by a different
                registration and ``replace`` is false.
            SettingsDeclarationError: The settings class cannot be checked.
            TypeError: ``name`` is not a string.
            ValueError: ``name`` is empty or has surrounding whitespace.
        """
        self._check_name(name)
        if settings is None:
            settings = getattr(implementation, "Settings", None)
        if settings is not None:
            check_settings_class(settings)
        existing = self._entries.get(name)
        if existing is not None and not replace:
            if existing.implementation is implementation and existing.settings is settings:
                return  # registered already: nothing to do
            raise DuplicateRegistrationError(
                self._kind,
                name,
                existing.implementation,
                implementation,
                existing_settings=existing.settings,
                new_settings=settings,
            )
        if description is None:
            description = summary(implementation)
        self._entries[name] = Registration(name, implementation, settings, description)

    def get(self, name: str) -> T:
        """Return the implementation registered under ``name``.

        Raises:
            UnknownComponentError: Nothing is registered under ``name``. The
                error suggests close matches and lists the registered names.
        """
        return self.entry(name).implementation

    def entry(self, name: str) -> Registration[T]:
        """Return the full registration under ``name``: implementation, settings, description.

        Raises:
            UnknownComponentError: Nothing is registered under ``name``.
        """
        return self._entry(name, location=None)

    def names(self) -> tuple[str, ...]:
        """Every registered name, sorted."""
        return tuple(sorted(self._entries))

    def catalogue(self) -> list[CatalogueEntry]:
        """Describe every registered component: name, description, and settings schema.

        The result is plain, JSON-ready data, sorted by name, suitable for
        documentation or for offering components to a procedure controller.
        """
        return [
            {
                "name": entry.name,
                "description": entry.description,
                "settings": (
                    settings_schema(entry.settings) if entry.settings else _no_settings_schema()
                ),
            }
            for entry in (self._entries[name] for name in self.names())
        ]

    # Validating and creating

    def validate(self, name: str, settings: Mapping[str, Any] | object | None = None) -> Any:
        """Check settings for the component ``name`` without creating it.

        ``settings`` is a mapping of plain values, an instance of the
        component's settings class, or ``None`` for all defaults.

        Returns:
            The validated settings instance, or ``None`` if the component
            takes no settings.

        Raises:
            UnknownComponentError: Nothing is registered under ``name``.
            ConfigError: The settings are invalid, listing every problem.
        """
        return self._validate(self.entry(name), settings, source=None, prefix=None)

    def validate_from(self, spec: ComponentSpec) -> Any:
        """``validate`` for a component entry; errors name the entry's file and path."""
        entry = self._entry(spec.name, location=_location(spec, "name"))
        return self._validate(
            entry, spec.settings, source=spec.source, prefix=join_path(spec.path, "settings")
        )

    @overload
    def create[R](
        self: "Registry[type[R]]",
        name: str,
        settings: Mapping[str, Any] | object | None = None,
        **live: Any,
    ) -> R: ...

    @overload
    def create[R](
        self: "Registry[Callable[..., R]]",
        name: str,
        settings: Mapping[str, Any] | object | None = None,
        **live: Any,
    ) -> R: ...

    def create(
        self, name: str, settings: Mapping[str, Any] | object | None = None, **live: Any
    ) -> Any:
        """Create the component registered under ``name``.

        Validates ``settings`` (see ``validate``) and calls the implementation
        with them as ``config=``, together with the live objects passed as
        keyword arguments::

            teachers.create("stepwise", {"explain": True}, task=task)
            # is StepwiseTeacher(config=StepwiseTeacherSettings(explain=True), task=task)

        A component that takes no settings is called with the live objects
        only. Which live objects a component needs is part of its area's
        contract; Core passes them on unchecked.

        Raises:
            UnknownComponentError: Nothing is registered under ``name``.
            ConfigError: The settings are invalid.
            TypeError: A live object is named ``config``, which is reserved
                for the validated settings.
            Exception: Anything the implementation raises, with a note naming
                the component.
        """
        entry = self.entry(name)
        validated = self._validate(entry, settings, source=None, prefix=None)
        return self._construct(entry, validated, live, location=None)

    def create_from(self, spec: ComponentSpec, **live: Any) -> Any:
        """``create`` for a component entry; errors name the entry's file and path.

        Example::

            config = load_config("run.yaml")
            teacher = teachers.create_from(config.component("teacher"), task=task)
        """
        validated = self.validate_from(spec)
        return self._construct(self.entry(spec.name), validated, live, location=_location(spec, ""))

    # Internals

    def _entry(self, name: str, *, location: str | None) -> Registration[T]:
        try:
            return self._entries[name]
        except KeyError:
            raise UnknownComponentError(
                self._kind, name, self._entries, location=location
            ) from None

    def _validate(
        self,
        entry: Registration[T],
        settings: object,
        *,
        source: str | None,
        prefix: str | None,
    ) -> Any:
        subject = f"settings for {self._kind} {entry.name!r}"
        if source:
            subject += f" in {source}"
        if entry.settings is None:
            if settings is None or (isinstance(settings, Mapping) and not settings):
                return None
            problem = Problem(
                prefix or "", f"{self._kind} {entry.name!r} takes no settings, got {show(settings)}"
            )
            raise ConfigError(subject, [problem])
        try:
            return validate_settings(entry.settings, settings)
        except ConfigError as error:
            problems = [
                Problem(join_path(prefix, p.path) if prefix else p.path, p.message)
                for p in error.problems
            ]
            raise ConfigError(subject, problems) from None

    def _construct(
        self,
        entry: Registration[T],
        validated: Any,
        live: dict[str, Any],
        *,
        location: str | None,
    ) -> Any:
        arguments = dict(live)
        if entry.settings is not None:
            if "config" in arguments:
                raise TypeError(
                    f"'config' cannot be passed as a live object: {self._kind} "
                    f"{entry.name!r} receives its validated settings as config=."
                )
            arguments["config"] = validated
        implementation = cast(Callable[..., Any], entry.implementation)
        try:
            return implementation(**arguments)
        except Exception as error:
            where = f" for {location}" if location else ""
            error.add_note(
                f"While creating {self._kind} {entry.name!r} ({describe(implementation)}){where}."
            )
            raise

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self._kind!r}, names={list(self.names())})"

    def _check_name(self, name: object) -> None:
        if not isinstance(name, str):
            raise TypeError(f"A {self._kind} name must be a string, not {type(name).__name__}.")
        if not name or name != name.strip():
            raise ValueError(
                f"A {self._kind} name must be non-empty, without leading or trailing "
                f"whitespace: {name!r}."
            )


def _location(spec: ComponentSpec, key: str) -> str | None:
    """Where in its source a spec's ``key`` is written, as ``"run.yaml: teacher.name"``."""
    if not spec.source and not spec.path:
        return None  # Built in Python: the caller knows where it is.
    path = join_path(spec.path, key) if key else spec.path
    if spec.source and path:
        return f"{spec.source}: {path}"
    return spec.source or path


def _no_settings_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
