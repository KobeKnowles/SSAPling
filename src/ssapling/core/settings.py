"""Settings: what a component declares it accepts, and how supplied values are checked.

A component's settings are a standard-library dataclass. Each field is one
setting: its annotation is the setting's type, its default is the setting's
default, and ``setting()`` adds a description and constraints::

    @dataclass(frozen=True)
    class StepwiseTeacherSettings:
        explain: bool = setting(False, description="Explain each correction.")
        max_interventions: int = setting(10, ge=0, description="Interventions per episode.")

        def validate(self) -> None:  # optional: rules that involve several settings
            ...

``validate_settings`` turns plain values, from Python or a file, into a checked
instance and reports every problem at once. ``settings_to_dict`` turns an
instance back into plain values. ``settings_schema`` describes the class as JSON
Schema. ``check_settings_class`` checks the declaration itself; registries call
it when a component is registered.

Supported setting types are ``bool``, ``int``, ``float``, ``str``, ``None``,
``Literal[...]``, unions such as ``int | None``, ``list[T]``, ``tuple[T, ...]``,
fixed tuples such as ``tuple[int, str]``, ``dict[str, T]``, nested settings
dataclasses, and ``Any``, which accepts any plain value (``None``, booleans,
numbers, strings, and lists and string-keyed dicts of plain values). Settings
hold plain values only, never live objects such as an environment or a model.
"""

import dataclasses
import difflib
import functools
import operator
import re
import types
import typing
from collections.abc import Mapping
from typing import Any, Literal

from ._plain import PLAIN, plain_copy
from ._text import join_path, show, summary
from .errors import ConfigError, Problem, SettingsDeclarationError

__all__ = [
    "check_settings_class",
    "setting",
    "settings_schema",
    "settings_to_dict",
    "validate_settings",
]

_METADATA_KEY = "ssapling.setting"
_MISSING: Any = dataclasses.MISSING


# Declaring settings


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _SettingInfo:
    description: str | None = None
    ge: float | None = None
    gt: float | None = None
    le: float | None = None
    lt: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


def setting(
    default: Any = _MISSING,
    *,
    default_factory: Any = _MISSING,
    description: str | None = None,
    ge: float | None = None,
    gt: float | None = None,
    le: float | None = None,
    lt: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
) -> Any:
    """Declare one setting: a dataclass field with a description and constraints.

    Args:
        default: The default value. Omit it, and ``default_factory``, to make
            the setting required.
        default_factory: Builds the default, for mutable defaults such as lists.
        description: A short description, exported in the JSON Schema.
        ge, gt, le, lt: Bounds for a number: ``>=``, ``>``, ``<=``, ``<``.
        min_length, max_length: Bounds on the length of a string, list,
            tuple, or dict.
        pattern: A regular expression a string must match in full.

    Constraints apply to the setting's own value, not to items inside it, and
    are skipped when the value is ``None``. Rules that need more, such as one
    setting depending on another, belong in the class's ``validate()`` method.
    """
    info = _SettingInfo(
        description=description,
        ge=ge,
        gt=gt,
        le=le,
        lt=lt,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
    )
    return dataclasses.field(
        default=default, default_factory=default_factory, metadata={_METADATA_KEY: info}
    )


# Types: how one annotation checks a value and describes itself as JSON Schema.
#
# ``check`` returns the value converted to the declared type (for example a
# list into a tuple) and appends a Problem for everything wrong with it.


class _Type:
    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def kinds(self) -> set[str]:
        """The JSON kinds of value this type accepts: number, string, array, object, ..."""
        raise NotImplementedError

    def _fail(self, value: Any, path: str, problems: list[Problem]) -> Any:
        problems.append(Problem(path, f"expected {self.describe()}, got {show(value)}"))
        return value


_SCALAR_NAMES: dict[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    type(None): "null",
}


@dataclasses.dataclass(frozen=True)
class _Scalar(_Type):
    cls: type

    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        is_bool = isinstance(value, bool)
        if self.cls is bool and is_bool:
            return value
        if self.cls is int and isinstance(value, int) and not is_bool:
            return value
        if self.cls is float and isinstance(value, int | float) and not is_bool:
            return float(value)
        if self.cls is str and isinstance(value, str):
            return value
        if self.cls is type(None) and value is None:
            return value
        return self._fail(value, path, problems)

    def schema(self) -> dict[str, Any]:
        return {"type": _SCALAR_NAMES[self.cls]}

    def describe(self) -> str:
        return "None" if self.cls is type(None) else self.cls.__name__

    def kinds(self) -> set[str]:
        return {"number" if self.cls in (int, float) else _SCALAR_NAMES[self.cls]}


@dataclasses.dataclass(frozen=True)
class _Literal(_Type):
    values: tuple[Any, ...]

    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        # Compare types too, so that True is not accepted for Literal[1].
        if any(type(value) is type(v) and value == v for v in self.values):
            return value
        return self._fail(value, path, problems)

    def schema(self) -> dict[str, Any]:
        return {"enum": list(self.values)}

    def describe(self) -> str:
        return "one of " + ", ".join(repr(v) for v in self.values)

    def kinds(self) -> set[str]:
        return set().union(*(_Scalar(type(v)).kinds() for v in self.values))


@dataclasses.dataclass(frozen=True)
class _Union(_Type):
    options: tuple[_Type, ...]

    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        for option in self.options:
            attempt: list[Problem] = []
            converted = option.check(value, path, attempt)
            if not attempt:
                return converted
        # For "X | None" with a value that is not None, X's own problems are
        # more useful than "expected X or None".
        non_null = [o for o in self.options if o != _Scalar(type(None))]
        if value is not None and len(non_null) == 1:
            return non_null[0].check(value, path, problems)
        return self._fail(value, path, problems)

    def schema(self) -> dict[str, Any]:
        return {"anyOf": [option.schema() for option in self.options]}

    def describe(self) -> str:
        return " or ".join(option.describe() for option in self.options)

    def kinds(self) -> set[str]:
        return set().union(*(option.kinds() for option in self.options))


@dataclasses.dataclass(frozen=True)
class _List(_Type):
    item: _Type

    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        if not isinstance(value, list | tuple):
            return self._fail(value, path, problems)
        return [self.item.check(v, f"{path}[{i}]", problems) for i, v in enumerate(value)]

    def schema(self) -> dict[str, Any]:
        return {"type": "array", "items": self.item.schema()}

    def describe(self) -> str:
        return f"list of {self.item.describe()}"

    def kinds(self) -> set[str]:
        return {"array"}


@dataclasses.dataclass(frozen=True)
class _Tuple(_Type):
    items: tuple[_Type, ...]
    variadic: bool  # tuple[T, ...]: any length, every item a T

    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        if not isinstance(value, list | tuple):
            return self._fail(value, path, problems)
        if self.variadic:
            item = self.items[0]
            return tuple(item.check(v, f"{path}[{i}]", problems) for i, v in enumerate(value))
        if len(value) != len(self.items):
            problems.append(Problem(path, f"expected {len(self.items)} items, got {len(value)}"))
            return value
        return tuple(
            t.check(v, f"{path}[{i}]", problems)
            for i, (t, v) in enumerate(zip(self.items, value, strict=True))
        )

    def schema(self) -> dict[str, Any]:
        if self.variadic:
            return {"type": "array", "items": self.items[0].schema()}
        return {
            "type": "array",
            "prefixItems": [t.schema() for t in self.items],
            "items": False,
            "minItems": len(self.items),
            "maxItems": len(self.items),
        }

    def describe(self) -> str:
        if self.variadic:
            return f"tuple of {self.items[0].describe()}"
        return f"tuple of ({', '.join(t.describe() for t in self.items)})"

    def kinds(self) -> set[str]:
        return {"array"}


@dataclasses.dataclass(frozen=True)
class _Dict(_Type):
    value_type: _Type

    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        if not isinstance(value, Mapping):
            return self._fail(value, path, problems)
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append(Problem(path, f"keys must be strings, got {show(key)}"))
                continue
            result[key] = self.value_type.check(item, join_path(path, key), problems)
        return result

    def schema(self) -> dict[str, Any]:
        return {"type": "object", "additionalProperties": self.value_type.schema()}

    def describe(self) -> str:
        return f"dict of str to {self.value_type.describe()}"

    def kinds(self) -> set[str]:
        return {"object"}


@dataclasses.dataclass(frozen=True)
class _Plain(_Type):
    """``Any``: any plain value, so settings never hold live objects."""

    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        return plain_copy(value, path, problems)

    def schema(self) -> dict[str, Any]:
        return {}

    def describe(self) -> str:
        return PLAIN

    def kinds(self) -> set[str]:
        return {"null", "boolean", "number", "string", "array", "object"}


@dataclasses.dataclass(frozen=True)
class _Nested(_Type):
    cls: type

    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        return _validate_into(self.cls, value, path, problems)

    def schema(self) -> dict[str, Any]:
        return _object_schema(self.cls)

    def describe(self) -> str:
        return f"{self.cls.__name__} settings"

    def kinds(self) -> set[str]:
        return {"object"}


def _parse(annotation: Any, path: str, problems: list[Problem]) -> _Type | None:
    """Turn a field's annotation into a _Type, or record why it is unsupported."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if annotation is Any:
        return _Plain()
    if annotation in (bool, int, float, str, type(None)):
        return _Scalar(annotation)
    if annotation is None:
        return _Scalar(type(None))
    if origin is Literal:
        if all(v is None or type(v) in (bool, int, str) for v in args):
            return _Literal(args)
        problems.append(Problem(path, "Literal values must be str, int, bool, or None"))
        return None
    if origin in (typing.Union, types.UnionType):
        options = _parse_all(args, path, problems)
        return _Union(options) if options else None
    if origin is list and len(args) == 1:
        item = _parse(args[0], path, problems)
        return _List(item) if item else None
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        item = _parse(args[0], path, problems)
        return _Tuple((item,), variadic=True) if item else None
    if origin is tuple and args:
        items = _parse_all(args, path, problems)
        return _Tuple(items, variadic=False) if items else None
    if origin is dict and len(args) == 2:
        if args[0] is not str:
            problems.append(Problem(path, "dict keys must be str, as in dict[str, int]"))
            return None
        value_type = _parse(args[1], path, problems)
        return _Dict(value_type) if value_type else None
    if annotation in (list, tuple, dict):
        problems.append(
            Problem(path, f"give the item types, as in list[int], not bare {annotation.__name__}")
        )
        return None
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        try:
            _compile(annotation)
        except SettingsDeclarationError as error:
            problems += [Problem(join_path(path, p.path), p.message) for p in error.problems]
            return None
        return _Nested(annotation)
    problems.append(
        Problem(
            path,
            f"unsupported setting type {annotation!r}; use bool, int, float, str, None, "
            "Literal, a union, list[T], tuple[...], dict[str, T], a settings dataclass, "
            "or Any",
        )
    )
    return None


def _parse_all(
    annotations: tuple[Any, ...], path: str, problems: list[Problem]
) -> tuple[_Type, ...] | None:
    """Parse several annotations; None if any of them is unsupported."""
    parsed = [_parse(a, path, problems) for a in annotations]
    supported = tuple(t for t in parsed if t is not None)
    return supported if len(supported) == len(parsed) else None


# Reading a settings class


@dataclasses.dataclass(frozen=True)
class _Field:
    name: str
    type: _Type
    info: _SettingInfo
    field: dataclasses.Field[Any]

    @property
    def required(self) -> bool:
        return self.field.default is _MISSING and self.field.default_factory is _MISSING

    def default(self) -> Any:
        factory: Any = self.field.default_factory
        return self.field.default if factory is _MISSING else factory()

    def check(self, value: Any, path: str, problems: list[Problem]) -> Any:
        found: list[Problem] = []
        converted = self.type.check(value, path, found)
        if not found:
            found += [Problem(path, m) for m in _constraint_problems(self.info, converted)]
        problems += found
        return converted


@functools.cache
def _compile(cls: type) -> tuple[_Field, ...]:
    """Read a settings class once: each field's type, constraints, and default."""
    name = getattr(cls, "__qualname__", repr(cls))
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise SettingsDeclarationError(name, [Problem("", "settings must be a dataclass")])
    try:
        hints = typing.get_type_hints(cls)
    except NameError as error:
        raise SettingsDeclarationError(
            name, [Problem("", f"annotations could not be resolved: {error}")]
        ) from None

    problems: list[Problem] = []
    fields = []
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        info = field.metadata.get(_METADATA_KEY, _SettingInfo())
        found: list[Problem] = []
        setting_type = _parse(hints[field.name], field.name, found)
        if setting_type is not None:
            found += _constraint_declaration_problems(field.name, setting_type, info)
        if setting_type is not None and not found:
            compiled = _Field(field.name, setting_type, info, field)
            if not compiled.required:
                bad_default: list[Problem] = []
                compiled.check(compiled.default(), field.name, bad_default)
                found += [Problem(p.path, f"invalid default: {p.message}") for p in bad_default]
            fields.append(compiled)
        problems += found
    if problems:
        raise SettingsDeclarationError(name, problems)
    return tuple(fields)


def _constraint_declaration_problems(
    path: str, setting_type: _Type, info: _SettingInfo
) -> list[Problem]:
    kinds = setting_type.kinds()
    problems = []
    numeric = {"ge": info.ge, "gt": info.gt, "le": info.le, "lt": info.lt}
    for name, bound in numeric.items():
        if bound is not None and "number" not in kinds:
            problems.append(
                Problem(path, f"{name}= applies to numbers, not {setting_type.describe()}")
            )
    lengths = {"min_length": info.min_length, "max_length": info.max_length}
    for name, bound in lengths.items():
        if bound is not None and not kinds & {"string", "array", "object"}:
            problems.append(
                Problem(
                    path,
                    f"{name}= applies to strings, lists, tuples, and dicts, "
                    f"not {setting_type.describe()}",
                )
            )
    if info.pattern is not None:
        if "string" not in kinds:
            problems.append(
                Problem(path, f"pattern= applies to strings, not {setting_type.describe()}")
            )
        else:
            try:
                re.compile(info.pattern)
            except re.error as error:
                problems.append(Problem(path, f"pattern {info.pattern!r} is not valid: {error}"))
    return problems


def _constraint_problems(info: _SettingInfo, value: Any) -> list[str]:
    problems = []
    if isinstance(value, int | float) and not isinstance(value, bool):
        bounds = [
            (">=", operator.ge, info.ge),
            (">", operator.gt, info.gt),
            ("<=", operator.le, info.le),
            ("<", operator.lt, info.lt),
        ]
        for symbol, test, bound in bounds:
            # "not test(...)" rather than a reversed test, so that NaN fails every bound.
            if bound is not None and not test(value, bound):
                problems.append(f"must be {symbol} {bound}, got {value!r}")
    if isinstance(value, str | list | tuple | dict):
        unit = "characters" if isinstance(value, str) else "items"
        if info.min_length is not None and len(value) < info.min_length:
            problems.append(f"must have at least {info.min_length} {unit}, got {len(value)}")
        if info.max_length is not None and len(value) > info.max_length:
            problems.append(f"must have at most {info.max_length} {unit}, got {len(value)}")
    if (
        isinstance(value, str)
        and info.pattern is not None
        and not re.fullmatch(info.pattern, value)
    ):
        problems.append(f"must match the pattern {info.pattern!r}, got {value!r}")
    return problems


def check_settings_class(cls: type) -> None:
    """Check that ``cls`` is a settings dataclass SSAPling can validate.

    Raises:
        SettingsDeclarationError: Listing every problem: unsupported types,
            constraints that do not fit their setting's type, invalid patterns,
            and defaults that fail their own type or constraints.
    """
    _compile(cls)


# Validation


def validate_settings[S](cls: type[S], values: Mapping[str, Any] | S | None = None) -> S:
    """Check supplied settings values and return an instance of ``cls``.

    ``values`` is a mapping of setting names to plain values (as written in
    Python or loaded from a file), an existing instance of ``cls`` (checked
    again), or ``None`` for all defaults. Omitted settings take their default.

    Checking covers types, unknown settings, missing required settings,
    constraints declared with ``setting()``, and the class's own ``validate()``
    method, which runs only once everything else passes. It reports every
    problem it finds, each with its path.

    Raises:
        ConfigError: The values are invalid.
        SettingsDeclarationError: ``cls`` itself is not a valid settings class.
    """
    check_settings_class(cls)
    problems: list[Problem] = []
    result = _validate_into(cls, {} if values is None else values, "", problems)
    if problems:
        raise ConfigError(f"settings for {cls.__qualname__}", problems)
    return typing.cast(S, result)


def _validate_into(cls: type, values: Any, path: str, problems: list[Problem]) -> Any:
    fields = _compile(cls)
    if isinstance(values, cls):
        values = settings_to_dict(values)
    if not isinstance(values, Mapping):
        problems.append(
            Problem(path, f"expected a mapping of {cls.__name__} settings, got {show(values)}")
        )
        return values

    count_before = len(problems)
    known = [field.name for field in fields]
    for key in values:
        if not isinstance(key, str):
            problems.append(Problem(path, f"setting names must be strings, got {show(key)}"))
        elif key not in known:
            message = "unknown setting"
            close = difflib.get_close_matches(key, known, n=1)
            if close:
                message += f"; did you mean {close[0]!r}?"
            problems.append(Problem(join_path(path, key), f"{message} (known: {', '.join(known)})"))

    checked = {}
    for field in fields:
        field_path = join_path(path, field.name)
        if field.name in values:
            checked[field.name] = field.check(values[field.name], field_path, problems)
        elif field.required:
            problems.append(Problem(field_path, "missing required setting"))
    if len(problems) > count_before:
        return values

    try:
        instance = cls(**checked)
        validate = getattr(instance, "validate", None)
        if callable(validate):
            validate()
    except ValueError as error:
        problems.append(Problem(path, str(error)))
        return values
    return instance


# Conversion and schema


def settings_to_dict(settings: object) -> dict[str, Any]:
    """Turn a settings instance into plain values: dicts, lists, and scalars.

    Nested settings become dicts and tuples become lists, so the result can be
    written as JSON or YAML. ``validate_settings`` turns it back into an equal
    instance.
    """
    if not dataclasses.is_dataclass(settings) or isinstance(settings, type):
        raise TypeError(f"expected a settings dataclass instance, got {show(settings)}")
    return {
        field.name: _plain(getattr(settings, field.name))
        for field in dataclasses.fields(settings)
        if field.init
    }


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return settings_to_dict(value)
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    return value


def settings_schema(cls: type) -> dict[str, Any]:
    """Describe a settings class as JSON Schema (draft 2020-12).

    The schema includes each setting's type, description, default, and
    constraints, and rejects unknown settings. Rules in ``validate()`` cannot
    be expressed in the schema, so a value that matches the schema can still
    fail ``validate_settings``.
    """
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **_object_schema(cls)}


def _object_schema(cls: type) -> dict[str, Any]:
    properties = {}
    for field in _compile(cls):
        schema = field.type.schema()
        info = field.info
        if info.description:
            schema["description"] = info.description
        if not field.required:
            schema["default"] = _plain(field.default())
        schema.update(_constraint_schema(info, field.type.kinds()))
        properties[field.name] = schema

    result: dict[str, Any] = {"title": cls.__name__}
    description = summary(cls)
    if description:
        result["description"] = description
    result |= {"type": "object", "properties": properties, "additionalProperties": False}
    required = [field.name for field in _compile(cls) if field.required]
    if required:
        result["required"] = required
    return result


def _constraint_schema(info: _SettingInfo, kinds: set[str]) -> dict[str, Any]:
    # JSON Schema keywords apply only to their own kind of value (minLength to
    # strings, minItems to arrays, ...), so each is added for every kind the
    # setting accepts.
    keywords: dict[str, Any] = {}
    for keyword, bound in [
        ("minimum", info.ge),
        ("exclusiveMinimum", info.gt),
        ("maximum", info.le),
        ("exclusiveMaximum", info.lt),
    ]:
        if bound is not None:
            keywords[keyword] = bound
    for kind, low, high in [
        ("string", "minLength", "maxLength"),
        ("array", "minItems", "maxItems"),
        ("object", "minProperties", "maxProperties"),
    ]:
        if kind in kinds:
            if info.min_length is not None:
                keywords[low] = info.min_length
            if info.max_length is not None:
                keywords[high] = info.max_length
    if info.pattern is not None:
        keywords["pattern"] = f"^(?:{info.pattern})$"
    return keywords
