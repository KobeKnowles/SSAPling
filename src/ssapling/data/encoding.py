"""Codecs: how payloads are written to storage and read back.

The store owns a file's structure; a codec owns the payloads inside it
(observations, actions, predictions, teacher contributions, an instance's
inputs). It turns them into **JSON-ready values** (``None``, booleans,
numbers, strings, lists, and dicts with string keys) and back, so every
stored file is readable JSON lines whatever the payloads are.

The built-in ``"json"`` codec handles plain Python values, and tags tuples
and bytes so they come back as tuples and bytes. A project whose payloads
hold other types, such as arrays, registers its own codec, usually by
handling those types and passing everything else to ``JsonCodec``::

    class ArrayCodec:
        def encode(self, value):
            if isinstance(value, np.ndarray):
                return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
            ...
"""

import base64
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from ..core import Registry

__all__ = ["Codec", "JsonCodec", "codecs"]


class Codec(Protocol):
    """What every codec provides: payloads to JSON-ready values, and back."""

    def encode(self, value: Any) -> Any:
        """``value`` as JSON-ready values. Raise ``TypeError`` for a value it cannot encode."""
        ...

    def decode(self, value: Any) -> Any:
        """The value ``encode`` was given, from what it returned."""
        ...


_TUPLE, _BYTES, _DICT = "__tuple__", "__bytes__", "__dict__"
_TAGS = (_TUPLE, _BYTES, _DICT)


class JsonCodec:
    """Plain Python values, with tuples and bytes tagged so they survive the round trip.

    It encodes ``None``, booleans, numbers, strings, lists, tuples, bytes,
    and mappings with string keys. A tuple becomes ``{"__tuple__": [...]}``
    and bytes become ``{"__bytes__": "<base64>"}``; a mapping that happens to
    look like a tag is wrapped as ``{"__dict__": {...}}``, so nothing is
    misread. Any other value raises ``TypeError`` naming where it is.
    """

    def encode(self, value: Any) -> Any:
        """``value`` as JSON-ready values."""
        return _encode(value, "")

    def decode(self, value: Any) -> Any:
        """The value ``encode`` was given."""
        return _decode(value)


codecs: Registry[Callable[..., Codec]] = Registry("codec")
"""Registered codecs. Also available as ``ssapling.registries.codecs``."""
codecs.register("json", JsonCodec)


def _encode(value: Any, path: str) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return [_encode(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, tuple):
        return {_TUPLE: [_encode(item, f"{path}[{i}]") for i, item in enumerate(value)]}
    if isinstance(value, bytes | bytearray):
        return {_BYTES: base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        encoded = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path or 'the value'}: keys must be strings, got {key!r}")
            encoded[key] = _encode(item, f"{path}.{key}" if path else key)
        if len(encoded) == 1 and next(iter(encoded)) in _TAGS:
            return {_DICT: encoded}
        return encoded
    raise TypeError(
        f"{path or 'the value'}: the json codec cannot encode {type(value).__name__}; "
        "register a codec that handles it"
    )


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if len(value) == 1:
            ((key, inner),) = value.items()
            if key == _TUPLE:
                return tuple(_decode(item) for item in inner)
            if key == _BYTES:
                return base64.b64decode(inner)
            if key == _DICT:
                return {k: _decode(v) for k, v in inner.items()}
        return {k: _decode(v) for k, v in value.items()}
    return value
