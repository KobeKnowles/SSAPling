"""The model contract, and the checks consumers make before using a model.

A model acts in one of two ways. The simple way is ``act(observations)``,
which returns an output. The buffered way is ``perceive(observations)`` to
put input in and ``poll()`` to read output, so a model can take as long as it
needs in between. ``as_buffered`` gives an act-only model the buffered
interface, so simple models never deal with buffers.

Everything else is optional, and a model that leaves it out simply does not
have it: episode resets, persistent state, an evaluation flag, learning
operations, learning signals, and descriptions of what it handles. Every
method may be ``def`` or ``async def``. These are structural protocols: a
downstream model implements them and inherits from nothing.

Consumers check before using an optional part::

    from ssapling import models

    if models.supports(model, "consolidate"):
        stats = model.learn("consolidate", limits=limits)
    models.check_descriptions(model, observations=..., actions=...)
"""

from collections.abc import Awaitable, Mapping
from typing import Any, Protocol

from ..core._declarations import name_of, names
from .errors import DescriptionMismatchError
from .types import DescriptionKind, DescriptionMismatch, Refusal

__all__ = [
    "ActModel",
    "BufferedModel",
    "Model",
    "check_descriptions",
    "filter_signals",
    "supports",
]


class ActModel(Protocol):
    """A model that maps observations to an output in one call.

    During interaction, observations become actions; in static solving,
    inputs become predictions.
    """

    def act(self, observations: Any) -> Any:
        """Return the output for ``observations``. May be ``async def``."""
        ...


class BufferedModel(Protocol):
    """A model that receives input and gives output separately, through buffers.

    What happens in between is the model's own business. It may process in
    the background, or offer ``advance()``, which does one internal step of
    processing, so that runs in simulated time are reproducible.
    """

    def perceive(self, observations: Any) -> Awaitable[None] | None:
        """Put ``observations`` in the input buffer, and return without waiting for output."""
        ...

    def poll(self) -> Any:
        """Read the next output from the output buffer, or ``NOTHING`` if none is ready."""
        ...


Model = ActModel | BufferedModel
"""Any model: act-only or buffered."""


def supports(model: object, operation: str) -> bool:
    """Whether ``model`` supports the learning operation ``operation``.

    A model lists its learning operations in ``learning_operations``, such
    as ``{"learn_from_batch", "consolidate"}``, and performs them with
    ``learn(operation, **payload)``. Operations are an open set of names
    chosen downstream. A model without ``learning_operations`` learns
    nothing, which is allowed: a fixed rule or a frozen network is a model.

    Raises:
        TypeError: ``learning_operations`` is not a collection of names (a
            single string is rejected), or the model lists operations but has
            no ``learn`` method.
    """
    if not isinstance(operation, str):
        raise TypeError(f"an operation is a name, got {operation!r}")
    return operation in _learning_operations(model)


def filter_signals(model: object, signals: Mapping[str, Any]) -> dict[str, Any]:
    """The learning signals ``model`` declares it uses, out of ``signals``; nothing else.

    A model declares its signals by name in ``signals``, such as
    ``{"reward", "supervision"}``. Only those are passed on, so, for example,
    teacher feedback never reaches a model as a reward unless it asks for
    it. A model that declares no signals receives none.

    Raises:
        TypeError: ``signals`` is not a mapping, or the model's ``signals``
            are not a collection of names.
    """
    if not isinstance(signals, Mapping):
        raise TypeError(f"signals must be a mapping of names to values, got {signals!r}")
    declared = getattr(model, "signals", None)
    used = (
        set(names(declared, f"{name_of(model)}.signals", example="('reward',)"))
        if declared is not None
        else set()
    )
    return {name: value for name, value in signals.items() if name in used}


def check_descriptions(
    model: object,
    *,
    observations: Any = None,
    actions: Any = None,
    participant: str | None = None,
) -> None:
    """Check that ``model`` handles an environment's observations and actions.

    Pass the environment's descriptions for the participant the model
    controls, as the environment reports them, or ``None`` where it gives
    none. SSAPling does not interpret them. For each kind that both sides
    describe:

    - If the model has ``accepts_description(kind, description)``, it decides:
      true accepts; false, or a ``Refusal`` saying why, refuses.
    - Otherwise its ``observation_description`` or ``action_description``
      must equal the environment's.

    Where either side gives no description there is nothing to compare, so
    nothing is checked.

    Raises:
        DescriptionMismatchError: The model cannot handle a description. The
            error lists every mismatch.
    """
    given: dict[DescriptionKind, Any] = {"observation": observations, "action": actions}
    accepts = getattr(model, "accepts_description", None)
    if accepts is not None and not callable(accepts):
        raise TypeError(
            f"{name_of(model)}.accepts_description must be a method taking a kind and a "
            f"description, got {accepts!r}"
        )
    mismatches = []
    for kind, description in given.items():
        if description is None:
            continue
        if accepts is not None:
            answer = accepts(kind, description)
            if not answer:
                reason = answer.reason if isinstance(answer, Refusal) else None
                mismatches.append(DescriptionMismatch(kind, description, reason=reason))
            continue
        handled = getattr(model, f"{kind}_description", None)
        if handled is not None and handled != description:
            mismatches.append(DescriptionMismatch(kind, description, handled))
    if mismatches:
        raise DescriptionMismatchError(name_of(model), mismatches, participant=participant)


# Reading a model's declarations


def _learning_operations(model: object) -> frozenset[str]:
    declared = getattr(model, "learning_operations", None)
    if declared is None:
        return frozenset()
    operations = frozenset(
        names(declared, f"{name_of(model)}.learning_operations", example="('consolidate',)")
    )
    if operations and not callable(getattr(model, "learn", None)):
        raise TypeError(
            f"{name_of(model)} lists learning operations but has no learn method; "
            "a model performs them with learn(operation, **payload)"
        )
    return operations
