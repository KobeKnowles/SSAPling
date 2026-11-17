"""Batching: grouping items for a consumer, with collation optionally in worker processes.

**Collation** turns a list of items (records, drawn instances, anything)
into one batch, for example by padding and stacking. It depends on the
model's representation, so the project supplies it: as a function, or as a
registered **collator** component built from a ``ComponentSpec``. Data does
not require torch or any other library.

Collation can run in worker processes. They are started with ``spawn``,
which works on every platform, so each worker builds its collator afresh:
a function or picklable object is sent to it, and a ``ComponentSpec`` is
rebuilt from the ``collators`` registry, after ``setup`` (such as the
project's ``register_components``) has run in that worker.
"""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ..core import ComponentSpec, Registry
from ..core._declarations import check_int

__all__ = ["batches", "collators"]

collators: Registry[Callable[..., Callable[[list[Any]], Any]]] = Registry("collator")
"""Registered collators. Also available as ``ssapling.registries.collators``.

A collator is built with its settings and then called with a list of items,
returning a batch. SSAPling registers none.
"""


def batches(
    items: Iterable[Any],
    collate: Callable[[list[Any]], Any] | ComponentSpec,
    *,
    batch_size: int,
    drop_last: bool = False,
    workers: int = 0,
    setup: Callable[[], None] | None = None,
) -> Iterator[Any]:
    """Group ``items`` into lists of ``batch_size`` and collate each into a batch.

    ``items`` may be a dataset or a stream, even an unbounded one: it is
    read only as far as the batches are taken. The batches come out in the
    order of the items.

    Example::

        for batch in batches(records, pad_and_stack, batch_size=32):
            model.learn("learn_from_batch", batch=batch)

        batches(records, ComponentSpec("pad", {"width": 30}), batch_size=32,
                workers=4, setup=register_components)

    Args:
        items: What to batch.
        collate: Turns a list of items into a batch: a function or other
            callable, or a ``ComponentSpec`` naming a registered collator.
        batch_size: How many items per batch.
        drop_last: Drop a last batch that has fewer than ``batch_size``
            items, instead of collating it.
        workers: How many worker processes collate batches. 0 collates in
            this process. With workers, ``collate`` and the items must be
            picklable, and a function must be defined at module level.
        setup: For workers only: a module-level function each worker runs
            once before collating, such as the project's
            ``register_components``, so a ``ComponentSpec`` can be built
            from the registries there.

    Raises:
        TypeError: ``collate`` is neither callable nor a ``ComponentSpec``,
            or ``setup`` is not callable.
        ValueError: ``batch_size`` is less than 1, or ``workers`` is
            negative.
        UnknownComponentError, ConfigError: A ``ComponentSpec`` names no
            registered collator, or invalid settings. Raised here when
            ``workers`` is 0; with workers, from the first batch, as is an
            error raised by ``setup``.

    An error raised by ``collate`` reaches the caller as itself, where that
    batch would have been, whether it runs here or in a worker.
    """
    check_int(batch_size, "batch_size", minimum=1)
    check_int(workers, "workers", minimum=0)
    if not (callable(collate) or isinstance(collate, ComponentSpec)):
        raise TypeError(f"collate must be callable or a ComponentSpec, got {collate!r}")
    if setup is not None and not callable(setup):
        raise TypeError(f"setup must be a function, got {setup!r}")
    chunks = _chunks(items, batch_size, drop_last)
    if workers == 0:
        return map(_collator(collate), chunks)
    return _in_workers(chunks, collate, workers, setup)


def _chunks(items: Iterable[Any], size: int, drop_last: bool) -> Iterator[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk and not drop_last:
        yield chunk


def _collator(collate: Callable[[list[Any]], Any] | ComponentSpec) -> Callable[[list[Any]], Any]:
    if isinstance(collate, ComponentSpec):
        collator: Callable[[list[Any]], Any] = collators.create_from(collate)
        return collator
    return collate


def _in_workers(
    chunks: Iterator[list[Any]],
    collate: Callable[[list[Any]], Any] | ComponentSpec,
    workers: int,
    setup: Callable[[], None] | None,
) -> Iterator[Any]:
    # Imported here: importing ssapling must not load multiprocessing.
    import multiprocessing
    from concurrent.futures import Future, ProcessPoolExecutor

    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(setup, collate),
    )
    pending: deque[Future[Any]] = deque()
    try:
        for chunk in chunks:
            pending.append(pool.submit(_collate_in_worker, chunk))
            if len(pending) >= 2 * workers:  # a few batches ahead, never the whole stream
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        pool.shutdown(cancel_futures=True)


# In each worker process

_worker_collate: Callable[[list[Any]], Any] | None = None
_worker_error: Exception | None = None


def _start_worker(
    setup: Callable[[], None] | None, collate: Callable[[list[Any]], Any] | ComponentSpec
) -> None:
    global _worker_collate, _worker_error
    try:
        if setup is not None:
            setup()
        _worker_collate = _collator(collate)
    except Exception as error:
        # Raised from the first batch instead, so the caller sees this error
        # rather than a pool that broke for no stated reason.
        _worker_error = error


def _collate_in_worker(chunk: list[Any]) -> Any:
    if _worker_error is not None:
        raise _worker_error
    assert _worker_collate is not None, "the worker was not started"
    return _worker_collate(chunk)
