"""Batching: items grouped and collated, in this process or in spawn-started workers."""

import itertools
import os
from pathlib import Path

import pytest

from ssapling.core import ComponentSpec, ConfigError, UnknownComponentError
from ssapling.data import RecordStore, batches
from tests.data.test_storage import digit_pad_record
from tests.fixtures.collators import refuse, register_components, total, with_process


def test_items_are_grouped_in_order_with_a_short_last_batch():
    assert list(batches(range(7), list, batch_size=3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(batches(range(7), list, batch_size=3, drop_last=True)) == [[0, 1, 2], [3, 4, 5]]
    assert list(batches(range(6), total, batch_size=2)) == [1, 5, 9]
    assert list(batches([], list, batch_size=3)) == []


def test_a_stream_is_read_only_as_far_as_batches_are_taken():
    taken = []

    def counting():
        for n in itertools.count():
            taken.append(n)
            yield n

    first_two = list(itertools.islice(batches(counting(), list, batch_size=4), 2))
    assert first_two == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert len(taken) <= 9


def test_a_registered_collator_is_built_from_its_spec():
    register_components()
    padded = batches([[1], [2, 3]], ComponentSpec("pad", {"width": 3, "fill": -1}), batch_size=2)
    assert list(padded) == [[[1, -1, -1], [2, 3, -1]]]
    with pytest.raises(ConfigError, match="width"):
        batches([], ComponentSpec("pad", {"width": 0}), batch_size=1)
    with pytest.raises(UnknownComponentError, match="Unknown collator 'stack'"):
        batches([], ComponentSpec("stack"), batch_size=1)


def test_records_can_be_batched(tmp_path: Path):
    store = RecordStore(tmp_path)
    for n in range(3):
        store.write(digit_pad_record(f"pad-{n}"))
    records = (store.read(record_id) for record_id in store.record_ids())
    lengths = list(batches(records, lambda batch: [len(r.steps) for r in batch], batch_size=2))
    assert lengths == [[3, 3], [3]]


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"batch_size": 0}, ValueError, "batch_size must be 1 or more"),
        ({"batch_size": 2.0}, TypeError, "batch_size must be an integer"),
        ({"workers": -1}, ValueError, "workers must be 0 or more"),
        ({"collate": "list"}, TypeError, "collate must be callable or a ComponentSpec"),
        ({"setup": "register"}, TypeError, "setup must be a function"),
    ],
)
def test_arguments_are_checked_when_batches_is_called(arguments, error, message):
    values = {"collate": list, "batch_size": 2, **arguments}
    with pytest.raises(error, match=message):
        batches(range(3), **values)


def test_errors_from_collation_reach_the_caller():
    with pytest.raises(ValueError, match=r"cannot collate \[0, 1\]"):
        list(batches(range(4), refuse, batch_size=2))


# Worker processes


def test_workers_collate_in_other_processes_and_keep_the_order():
    results = list(batches(range(10), with_process, batch_size=3, workers=2))
    assert [items for _, items in results] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert os.getpid() not in {pid for pid, _ in results}


def test_workers_rebuild_a_registered_collator_after_setup():
    spec = ComponentSpec("pad", {"width": 2})
    padded = batches([[1], [2], [3]], spec, batch_size=2, workers=2, setup=register_components)
    assert list(padded) == [[[1, 0], [2, 0]], [[3, 0]]]


def test_an_error_while_setting_up_a_worker_reaches_the_caller_as_itself():
    with pytest.raises(UnknownComponentError, match="Unknown collator 'stack'"):
        list(batches(range(4), ComponentSpec("stack"), batch_size=2, workers=1))


def test_an_error_from_collation_in_a_worker_reaches_the_caller():
    with pytest.raises(ValueError, match=r"cannot collate \[0, 1\]"):
        list(batches(range(4), refuse, batch_size=2, workers=1))


def test_with_workers_a_stream_is_read_only_a_few_batches_ahead():
    taken = []

    def counting():
        for n in itertools.count():
            taken.append(n)
            yield n

    stream = batches(counting(), total, batch_size=2, workers=2)
    assert next(stream) == 1
    assert len(taken) == 8  # two batches of two in flight for each of the two workers
    stream.close()  # type: ignore[attr-defined]


def test_workers_stop_when_an_unbounded_stream_is_no_longer_read():
    stream = batches(itertools.count(), total, batch_size=2, workers=2)
    assert list(itertools.islice(stream, 3)) == [1, 5, 9]
    stream.close()  # type: ignore[attr-defined]  # stops the workers
