"""Instances: a source of instances given as a list. And `where` matching a tuple descriptor that a
YAML or JSON file gives as a list."""

import pytest

from ssapling.data import Collections, Instances, UniformSampler
from ssapling.tasks import Instance, UnknownInstanceError
from ssapling.tasks import supports as task_supports


def pair(name: str, lengths: tuple[int, int]) -> Instance:
    return Instance(id=name, inputs={"name": name}, descriptors={"lengths": lengths})


def test_instances_lists_the_instances_it_was_given_in_order():
    a, b = pair("a", (1, 1)), pair("b", (2, 1))
    source = Instances([a, b])
    assert list(source.ids()) == ["a", "b"]
    assert source.get("b") is b
    assert task_supports(source, "enumerate")


def test_a_sampler_draws_from_instances_as_from_a_task():
    only = pair("only", (1, 1))
    assert UniformSampler(source=Instances([only])).draw(seed=5).instance is only


def test_an_id_given_twice_is_refused():
    with pytest.raises(ValueError, match="'a'"):
        Instances([pair("a", (1, 1)), pair("a", (2, 2))])


def test_an_unknown_id_is_refused_as_a_task_refuses_it():
    with pytest.raises(UnknownInstanceError, match="'z'"):
        Instances([pair("a", (1, 1))]).get("z")


def test_where_matches_a_tuple_descriptor_given_as_a_list():
    """Given instances whose `lengths` descriptor is a tuple, when a collection is built with
    where {"lengths": [2, 1]}, as a YAML or JSON file gives it, then it holds only (2, 1)s.
    Catches: [2, 1] never equal to (2, 1), so a filter on a length pair matched nothing."""
    source = Instances([pair(f"p{n}", (2, 1)) for n in range(3)] + [pair("q", (1, 2))])
    built = Collections(source).build(
        "pairs", sampler=UniformSampler(source=source), size=3, seed=1, where={"lengths": [2, 1]}
    )
    assert {drawn.instance.descriptors["lengths"] for drawn in built} == {(2, 1)}
