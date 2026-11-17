"""Provenance: plain values, enough to trace an item or make it again."""

import pickle

import pytest

from ssapling.core import Registry, settings_to_dict
from ssapling.data import Provenance
from tests.fixtures.tasks import AdditionSettings, AdditionTask, PuzzleBook


def test_provenance_holds_where_an_item_came_from():
    origin = Provenance(
        source="addition",
        source_settings={"max_digits": 3},
        collection="validation",
        sampler="uniform",
        sampler_settings={"low": 1, "high": 3},
        params={"digits": 2},
        seed=7,
        position=12,
    )
    assert origin.source == "addition"
    assert origin.collection == "validation"
    assert origin.sampler_settings == {"low": 1, "high": 3}
    assert (origin.seed, origin.position) == (7, 12)
    assert Provenance(source="puzzles", position=0).source_settings == {}


def test_a_seed_or_a_position_is_required():
    assert Provenance(source="addition", seed=0).position is None
    assert Provenance(source="puzzles", position=0).seed is None
    with pytest.raises(ValueError, match="needs a seed or a position"):
        Provenance(source="addition")


def test_seed_and_position_are_integers():
    with pytest.raises(TypeError, match="seed must be an integer"):
        Provenance(source="addition", seed=True)
    with pytest.raises(TypeError, match="position must be an integer"):
        Provenance(source="puzzles", position=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="position must be 0 or more"):
        Provenance(source="puzzles", position=-1)


def test_names_are_non_empty_strings():
    with pytest.raises(ValueError, match="source must be a non-empty string"):
        Provenance(source="", seed=0)
    with pytest.raises(ValueError, match="sampler must be a non-empty string"):
        Provenance(source="addition", sampler="", seed=0)


def test_sampler_settings_need_a_sampler():
    with pytest.raises(ValueError, match="sampler_settings were given without a sampler"):
        Provenance(source="addition", sampler_settings={"low": 1}, seed=0)


def test_settings_are_given_as_plain_values():
    settings = settings_to_dict(AdditionSettings(max_digits=3))
    assert Provenance(source="addition", source_settings=settings, seed=0).source_settings == {
        "max_digits": 3
    }
    with pytest.raises(TypeError, match=r"turn a settings instance into one with settings_to_dict"):
        Provenance(source="addition", source_settings=AdditionSettings(), seed=0)  # type: ignore[arg-type]


def test_values_are_copied_as_plain_values_with_tuples_as_lists():
    params = {"shape": (2, 3), "nested": {"weights": [0.5, 0.5]}}
    origin = Provenance(source="grid", params=params, seed=0)
    params["nested"]["weights"].append(1.0)  # type: ignore[index]
    assert origin.params == {"shape": [2, 3], "nested": {"weights": [0.5, 0.5]}}


@pytest.mark.parametrize(
    "value",
    [object(), {1, 2}, b"bytes", complex(1, 2)],
    ids=["object", "set", "bytes", "complex"],
)
def test_values_that_are_not_plain_are_rejected(value):
    with pytest.raises(TypeError, match=r"plain values only: params\.x: expected a plain value"):
        Provenance(source="addition", params={"x": value}, seed=0)


def test_nested_keys_must_be_strings():
    with pytest.raises(TypeError, match=r"params\.x: keys must be strings, got int 1"):
        Provenance(source="addition", params={"x": {1: "a"}}, seed=0)
    with pytest.raises(TypeError, match="the keys of params must be strings"):
        Provenance(source="addition", params={1: "a"}, seed=0)  # type: ignore[dict-item]


def test_every_value_that_is_not_plain_is_reported_at_once():
    with pytest.raises(TypeError) as caught:
        Provenance(source="addition", params={"x": object(), "y": [1, {2}]}, seed=0)
    assert "params.x: expected" in str(caught.value)
    assert r"params.y[1]: expected" in str(caught.value)


def test_provenance_pickles():
    origin = Provenance(source="addition", params={"digits": 2}, seed=7)
    assert pickle.loads(pickle.dumps(origin)) == origin


# Enough to make the item again


def test_a_generated_instance_is_made_again_from_its_provenance():
    tasks: Registry[type[AdditionTask]] = Registry("task")
    tasks.register("addition", AdditionTask)
    task = tasks.create("addition", {"max_digits": 3})
    instance = task.generate(params={"digits": 2}, seed=41)
    origin = Provenance(
        source="addition",
        source_settings=settings_to_dict(task.config),
        params={"digits": 2},
        seed=41,
    )

    rebuilt = tasks.create(origin.source, origin.source_settings)
    assert rebuilt.generate(params=origin.params, seed=origin.seed) == instance  # type: ignore[arg-type]


def test_an_enumerated_instance_is_found_again_from_its_position():
    book = PuzzleBook()
    origin = Provenance(source="puzzle_book", position=1)
    assert book.get(book.ids()[origin.position]) == book.get("p1-mirror")  # type: ignore[index]
