"""Collections: disjoint by problem, the remainder stream, held-out sets, and provenance."""

import itertools
from collections.abc import Iterable, Sequence, Sized

import pytest

from ssapling.core import Registry, settings_to_dict
from ssapling.data import (
    Collections,
    Drawn,
    HeldOutError,
    RecordHeader,
    Sample,
    SourceExhaustedError,
    StratifiedSampler,
    StratifiedSettings,
    UniformSampler,
    UnknownCollectionError,
    samplers,
)
from tests.fixtures.tasks import AdditionSettings, AdditionTask, NumberBook, PuzzleBook


def keys(items: Iterable[Drawn]) -> list[object]:
    return [drawn.instance.problem_key for drawn in items]


# Building


def test_a_collection_holds_distinct_problems_with_their_provenance():
    book = NumberBook()
    collections = Collections(book, source_name="number_book")
    validation = collections.build(
        "validation", sampler=UniformSampler(source=book), size=5, seed=1
    )

    assert len(validation) == 5
    assert len(set(keys(validation))) == 5  # at most one instance per problem
    origin = validation[0].provenance
    assert (origin.source, origin.collection, origin.sampler) == (
        "number_book",
        "validation",
        "UniformSampler",
    )
    assert origin.seed is not None
    assert book.ids()[origin.position] == validation[0].instance.id  # type: ignore[index]
    assert collections.names() == ("validation",)
    assert collections.get("validation") is validation


def test_the_same_seed_builds_the_same_collection():
    def build() -> tuple[Drawn, ...]:
        book = NumberBook()
        collections = Collections(book)
        collections.build("a", sampler=UniformSampler(source=book), size=4, seed=1)
        return collections.build("b", sampler=UniformSampler(source=book), size=4, seed=2)

    assert build() == build()


def test_collections_never_share_a_problem():
    book = NumberBook()  # 30 numbers, 15 problems: 2k and 2k + 1 are one problem
    collections = Collections(book)
    uniform = UniformSampler(source=book)
    validation = collections.build("validation", sampler=uniform, size=7, seed=1)
    test = collections.build("test", sampler=uniform, size=8, seed=2)
    assert set(keys(validation)).isdisjoint(keys(test))
    assert len(set(keys(validation)) | set(keys(test))) == 15


def test_equivalent_instances_are_never_split_between_collections():
    book = PuzzleBook()  # p1 and p1-mirror share the key "family-1"
    collections = Collections(book)
    uniform = UniformSampler(source=book)
    first = collections.build("first", sampler=uniform, size=1, seed=0)
    rest = collections.build("rest", sampler=uniform, size=2, seed=1)
    ids = {drawn.instance.id for drawn in first + rest}
    assert not {"p1", "p1-mirror"} <= ids
    assert len(set(keys(first + rest))) == 3


def test_where_restricts_a_collection_by_descriptors():
    book = NumberBook()
    collections = Collections(book)
    small = collections.build(
        "small", sampler=UniformSampler(source=book), size=5, seed=1, where={"size": "small"}
    )
    assert {drawn.instance.descriptors["size"] for drawn in small} == {"small"}


# Running out


def test_running_out_raises_an_error_naming_the_sampler_and_collections():
    book = NumberBook()
    collections = Collections(book)
    uniform = UniformSampler(source=book)
    collections.build("validation", sampler=uniform, size=10, seed=1)
    with pytest.raises(SourceExhaustedError) as caught:
        collections.build("test", sampler=uniform, size=6, seed=2, max_draws=200)
    error = caught.value
    assert (error.sampler, error.reserved, error.draws, error.building) == (
        "UniformSampler",
        ("validation",),
        200,
        "test",
    )
    assert "UniformSampler drew 200 times for collection 'test'" in str(error)
    assert "'validation'" in str(error)


def test_a_failed_build_reserves_nothing():
    book = NumberBook()
    collections = Collections(book)
    uniform = UniformSampler(source=book)
    with pytest.raises(SourceExhaustedError):
        collections.build("too_big", sampler=uniform, size=16, seed=1, max_draws=200)
    assert "too_big" not in collections
    assert len(collections.build("all", sampler=uniform, size=15, seed=1, max_draws=1000)) == 15


def test_an_impossible_restriction_runs_out_rather_than_looping():
    book = NumberBook()
    with pytest.raises(SourceExhaustedError, match="drew 50 times"):
        Collections(book).build(
            "none",
            sampler=UniformSampler(source=book),
            size=1,
            seed=1,
            where={"size": "huge"},
            max_draws=50,
        )


# The remainder


def test_the_remainder_excludes_every_reserved_problem_even_those_reserved_later():
    book = NumberBook()
    collections = Collections(book)
    uniform = UniformSampler(source=book)
    collections.build("validation", sampler=uniform, size=5, seed=1)
    training = collections.remainder(sampler=uniform, seed=3)
    early = list(itertools.islice(training, 50))
    collections.build("late", sampler=uniform, size=5, seed=2)
    late = list(itertools.islice(training, 200))

    reserved_first = set(keys(collections.get("validation")))
    reserved_late = set(keys(collections.get("late")))
    assert reserved_first.isdisjoint(keys(early + late))
    assert reserved_late.isdisjoint(keys(late))
    assert all(drawn.provenance.collection is None for drawn in early)
    assert len(keys(late)) > len(set(keys(late)))  # the remainder may repeat a problem


def test_the_remainder_is_reproducible_from_its_seed():
    book = NumberBook()
    first = Collections(book).remainder(sampler=UniformSampler(source=book), seed=3)
    second = Collections(book).remainder(sampler=UniformSampler(source=book), seed=3)
    assert list(itertools.islice(first, 20)) == list(itertools.islice(second, 20))


def test_the_remainder_runs_out_when_everything_is_reserved():
    book = NumberBook(count=4)
    collections = Collections(book)
    uniform = UniformSampler(source=book)
    collections.build("all", sampler=uniform, size=2, seed=1)
    training = collections.remainder(sampler=uniform, seed=2, max_draws=30)
    with pytest.raises(SourceExhaustedError, match="for the remainder"):
        next(training)


# Held out


def test_a_held_out_collection_is_served_only_for_evaluation():
    book = NumberBook()
    collections = Collections(book)
    test = collections.build(
        "test", sampler=UniformSampler(source=book), size=3, seed=1, held_out=True
    )
    assert collections.held_out("test")
    with pytest.raises(HeldOutError, match="served only with purpose='evaluation', not with no"):
        collections.get("test")
    with pytest.raises(HeldOutError, match="not with purpose 'training'"):
        collections.get("test", purpose="training")
    assert collections.get("test", purpose="evaluation") == test


# Datasets and streams


def test_datasets_and_streams_are_iterated_alike():
    book = NumberBook()
    collections = Collections(book)
    uniform = UniformSampler(source=book)
    dataset = collections.build("validation", sampler=uniform, size=5, seed=1)
    stream = collections.remainder(sampler=uniform, seed=2)

    def first_ids(items: Iterable[Drawn]) -> list[str]:
        return [drawn.instance.id for drawn in itertools.islice(items, 3)]

    assert len(first_ids(dataset)) == len(first_ids(stream)) == 3
    assert isinstance(dataset, Sequence)  # so it can be counted and indexed
    assert not isinstance(stream, Sized)


# Provenance enough to make an item again


def test_a_generated_item_is_made_again_from_its_provenance():
    tasks: Registry[type[AdditionTask]] = Registry("task")
    tasks.register("addition", AdditionTask)
    task = tasks.create("addition", {"max_digits": 3})
    sampler = samplers.create("generated", {"vary": "digits", "values": [1, 2, 3]}, source=task)
    collections = Collections(task, source_name="addition")
    drawn = collections.build("validation", sampler=sampler, size=10, seed=1)[7]

    origin = drawn.provenance
    assert origin.source_settings == settings_to_dict(AdditionSettings(max_digits=3))  # from config
    assert origin.sampler == "GeneratedSampler"
    assert origin.sampler_settings["vary"] == "digits"
    rebuilt = tasks.create(origin.source, origin.source_settings)
    assert rebuilt.generate(params=origin.params, seed=origin.seed) == drawn.instance  # type: ignore[arg-type]


def test_a_listed_item_is_found_again_from_its_provenance():
    book = NumberBook()
    sampler = StratifiedSampler(StratifiedSettings(by="size"), source=book)
    drawn = Collections(book).build("validation", sampler=sampler, size=4, seed=1)[2]
    assert drawn.provenance.sampler_settings == {"by": "size", "weights": {}}
    assert book.get(book.ids()[drawn.provenance.position]) == drawn.instance  # type: ignore[index]


def test_a_record_header_carries_the_provenance():
    book = NumberBook()
    drawn = Collections(book).build("v", sampler=UniformSampler(source=book), size=1, seed=1)[0]
    header = RecordHeader(
        id="ep-1",
        instance_id=drawn.instance.id,
        participants=("solver",),
        provenance=drawn.provenance,
    )
    assert header.provenance is drawn.provenance


def test_adding_a_collection_checks_its_name_its_items_and_the_problems_already_reserved():
    book = NumberBook()  # 2k and 2k + 1 are one problem
    validation = Collections(book).build(
        "validation", sampler=UniformSampler(source=book), size=3, seed=1
    )
    collections = Collections(book)
    assert collections.add("validation", validation) == validation
    assert not collections.held_out("validation")

    with pytest.raises(ValueError, match="collection 'validation' is already built"):
        collections.add("validation", ())
    with pytest.raises(TypeError, match="a collection holds Drawn items"):
        collections.add("loose", [validation[0].instance])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="already in this collection or reserved by another"):
        collections.add("again", validation[:1])
    with pytest.raises(ValueError, match="already in this collection or reserved by another"):
        Collections(book).add("twice", [validation[0], validation[0]])
    assert collections.names() == ("validation",)  # nothing was added by the failed calls


# Mistakes


def test_unknown_and_duplicate_names_are_reported():
    book = NumberBook()
    collections = Collections(book)
    collections.build("validation", sampler=UniformSampler(source=book), size=1, seed=1)
    with pytest.raises(
        UnknownCollectionError, match="No collection 'valid'; the collections are: 'validation'"
    ):
        collections.get("valid")
    with pytest.raises(ValueError, match="collection 'validation' is already built"):
        collections.build("validation", sampler=UniformSampler(source=book), size=1, seed=2)
    with pytest.raises(ValueError, match="a collection name must be a non-empty string"):
        collections.build("", sampler=UniformSampler(source=book), size=1, seed=2)


def test_a_sampler_must_draw_from_the_same_source():
    with pytest.raises(ValueError, match="draws from NumberBook, not from this source"):
        Collections(NumberBook()).build(
            "v", sampler=UniformSampler(source=NumberBook()), size=1, seed=1
        )


def test_only_a_source_can_be_collected_from():
    with pytest.raises(TypeError, match="is not a source"):
        Collections(object())


def test_a_downstream_sampler_must_return_samples():
    class Lazy:
        def draw(self, *, seed: int) -> object:
            return NumberBook().instances[seed % 30]

    with pytest.raises(TypeError, match=r"Lazy\.draw must return a Sample, got Instance"):
        Collections(NumberBook()).build("v", sampler=Lazy(), size=1, seed=1)  # type: ignore[arg-type]


def test_a_downstream_sampler_without_a_source_attribute_works():
    book = NumberBook()

    class Evens:
        def draw(self, *, seed: int) -> Sample:
            return Sample(instance=book.instances[2 * (seed % 15)])

    drawn = Collections(book).build("evens", sampler=Evens(), size=3, seed=1)
    assert all(d.instance.inputs % 2 == 0 for d in drawn)
    assert drawn[0].provenance.sampler_settings == {}


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"size": -1}, ValueError, "size must be 0 or more"),
        ({"size": 1.0}, TypeError, "size must be an integer"),
        ({"seed": True}, TypeError, "seed must be an integer"),
        ({"max_draws": 0}, ValueError, "max_draws must be 1 or more"),
    ],
)
def test_counts_and_seeds_are_checked(arguments, error, message):
    book = NumberBook()
    values = {"size": 1, "seed": 1, **arguments}
    with pytest.raises(error, match=message):
        Collections(book).build("v", sampler=UniformSampler(source=book), **values)
