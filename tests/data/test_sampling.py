"""Samplers: reproducible draws from listed and generated sources, weights, and state."""

from collections import Counter

import pytest

from ssapling import registries as r
from ssapling.core import ConfigError
from ssapling.data import (
    GeneratedSampler,
    GeneratedSettings,
    Sample,
    StratifiedSampler,
    StratifiedSettings,
    UniformSampler,
)
from ssapling.tasks import Instance
from tests.fixtures.tasks import AdditionSettings, AdditionTask, NumberBook, PuzzleBook

SEEDS = range(2000)


def addition() -> AdditionTask:
    return AdditionTask(AdditionSettings(max_digits=3))


# Samples


def test_a_sample_holds_the_instance_and_how_to_make_it_again():
    instance = Instance(id="n1", inputs=1)
    sample = Sample(instance=instance, params={"digits": 2}, position=1)
    assert (sample.instance, sample.params, sample.position) == (instance, {"digits": 2}, 1)
    with pytest.raises(TypeError, match="a sample's instance must be an Instance"):
        Sample(instance="n1")  # type: ignore[arg-type]


def test_a_samples_params_are_copied_when_it_is_made():
    params = {"digits": 2}
    sample = Sample(instance=Instance(id="n1", inputs=1), params=params)
    params["digits"] = 3
    assert sample.params == {"digits": 2}


# Uniform


def test_the_same_seed_gives_the_same_sample():
    for sampler in (
        UniformSampler(source=NumberBook()),
        StratifiedSampler(StratifiedSettings(by="size"), source=NumberBook()),
        GeneratedSampler(GeneratedSettings(vary="digits", values=[1, 2, 3]), source=addition()),
    ):
        assert [sampler.draw(seed=s) for s in range(20)] == [
            sampler.draw(seed=s) for s in range(20)
        ]
        assert len({sampler.draw(seed=s).instance.id for s in range(20)}) > 1


def test_uniform_draws_every_instance_and_records_its_position():
    book = NumberBook(count=5)
    sampler = UniformSampler(source=book)
    samples = [sampler.draw(seed=s) for s in range(200)]
    assert {sample.instance.id for sample in samples} == set(book.ids())
    assert all(book.ids()[s.position] == s.instance.id for s in samples)  # type: ignore[index]


def test_listed_and_generated_sources_need_their_own_samplers():
    with pytest.raises(TypeError, match=r"does not list its instances.*use 'generated'"):
        UniformSampler(source=addition())
    with pytest.raises(TypeError, match=r"cannot generate instances.*'uniform' or 'stratified'"):
        GeneratedSampler(source=PuzzleBook())


def test_a_source_with_no_instances_is_refused():
    with pytest.raises(ValueError, match="lists no instances"):
        UniformSampler(source=NumberBook(count=0))


# Stratified


def test_each_stratum_is_equally_likely_however_many_instances_it_has():
    sampler = StratifiedSampler(
        StratifiedSettings(by="size"), source=NumberBook()
    )  # 10 small, 20 large
    sizes = Counter(sampler.draw(seed=s).instance.descriptors["size"] for s in SEEDS)
    assert sampler.strata == ("small", "large")
    assert 0.45 < sizes["small"] / len(SEEDS) < 0.55


def test_weights_make_strata_likelier_or_leave_them_out():
    book = NumberBook()
    heavy = StratifiedSampler(
        StratifiedSettings(by="size", weights={"small": 3, "large": 1}), source=book
    )
    only_large = StratifiedSampler(StratifiedSettings(by="size", weights={"large": 1}), source=book)
    sizes = Counter(heavy.draw(seed=s).instance.descriptors["size"] for s in SEEDS)
    assert 0.7 < sizes["small"] / len(SEEDS) < 0.8
    assert {only_large.draw(seed=s).instance.descriptors["size"] for s in SEEDS} == {"large"}


def test_strata_are_descriptor_values_as_strings():
    sampler = StratifiedSampler(
        StratifiedSettings(by="parity", weights={"1": 1}), source=NumberBook()
    )
    assert sampler.strata == ("0", "1")
    assert {sampler.draw(seed=s).instance.descriptors["parity"] for s in range(100)} == {1}


def test_weights_must_name_existing_strata():
    with pytest.raises(ValueError, match=r"weights name 'medium'.*: 'small', 'large'"):
        StratifiedSampler(StratifiedSettings(by="size", weights={"medium": 1}), source=NumberBook())


def test_every_instance_needs_the_descriptor():
    with pytest.raises(ValueError, match="instance 'n0' has no descriptor 'colour'"):
        StratifiedSampler(StratifiedSettings(by="colour"), source=NumberBook())


def test_weights_are_checked_like_any_settings():
    with pytest.raises(ConfigError, match="must be a finite number of 0 or more"):
        r.samplers.create(
            "stratified", {"by": "size", "weights": {"small": -1}}, source=NumberBook()
        )
    with pytest.raises(ConfigError, match="at least one weight must be more than 0"):
        r.samplers.create(
            "stratified", {"by": "size", "weights": {"small": 0}}, source=NumberBook()
        )


def test_weights_change_over_time_and_are_saved_with_the_state():
    sampler = StratifiedSampler(StratifiedSettings(by="size"), source=NumberBook())
    sampler.set_weights({"large": 1})
    assert sampler.config.weights == {"large": 1.0}  # config always says what is used
    assert {sampler.draw(seed=s).instance.descriptors["size"] for s in range(100)} == {"large"}

    saved = sampler.export_state()
    sampler.set_weights({"small": 1})
    sampler.restore_state(saved)
    assert sampler.config.weights == {"large": 1.0}


def test_settings_made_directly_are_checked_too():
    with pytest.raises(ConfigError, match="at least one weight must be more than 0"):
        StratifiedSampler(StratifiedSettings(by="size", weights={"small": 0}), source=NumberBook())
    with pytest.raises(ConfigError, match="values must list what it takes"):
        GeneratedSampler(GeneratedSettings(vary="digits"), source=addition())


def test_bad_new_weights_leave_the_sampler_unchanged():
    sampler = StratifiedSampler(
        StratifiedSettings(by="size", weights={"small": 1}), source=NumberBook()
    )
    with pytest.raises(ValueError, match="weights name 'medium'"):
        sampler.set_weights({"medium": 1})
    with pytest.raises(ConfigError):
        sampler.set_weights({"small": -1})
    assert sampler.config.weights == {"small": 1.0}


# Generated


def test_generated_passes_fixed_params_and_records_them():
    sampler = GeneratedSampler(GeneratedSettings(params={"digits": 3}), source=addition())
    sample = sampler.draw(seed=5)
    assert sample.params == {"digits": 3}
    assert sample.instance == addition().generate(params={"digits": 3}, seed=5)
    assert sample.position is None


def test_generated_varies_one_parameter_by_weight():
    settings = GeneratedSettings(vary="digits", values=[1, 2, 3], weights={"3": 1, "1": 1})
    sampler = GeneratedSampler(settings, source=addition())
    digits = Counter(sampler.draw(seed=s).params["digits"] for s in SEEDS)
    assert set(digits) == {1, 3}
    assert 0.45 < digits[3] / len(SEEDS) < 0.55
    sampler.set_weights({"2": 1})
    assert {sampler.draw(seed=s).instance.descriptors["digits"] for s in range(50)} == {2}


def test_generated_without_settings_calls_generate_with_the_seed_alone():
    sample = GeneratedSampler(source=addition()).draw(seed=9)
    assert sample.params == {}
    assert sample.instance == addition().generate(seed=9)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"values": [1, 2]}, "values and weights apply to a varied parameter; set vary"),
        ({"vary": "digits"}, "values must list what it takes"),
        (
            {"vary": "digits", "values": [1], "params": {"digits": 2}},
            "both fixed in params and varied",
        ),
        ({"vary": "digits", "values": [1, "1"]}, "values must be distinct as strings"),
        ({"vary": "digits", "values": [1, 2], "weights": {"4": 1}}, r"weights name '4'.*'1', '2'"),
    ],
)
def test_generated_settings_are_checked(settings, message):
    with pytest.raises(ConfigError, match=message):
        r.samplers.create("generated", settings, source=addition())


# The registry


def test_the_built_in_samplers_are_registered():
    assert r.samplers.names() == ("generated", "stratified", "uniform")
    sampler = r.samplers.create("stratified", {"by": "size"}, source=NumberBook())
    assert isinstance(sampler, StratifiedSampler)
    assert sampler.config == StratifiedSettings(by="size")
    assert {entry["name"] for entry in r.samplers.catalogue()} == {
        "generated",
        "stratified",
        "uniform",
    }
