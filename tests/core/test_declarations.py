"""The internal helpers that read what a component declares, shared by every area."""

import pytest

from ssapling.core._declarations import (
    check_identifier,
    check_int,
    has_feature,
    keyed,
    name_of,
    names,
    takes,
)

FEATURES = {"render": ("render",), "snapshot": ("snapshot", "restore")}


class Renders:
    def render(self) -> str:
        return "frame"


def test_name_of_a_component_or_its_class():
    assert name_of(Renders()) == name_of(Renders) == "Renders"


# Names


def test_names_are_a_collection_of_non_empty_strings():
    assert names({"a"}, "X.ops") == ("a",)
    assert names(("a", "b"), "X.ops") == ("a", "b")


def test_a_single_string_is_not_a_collection_of_names():
    with pytest.raises(TypeError, match=r"X.ops must be a collection of names, such as \('op',\)"):
        names("op", "X.ops", example="('op',)")


@pytest.mark.parametrize("value", [{"a", ""}, ["a", 3], 7])
def test_names_reject_anything_else(value):
    with pytest.raises(TypeError, match=r"X\.ops must"):
        names(value, "X.ops")


# Keyed mappings


def test_keyed_copies_a_mapping_with_string_keys():
    original = {"student": 1}
    copy = keyed(original, "rewards", "agent ids")
    original["student"] = 2
    assert copy == {"student": 1}
    with pytest.raises(TypeError, match="rewards must be a mapping keyed by agent ids"):
        keyed([1], "rewards", "agent ids")
    with pytest.raises(TypeError, match=r"the keys of rewards must be strings \(agent ids\)"):
        keyed({1: 1}, "rewards", "agent ids")


# Plain checks


def test_an_identifier_is_a_non_empty_string():
    check_identifier("ep-1", "a record id")
    for value in ("", None, 3):
        with pytest.raises(ValueError, match="a record id must be a non-empty string"):
            check_identifier(value, "a record id")


def test_an_integer_is_not_a_bool_and_meets_its_minimum():
    check_int(-3, "seed")
    check_int(0, "index", minimum=0)
    for value in (True, 1.0, "1"):
        with pytest.raises(TypeError, match="index must be an integer"):
            check_int(value, "index", minimum=0)
    with pytest.raises(ValueError, match="batch_size must be 1 or more, got 0"):
        check_int(0, "batch_size", minimum=1)


# Parameters


def test_takes_finds_a_keyword_parameter():
    def with_key(observations: object, *, key: object = None) -> None: ...
    def with_kwargs(observations: object, **options: object) -> None: ...
    def without(observations: object) -> None: ...
    def positional_only(key: object, /) -> None: ...

    assert takes(with_key, "key")
    assert takes(with_kwargs, "key")
    assert not takes(without, "key")
    assert not takes(positional_only, "key")
    assert not takes(None, "key")


# Features declared by methods


def test_a_feature_is_offered_by_having_its_methods():
    assert has_feature(Renders(), "render", FEATURES, word="capability", words="capabilities")
    assert not has_feature(Renders(), "snapshot", FEATURES, word="capability", words="capabilities")


def test_a_components_own_supports_can_switch_a_feature_off_but_not_on():
    class SwitchedOff(Renders):
        def supports(self, feature: str) -> bool:
            return feature != "render"

    class Boastful(Renders):
        def supports(self, feature: str) -> bool:
            return True

    assert not has_feature(
        SwitchedOff(), "render", FEATURES, word="capability", words="capabilities"
    )
    assert not has_feature(
        Boastful(), "snapshot", FEATURES, word="capability", words="capabilities"
    )


def test_a_feature_can_carry_an_extra_requirement():
    def check(requirement_met: bool) -> bool:
        return has_feature(
            Renders(),
            "render",
            FEATURES,
            word="capability",
            words="capabilities",
            requirement=lambda component: requirement_met,
        )

    assert not check(requirement_met=False)
    assert check(requirement_met=True)


def test_mistakes_are_reported_in_the_areas_words():
    with pytest.raises(ValueError, match="unknown capability 'fly'; the capabilities are 'render'"):
        has_feature(Renders(), "fly", FEATURES, word="capability", words="capabilities")

    class Listed(Renders):
        supports = ("render",)

    with pytest.raises(TypeError, match=r"Listed.supports must be a method that takes a feature"):
        has_feature(Listed(), "render", FEATURES, word="feature", words="features")
