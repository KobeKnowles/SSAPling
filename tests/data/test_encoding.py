"""The json codec: plain values in, JSON-ready values out, and back exactly."""

import json

import pytest

from ssapling import registries as r
from ssapling.data import JsonCodec

CODEC = JsonCodec()


def round_trip(value):
    return CODEC.decode(json.loads(json.dumps(CODEC.encode(value))))


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        3,
        2.5,
        "text with\nnewline",
        [1, [2, 3]],
        {"grid": [[0, 1], [1, 0]], "label": None},
        (1, "a"),
        {"key": ("pair", 3), "nested": [(1, 2), {"deep": ()}]},
        b"\x00\xffbytes",
        {"__tuple__": [1, 2]},  # a mapping that only looks like a tag
        {"__bytes__": "x", "other": 1},
    ],
)
def test_values_come_back_exactly(value):
    back = round_trip(value)
    assert back == value
    assert type(back) is type(value)


def test_tuples_and_bytes_are_tagged_and_the_rest_is_plain_json():
    assert CODEC.encode({"shape": (2, 3), "raw": b"ab", "n": 1}) == {
        "shape": {"__tuple__": [2, 3]},
        "raw": {"__bytes__": "YWI="},
        "n": 1,
    }
    assert CODEC.encode({"__tuple__": [1]}) == {"__dict__": {"__tuple__": [1]}}


def test_a_value_it_cannot_encode_is_named_by_where_it_is():
    with pytest.raises(
        TypeError, match=r"observations\.student\[1\]: the json codec cannot encode set"
    ):
        CODEC.encode({"observations": {"student": [1, {2}]}})
    with pytest.raises(TypeError, match="keys must be strings, got 1"):
        CODEC.encode({"lanes": {1: "a"}})


def test_the_json_codec_is_registered():
    assert "json" in r.codecs
    assert isinstance(r.codecs.create("json"), JsonCodec)
