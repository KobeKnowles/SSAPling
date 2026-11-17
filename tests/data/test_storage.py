"""The built-in store: records written step by step and read back exactly; collections."""

import gzip
import itertools
import json
from pathlib import Path

import pytest

from ssapling.core import UnknownComponentError
from ssapling.data import (
    Collections,
    Provenance,
    Record,
    RecordBuilder,
    RecordHeader,
    RecordStep,
    RecordStore,
    RecordSummary,
    Store,
    StoreError,
    UniformSampler,
    UnknownDocumentError,
    UnknownRecordError,
    samplers,
)
from ssapling.environments import ResetResult, StepResult
from ssapling.teachers import TeacherRequest, TeacherResponse
from tests.fixtures.collators import register_components
from tests.fixtures.environments import DigitPadEnv, NimEnv
from tests.fixtures.tasks import AdditionSettings, AdditionTask, NumberBook, PuzzleBook
from tests.fixtures.teachers import DigitTutor, DigitTutorSettings


def digit_pad_record(record_id: str = "pad-1") -> Record:
    """A record with a teacher in the environment, provenance, seeds, and tagged payloads."""
    env = DigitPadEnv(teacher=DigitTutor(DigitTutorSettings(intervene=True), task=AdditionTask()))
    builder = RecordBuilder(
        RecordHeader(
            id=record_id,
            instance_id="add-45-57",
            participants=("student",),
            reset=env.reset(options={"a": 45, "b": 57}),
            provenance=Provenance(source="addition", params={"digits": 2}, seed=41),
            seeds={"reset": 7},
            clock="simulated",
            teacher_placement="environment",
            metadata={"shape": (2, 3), "raw": b"\x00\x01"},
        )
    )
    for index, digit in enumerate("301"):
        result = env.step({"student": digit})
        builder.append(
            RecordStep(
                index=index,
                acting=("student",),
                proposals={"student": digit},
                executed_actions=result.executed_actions,
                teacher_responses=result.teacher_responses or (),
                transition=result,
            )
        )
    return builder.finish(RecordSummary.terminated(metadata={"steps": 3}))


def nim_record() -> Record:
    env = NimEnv(stones=5)
    builder = RecordBuilder(
        RecordHeader(id="nim-1", instance_id="nim", participants=NimEnv.agents, reset=env.reset())
    )
    for index, take in enumerate((2, 2, 1)):
        (player,) = env.next_agents
        builder.append(
            RecordStep(
                index=index,
                acting=(player,),
                proposals={player: take},
                transition=env.step({player: take}),
            )
        )
    return builder.finish(RecordSummary.terminated())


def static_record() -> Record:
    builder = RecordBuilder(RecordHeader(id="static-1", instance_id="p1", participants=("solver",)))
    builder.append(
        RecordStep(
            index=0,
            acting=("solver",),
            presented={"solver": [[1, 0]]},
            predictions={"solver": [[0, 1]]},
        )
    )
    return builder.record  # unfinished


# The store's directory


def test_a_new_store_writes_its_manifest(tmp_path: Path):
    RecordStore(tmp_path / "store")
    manifest = json.loads((tmp_path / "store" / "manifest.json").read_text())
    assert manifest == {"format": "ssapling-store", "version": 1}
    RecordStore(tmp_path / "store")  # and opens again


def test_a_directory_that_is_not_a_store_is_refused(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("mine")
    with pytest.raises(StoreError, match="not empty and has no SSAPling manifest"):
        RecordStore(tmp_path)


def test_a_store_from_a_newer_ssapling_is_refused(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "ssapling-store", "version": 2}))
    with pytest.raises(StoreError, match=r"version 2.*upgrade SSAPling"):
        RecordStore(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "other"}))
    with pytest.raises(StoreError, match="not an SSAPling store manifest"):
        RecordStore(tmp_path)


def test_an_unknown_codec_is_refused_when_the_store_is_opened(tmp_path: Path):
    with pytest.raises(UnknownComponentError, match="Unknown codec 'msgpack'"):
        RecordStore(tmp_path, codec="msgpack")


def test_the_built_in_store_is_a_store(tmp_path: Path):
    store: Store = RecordStore(tmp_path)  # checked by the type checker
    assert store.record_ids() == ()


# Records


@pytest.mark.parametrize("make", [digit_pad_record, nim_record, static_record])
def test_a_record_comes_back_exactly(tmp_path: Path, make):
    record = make()
    store = RecordStore(tmp_path)
    store.write(record)
    assert store.read(record.header.id) == record


def test_tuples_and_bytes_in_payloads_survive(tmp_path: Path):
    store = RecordStore(tmp_path)
    store.write(digit_pad_record())
    metadata = store.read("pad-1").header.metadata
    assert metadata == {"shape": (2, 3), "raw": b"\x00\x01"}


def test_a_record_is_written_step_by_step_and_readable_before_it_is_finished(tmp_path: Path):
    record = digit_pad_record()
    store = RecordStore(tmp_path)
    store.begin(record.header)
    store.append("pad-1", record.steps[0])
    so_far = store.read("pad-1")
    assert so_far.steps == record.steps[:1]
    assert not so_far.finished
    for step in record.steps[1:]:
        store.append("pad-1", step)
    store.finish("pad-1", record.summary)  # type: ignore[arg-type]
    assert store.read("pad-1") == record


def test_every_line_is_json_and_the_file_names_its_codec(tmp_path: Path):
    store = RecordStore(tmp_path)
    store.write(digit_pad_record())
    lines = (tmp_path / "records" / "pad-1.jsonl").read_text().splitlines()
    documents = [json.loads(line) for line in lines]
    assert [doc["kind"] for doc in documents] == ["header", "step", "step", "step", "summary"]
    assert documents[0]["codec"] == "json"


def test_a_partly_written_last_line_is_ignored(tmp_path: Path):
    record = digit_pad_record()
    store = RecordStore(tmp_path)
    store.begin(record.header)
    store.append("pad-1", record.steps[0])
    with (tmp_path / "records" / "pad-1.jsonl").open("ab") as file:
        file.write(b'{"kind":"step","ind')  # the run stopped mid-write
    assert store.read("pad-1").steps == record.steps[:1]


def test_a_damaged_line_is_reported_with_its_file_and_line(tmp_path: Path):
    store = RecordStore(tmp_path)
    store.write(digit_pad_record())
    path = tmp_path / "records" / "pad-1.jsonl"
    lines = path.read_bytes().split(b"\n")
    lines[2] = b"{not json"
    path.write_bytes(b"\n".join(lines))
    with pytest.raises(StoreError, match=r"pad-1\.jsonl, line 3: not valid JSON"):
        store.read("pad-1")


def lines_changed(change):
    """A damage: ``change`` applied to a file's lines, each read as JSON."""

    def damage(text: str) -> str:
        lines = [json.loads(line) for line in text.splitlines()]
        return "".join(json.dumps(line) + "\n" for line in change(lines))

    return damage


DAMAGES = {
    "record without a header": (
        "records/pad-1.jsonl",
        lines_changed(lambda ls: [{**ls[0], "kind": "step"}, *ls[1:]]),
        "the first line is not a record header",
    ),
    "record going on after its summary": (
        "records/pad-1.jsonl",
        lines_changed(lambda ls: [*ls, ls[1]]),
        "a line follows the summary",
    ),
    "record with an unknown line": (
        "records/pad-1.jsonl",
        lines_changed(lambda ls: [ls[0], {**ls[1], "kind": "note"}, *ls[2:]]),
        "unknown kind of line 'note'",
    ),
    "record naming no codec": (
        "records/pad-1.jsonl",
        lines_changed(lambda ls: [{k: v for k, v in ls[0].items() if k != "codec"}, *ls[1:]]),
        "the first line does not name a codec",
    ),
    "collection without a header": (
        "collections/validation.jsonl",
        lines_changed(lambda ls: [{**ls[0], "kind": "record"}, *ls[1:]]),
        "the first line is not a collection header",
    ),
    "document with an extra line": (
        "documents/run.jsonl",
        lines_changed(lambda ls: [*ls, ls[1]]),
        "not a document file",
    ),
    "manifest that is not JSON": (
        "manifest.json",
        lambda text: "{not json",
        "not valid JSON",
    ),
}

READS = {  # what reads each damaged file, by the folder it is in
    "records": lambda root, book: RecordStore(root).read("pad-1"),
    "collections": lambda root, book: RecordStore(root).load_collections(book),
    "documents": lambda root, book: RecordStore(root).load_document("run"),
    "manifest.json": lambda root, book: RecordStore(root),
}


@pytest.mark.parametrize("damage", DAMAGES.values(), ids=DAMAGES.keys())
def test_a_damaged_store_file_is_reported_never_misread(tmp_path: Path, damage):
    file, change, message = damage
    store = RecordStore(tmp_path)
    store.write(digit_pad_record())
    book = NumberBook()
    collections = Collections(book)
    collections.build("validation", sampler=UniformSampler(source=book), size=2, seed=1)
    store.save_collection(collections, "validation")
    store.save_document("run", {"seed": 7})
    path = tmp_path / file
    path.write_text(change(path.read_text()))

    with pytest.raises(StoreError, match=message):
        READS[file.split("/")[0]](tmp_path, book)


def test_steps_are_checked_and_a_bad_one_is_not_written(tmp_path: Path):
    record = digit_pad_record()
    store = RecordStore(tmp_path)
    store.begin(record.header)
    path = tmp_path / "records" / "pad-1.jsonl"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="step index 1 is out of order"):
        store.append("pad-1", record.steps[1])
    with pytest.raises(TypeError, match="the json codec cannot encode set"):
        store.append("pad-1", RecordStep(index=0, acting=("student",), proposals={"student": {1}}))
    assert path.read_bytes() == before


def test_a_header_that_cannot_be_encoded_creates_no_record(tmp_path: Path):
    header = RecordHeader(id="ep-1", instance_id="i", participants=("a",), metadata={"bad": {1, 2}})
    store = RecordStore(tmp_path)
    with pytest.raises(TypeError, match=r"metadata\.bad"):
        store.begin(header)
    assert store.record_ids() == ()


def test_record_ids_round_trip_through_file_names(tmp_path: Path):
    store = RecordStore(tmp_path)
    for record_id in ("run/1:a", "épisode 2", "..", "CON-3"):
        store.write(Record(header=RecordHeader(id=record_id, instance_id="i", participants=("a",))))
    assert store.record_ids() == ("..", "CON-3", "run/1:a", "épisode 2")
    assert store.read("run/1:a").header.id == "run/1:a"


def test_a_record_id_is_used_once(tmp_path: Path):
    store = RecordStore(tmp_path)
    store.write(digit_pad_record())
    with pytest.raises(ValueError, match="already holds a record 'pad-1'"):
        store.begin(digit_pad_record().header)


def test_unknown_records_are_reported(tmp_path: Path):
    store = RecordStore(tmp_path)
    with pytest.raises(UnknownRecordError, match="holds no record 'ep-9'"):
        store.read("ep-9")
    with pytest.raises(UnknownRecordError):
        store.append("ep-9", RecordStep(index=0, acting=()))
    with pytest.raises(ValueError, match="a record id must be a non-empty string"):
        store.read("")


def test_only_the_store_that_began_a_record_writes_to_it(tmp_path: Path):
    record = digit_pad_record()
    first = RecordStore(tmp_path)
    first.begin(record.header)
    first.append("pad-1", record.steps[0])

    later = RecordStore(tmp_path)  # after a restart, say
    with pytest.raises(ValueError, match="'pad-1' takes no more steps here"):
        later.append("pad-1", record.steps[1])
    with pytest.raises(ValueError, match="'pad-1' takes no more steps here"):
        later.finish("pad-1", RecordSummary.truncated())
    assert later.read("pad-1").steps == record.steps[:1]  # unfinished, as it was written

    with pytest.raises(ValueError, match="step index 2 is out of order: the next step is 1"):
        first.append("pad-1", record.steps[2])
    for step in record.steps[1:]:
        first.append("pad-1", step)
    first.finish("pad-1", RecordSummary.terminated(metadata={"steps": 3}))
    assert later.read("pad-1") == record
    with pytest.raises(ValueError, match="'pad-1' takes no more steps here"):
        first.finish("pad-1", RecordSummary.truncated())


def test_documents_are_saved_whole_and_never_overwritten(tmp_path: Path):
    store = RecordStore(tmp_path)
    document = {"aggregates": {"accuracy": 0.75}, "pair": (1, 2), "raw": b"\x00"}
    store.save_document("results/assess-1", document)
    assert RecordStore(tmp_path).load_document("results/assess-1") == document
    with pytest.raises(ValueError, match="already holds a document 'results/assess-1'"):
        store.save_document("results/assess-1", {})
    with pytest.raises(UnknownDocumentError, match="holds no document 'nothing'"):
        store.load_document("nothing")


def test_the_store_lists_its_documents(tmp_path: Path):
    store = RecordStore(tmp_path)
    assert store.document_names() == ()
    for name in ("run", "checkpoints/12", "resume-1/checkpoints/15"):
        store.save_document(name, {})
    assert store.document_names() == ("checkpoints/12", "resume-1/checkpoints/15", "run")


def test_each_file_is_read_with_the_codec_it_names(tmp_path: Path):
    register_components()  # registers the "sets" codec
    header = RecordHeader(
        id="ep-1", instance_id="i", participants=("a",), metadata={"tags": {"x", "y"}}
    )
    RecordStore(tmp_path, codec="sets").write(Record(header=header))
    assert RecordStore(tmp_path).read("ep-1").header.metadata == {
        "tags": {"x", "y"}
    }  # json by default


def test_a_record_from_a_newer_ssapling_is_refused_with_its_file(tmp_path: Path):
    store = RecordStore(tmp_path)
    store.write(nim_record())
    path = tmp_path / "records" / "nim-1.jsonl"
    path.write_text(path.read_text().replace('"schema_version":1', '"schema_version":2'))
    with pytest.raises(ValueError, match="schema version 2") as caught:
        store.read("nim-1")
    assert any("nim-1.jsonl" in note for note in caught.value.__notes__)


def test_responses_of_every_status_come_back(tmp_path: Path):
    request = TeacherRequest(id="r", operation="review_action")
    answer = TeacherResponse.answered(
        request, {"feedback": {"a": ("note", 1)}}, detail={"cost": 0.1}
    )
    responses = [
        answer,
        answer.as_stale(),
        TeacherResponse.failed(request, "boom"),
        TeacherResponse.abstained(request),
    ]
    header = RecordHeader(
        id="ep-1",
        instance_id="i",
        participants=("a",),
        reset=ResetResult(observations={"a": 0}),
        teacher_placement="coordinator",
    )
    step = RecordStep(
        index=0,
        acting=("a",),
        teacher_responses=responses,
        transition=StepResult(observations={"a": 1}),
    )
    store = RecordStore(tmp_path)
    store.write(Record(header=header, steps=(step,)))
    assert store.read("ep-1").steps[0].teacher_responses == tuple(responses)


# Collections


def test_collections_are_saved_and_loaded_with_their_reservations(tmp_path: Path):
    book = NumberBook()
    collections = Collections(book, source_name="number_book")
    uniform = UniformSampler(source=book)
    validation = collections.build("validation", sampler=uniform, size=5, seed=1)
    test = collections.build("test", sampler=uniform, size=5, seed=2, held_out=True)
    store = RecordStore(tmp_path)
    store.save_collection(collections, "validation")
    store.save_collection(collections, "test")

    loaded = store.load_collections(book, source_name="number_book")
    assert loaded.names() == ("validation", "test")
    assert loaded.get("validation") == validation
    assert loaded.held_out("test")
    assert loaded.get("test", purpose="evaluation") == test

    reserved = {d.instance.problem_key for d in validation + test}
    training = list(
        itertools.islice(loaded.remainder(sampler=UniformSampler(source=book), seed=3), 100)
    )
    assert reserved.isdisjoint(d.instance.problem_key for d in training)
    more = loaded.build("more", sampler=UniformSampler(source=book), size=5, seed=4)
    assert reserved.isdisjoint(d.instance.problem_key for d in more)


def test_generated_instances_and_tuple_keys_survive_saving(tmp_path: Path):
    task = AdditionTask(AdditionSettings(max_digits=3))
    sampler = samplers.create("generated", {"vary": "digits", "values": [1, 2, 3]}, source=task)
    collections = Collections(task, source_name="addition")
    validation = collections.build("validation", sampler=sampler, size=20, seed=1)
    store = RecordStore(tmp_path)
    store.save_collection(collections, "validation")
    loaded = store.load_collections(task).get("validation")
    assert loaded == validation
    assert isinstance(loaded[0].instance.equivalence_key, tuple)


def test_a_collection_is_saved_once(tmp_path: Path):
    book = PuzzleBook()
    collections = Collections(book)
    collections.build("v", sampler=UniformSampler(source=book), size=1, seed=1)
    store = RecordStore(tmp_path)
    store.save_collection(collections, "v")
    with pytest.raises(ValueError, match="already holds a collection 'v'"):
        store.save_collection(collections, "v")


def test_loading_from_an_empty_store_gives_no_collections(tmp_path: Path):
    assert RecordStore(tmp_path).load_collections(NumberBook()).names() == ()


# Compressed records


def test_a_finished_record_is_compressed_and_reads_back_unchanged(tmp_path: Path):
    """Given a store with compress=True, when a record is written whole, then its file is gzip,
    named for the record with .jsonl.gz, and it reads back unchanged, through any store."""
    record = digit_pad_record()
    RecordStore(tmp_path, compress=True).write(record)
    (path,) = (tmp_path / "records").iterdir()
    assert path.name == "pad-1.jsonl.gz"
    assert gzip.decompress(path.read_bytes()).startswith(b'{"kind":"header"')
    assert RecordStore(tmp_path, compress=True).read("pad-1") == record
    assert RecordStore(tmp_path).read("pad-1") == record


def test_an_unfinished_record_stays_plain_and_readable(tmp_path: Path):
    """Given a store with compress=True, when a record is begun and given two steps but not
    finished, then its file is plain, as without compression, and reads back with those steps:
    an interrupted run loses nothing."""
    record = digit_pad_record()
    store = RecordStore(tmp_path, compress=True)
    store.begin(record.header)
    for step in record.steps[:2]:
        store.append(record.header.id, step)
    assert [p.name for p in (tmp_path / "records").iterdir()] == ["pad-1.jsonl"]
    read = store.read("pad-1")
    assert read.steps == record.steps[:2]
    assert read.summary is None


def test_plain_and_compressed_records_read_alike(tmp_path: Path):
    """Given one store holding a record written without compression and one written with it, then
    both read back, and record_ids lists each once."""
    plain, packed = digit_pad_record("plain"), digit_pad_record("packed")
    RecordStore(tmp_path).write(plain)
    RecordStore(tmp_path, compress=True).write(packed)
    store = RecordStore(tmp_path)
    assert store.record_ids() == ("packed", "plain")
    assert (store.read("plain"), store.read("packed")) == (plain, packed)


def test_a_compressed_record_id_is_used_once(tmp_path: Path):
    """Given a compressed record, when another record with its id is begun, then it is refused,
    by a store with compression or without."""
    RecordStore(tmp_path, compress=True).write(digit_pad_record())
    for compress in (True, False):
        with pytest.raises(ValueError, match="already holds a record 'pad-1'"):
            RecordStore(tmp_path, compress=compress).begin(digit_pad_record().header)


def test_a_compressed_record_has_the_same_bytes_each_time(tmp_path: Path):
    """Given the same record written compressed into two stores, then the two files are byte for
    byte the same: no time is stored in the gzip header."""
    for name in ("one", "two"):
        RecordStore(tmp_path / name, compress=True).write(digit_pad_record())
    one, two = ((tmp_path / n / "records" / "pad-1.jsonl.gz").read_bytes() for n in ("one", "two"))
    assert one == two
