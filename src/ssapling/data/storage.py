"""Storage: saving records and collections, and reading them back.

``Store`` is the storage interface the rest of SSAPling uses. ``RecordStore``
is the built-in store: a directory of JSON-lines files that uses only the
standard library. A project that needs another storage system implements
``Store`` itself.

The store owns the structure: the file layout, appending to a record while
it is being written, reading back, reservations, provenance, and schema
versions. It never interprets payloads: a codec turns them into JSON-ready
values and back, and every file names the codec it was written with, so it
is read back with the right one.
"""

import dataclasses
import gzip
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, unquote

from ..core._declarations import check_identifier
from ..environments.types import ResetResult, StepResult
from ..tasks import Instance
from ..teachers.types import TeacherResponse
from .collections import Collections, Drawn
from .encoding import Codec, codecs
from .errors import StoreError, UnknownDocumentError, UnknownRecordError
from .provenance import Provenance
from .records import AgentTiming, Record, RecordBuilder, RecordHeader, RecordStep, RecordSummary

__all__ = ["STORE_FORMAT", "STORE_VERSION", "RecordStore", "Store"]

STORE_FORMAT = "ssapling-store"
"""The format name in a store's manifest."""

STORE_VERSION = 1
"""The version of the store layout this SSAPling writes and reads."""

_SUFFIX = ".jsonl"
_COMPRESSED = ".jsonl.gz"  # a finished record, with compress=True


class Store(Protocol):
    """What a store provides: records written step by step, collections, and documents.

    A record is begun with its header, appended to step by step, and
    finished with its summary; it can be read back at any point, finished or
    not. Collections are saved one at a time and loaded back together, with
    their reservations. A document is anything else saved whole by name,
    such as assessment results.
    """

    def begin(self, header: RecordHeader) -> None:
        """Start a new record."""
        ...

    def append(self, record_id: str, step: RecordStep) -> None:
        """Add the next step to a record that is not finished."""
        ...

    def finish(self, record_id: str, summary: RecordSummary) -> None:
        """Say how a record's attempt ended; it takes no more steps."""
        ...

    def read(self, record_id: str) -> Record:
        """The record as written so far."""
        ...

    def record_ids(self) -> tuple[str, ...]:
        """The ids of every record in the store."""
        ...

    def save_collection(self, collections: Collections, name: str) -> None:
        """Save one collection, held out or not, with its items' provenance."""
        ...

    def load_collections(
        self,
        source: object,
        *,
        source_name: str | None = None,
        source_settings: Mapping[str, Any] | None = None,
    ) -> Collections:
        """Every saved collection, drawn from ``source``, with its reservations."""
        ...

    def save_document(self, name: str, document: Mapping[str, Any]) -> None:
        """Save ``document`` under ``name``."""
        ...

    def load_document(self, name: str) -> dict[str, Any]:
        """The document saved under ``name``."""
        ...

    def document_names(self) -> tuple[str, ...]:
        """The names of every document in the store."""
        ...


class RecordStore:
    """The built-in store: a directory of JSON-lines files, standard library only.

    Example::

        store = RecordStore("runs/baseline")
        store.begin(header)
        store.append(header.id, step)          # one line per step, written at once
        store.finish(header.id, RecordSummary.terminated())
        store.read(header.id)                  # a Record, also before it is finished

    Layout::

        runs/baseline/
            manifest.json                   # the store's format and version
            records/<record id>.jsonl       # a header line, one line per step, a summary line
            collections/<name>.jsonl        # a header line, one line per item
            documents/<name>.jsonl          # a header line, then the document

    Each step is checked as it is appended, by the same rules as
    ``RecordBuilder``, and written straight away, so a record survives an
    interrupted run. Reading back ignores a last line that was only partly
    written. Ids and names are written into file names with
    percent-encoding, so any string is safe. Only the store that began a
    record appends to it and finishes it; a record left unfinished by an
    interrupted run stays as it was written.

    With ``compress=True``, each record's file is gzip-compressed when the
    record finishes (``records/<record id>.jsonl.gz``), which makes a long
    record about thirty times smaller; until then it is written plain, step by
    step, as always, so an interrupted run loses nothing. Every store reads
    plain and compressed records alike. Collections and documents are never
    compressed.

    Args:
        root: The store's directory. It is created if missing; an existing
            directory must be a store already, or empty.
        codec: The registered codec new files are written with.
            ``"json"`` by default.
        compress: Compress each record's file when the record finishes.

    Raises:
        StoreError: ``root`` is not empty and not a store, or was written by
            a newer SSAPling.
        UnknownComponentError: ``codec`` is not a registered codec.
    """

    def __init__(
        self, root: str | os.PathLike[str], *, codec: str = "json", compress: bool = False
    ) -> None:
        self.root = Path(root)
        self._codec_name = codec
        self._compress = compress
        self._codecs: dict[str, Codec] = {}
        self._codec(codec)
        _open(self.root)
        self._building: dict[str, RecordBuilder] = {}  # records begun here and not finished

    # Records

    def begin(self, header: RecordHeader) -> None:
        """Start a new record, writing its header.

        Raises:
            ValueError: The store already holds a record with this id.
            TypeError: ``header`` is not a ``RecordHeader``, or the codec
                cannot encode one of its payloads.
        """
        if not isinstance(header, RecordHeader):
            raise TypeError(f"header must be a RecordHeader, got {header!r}")
        path = self._record_path(header.id)
        if path.exists() or _compressed(path).exists():
            raise ValueError(f"the store at {self.root} already holds a record {header.id!r}")
        line = self._line({"kind": "header", "codec": self._codec_name, **_header_doc(header)})
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as file:
            file.write(line)
        self._building[header.id] = RecordBuilder(header)

    def append(self, record_id: str, step: RecordStep) -> None:
        """Add the next step to a record, checking it as ``RecordBuilder.append`` does.

        Raises:
            UnknownRecordError: The store holds no such record.
            ValueError: This store did not begin the record, or it is
                finished, or the step breaks one of the builder's checks.
                Nothing is written.
            TypeError: The codec cannot encode one of the step's payloads.
                Nothing is written.
        """
        builder = self._builder(record_id)
        if not isinstance(step, RecordStep):
            raise TypeError(f"step must be a RecordStep, got {step!r}")
        line = self._line({"kind": "step", **_step_doc(step)})
        builder.append(step)
        self._write(record_id, line)

    def finish(self, record_id: str, summary: RecordSummary) -> None:
        """Write how the record's attempt ended; it takes no more steps.

        Raises:
            UnknownRecordError: The store holds no such record.
            ValueError: This store did not begin the record, or it is
                already finished.
        """
        builder = self._builder(record_id)
        if not isinstance(summary, RecordSummary):
            raise TypeError(f"summary must be a RecordSummary, got {summary!r}")
        line = self._line({"kind": "summary", **_fields(summary)})
        builder.finish(summary)
        self._write(record_id, line)
        del self._building[record_id]
        if self._compress:
            path = self._record_path(record_id)
            packed = _compressed(path)
            partial = packed.with_name(packed.name + ".partial")
            partial.write_bytes(gzip.compress(path.read_bytes(), mtime=0))  # no time: same bytes
            partial.replace(packed)
            path.unlink()

    def write(self, record: Record) -> None:
        """Store a whole record at once: ``begin``, ``append`` each step, and ``finish``."""
        self.begin(record.header)
        for step in record.steps:
            self.append(record.header.id, step)
        if record.summary is not None:
            self.finish(record.header.id, record.summary)

    def read(self, record_id: str) -> Record:
        """The record as written so far: finished or not.

        Raises:
            UnknownRecordError: The store holds no such record.
            StoreError: The file is not what SSAPling writes.
        """
        path = self._record_path(record_id)
        if not path.exists():
            path = _compressed(path)
        if not path.exists():
            raise UnknownRecordError(record_id, str(self.root))
        header, *rest = self._read_docs(path)
        if header.pop("kind", None) != "header":
            raise StoreError(f"{path}: the first line is not a record header")
        del header["codec"]
        with _reading(path):
            steps: list[RecordStep] = []
            summary = None
            for doc in rest:
                kind = doc.pop("kind", None)
                if summary is not None:
                    raise StoreError(f"{path}: a line follows the summary")
                if kind == "step":
                    steps.append(_step(doc))
                elif kind == "summary":
                    summary = RecordSummary(**doc)
                else:
                    raise StoreError(f"{path}: unknown kind of line {kind!r}")
            return Record(header=_header(header), steps=tuple(steps), summary=summary)

    def record_ids(self) -> tuple[str, ...]:
        """The ids of every record in the store, sorted."""
        return tuple(
            sorted(
                {*_names_in(self.root / "records"), *_names_in(self.root / "records", _COMPRESSED)}
            )
        )

    # Collections

    def save_collection(self, collections: Collections, name: str) -> None:
        """Save the collection ``name``, held out or not, with its items' provenance.

        Raises:
            ValueError: The store already holds a collection with this name.
            UnknownCollectionError: ``collections`` has no such collection.
        """
        held_out = collections.held_out(name)
        items = collections.get(name, purpose="evaluation")  # saved whether held out or not
        directory = self.root / "collections"
        path = directory / _file_name(name)
        if path.exists():
            raise ValueError(f"the store at {self.root} already holds a collection {name!r}")
        header = {
            "kind": "collection",
            "codec": self._codec_name,
            "name": name,
            "held_out": held_out,
            "index": len(_names_in(directory)),
        }
        lines = [self._line(header)]
        lines += [self._line(_drawn_doc(drawn)) for drawn in items]
        directory.mkdir(exist_ok=True)
        with path.open("xb") as file:
            file.write(b"".join(lines))

    def load_collections(
        self,
        source: object,
        *,
        source_name: str | None = None,
        source_settings: Mapping[str, Any] | None = None,
    ) -> Collections:
        """Every saved collection, as ``Collections`` drawing from ``source``.

        Loading restores each collection's items, whether it is held out,
        and its reservations, so the remainder and any new collection still
        avoid every saved problem. The collections keep the order they were
        saved in. ``source_name`` and ``source_settings`` are as for
        ``Collections``.

        Raises:
            StoreError: A collection file is not what SSAPling writes.
        """
        collections = Collections(source, source_name=source_name, source_settings=source_settings)
        loaded = []
        for path in (self.root / "collections").glob(f"*{_SUFFIX}"):
            header, *items = self._read_docs(path)
            if header.get("kind") != "collection":
                raise StoreError(f"{path}: the first line is not a collection header")
            loaded.append((header.get("index", 0), header, items, path))
        for _, header, items, path in sorted(loaded, key=lambda entry: entry[0]):
            with _reading(path):
                drawn = [_drawn(doc) for doc in items]
                collections.add(header["name"], drawn, held_out=header["held_out"])
        return collections

    # Documents

    def save_document(self, name: str, document: Mapping[str, Any]) -> None:
        """Save ``document``, a mapping of payloads the codec can encode, under ``name``.

        A saved name is never overwritten.

        Raises:
            ValueError: The store already holds a document with this name.
            TypeError: The codec cannot encode part of the document.
        """
        path = self._document_path(name)
        if path.exists():
            raise ValueError(f"the store at {self.root} already holds a document {name!r}")
        lines = self._line({"kind": "document", "codec": self._codec_name, "name": name})
        lines += self._line({"document": dict(document)})
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as file:
            file.write(lines)

    def load_document(self, name: str) -> dict[str, Any]:
        """The document saved under ``name``.

        Raises:
            UnknownDocumentError: The store holds no such document.
            StoreError: The file is not what SSAPling writes.
        """
        path = self._document_path(name)
        if not path.exists():
            raise UnknownDocumentError(name, str(self.root))
        docs = self._read_docs(path)
        if docs[0].get("kind") != "document" or len(docs) != 2:
            raise StoreError(f"{path}: not a document file")
        document: dict[str, Any] = docs[1]["document"]
        return document

    def document_names(self) -> tuple[str, ...]:
        """The names of every document in the store, sorted."""
        return _names_in(self.root / "documents")

    # Internals

    def _codec(self, name: str) -> Codec:
        if name not in self._codecs:
            self._codecs[name] = codecs.create(name)
        return self._codecs[name]

    def _line(self, document: Mapping[str, Any]) -> bytes:
        """One line of a file: ``document`` encoded with the store's codec, as JSON."""
        encoded = self._codec(self._codec_name).encode(dict(document))
        text = json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))
        return text.encode("utf-8") + b"\n"

    def _write(self, record_id: str, line: bytes) -> None:
        with self._record_path(record_id).open("ab") as file:
            file.write(line)

    def _builder(self, record_id: str) -> RecordBuilder:
        """The builder of a record this store began and has not finished."""
        if record_id in self._building:
            return self._building[record_id]
        path = self._record_path(record_id)
        if not path.exists() and not _compressed(path).exists():
            raise UnknownRecordError(record_id, str(self.root))
        raise ValueError(
            f"record {record_id!r} takes no more steps here: it is finished, or another "
            "RecordStore began it. Only the store that begins a record writes to it."
        )

    def _read_docs(self, path: Path) -> list[dict[str, Any]]:
        """Every complete line of ``path``, decoded with the codec its first line names.

        A last line without its newline was only partly written, so it is
        ignored.
        """
        data = path.read_bytes()
        if path.name.endswith(_COMPRESSED):
            data = gzip.decompress(data)
        complete = data.split(b"\n")[:-1]
        raw = []
        for number, line in enumerate(complete, 1):
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise StoreError(f"{path}, line {number}: not valid JSON ({error.msg})") from None
        if not raw or not isinstance(raw[0], dict) or "codec" not in raw[0]:
            raise StoreError(f"{path}: the first line does not name a codec")
        codec = self._codec(raw[0]["codec"])
        return [codec.decode(doc) for doc in raw]

    def _record_path(self, record_id: str) -> Path:
        check_identifier(record_id, "a record id")
        return self.root / "records" / _file_name(record_id)

    def _document_path(self, name: str) -> Path:
        check_identifier(name, "a document name")
        return self.root / "documents" / _file_name(name)

    def __repr__(self) -> str:
        packed = ", compress=True" if self._compress else ""
        return f"RecordStore({str(self.root)!r}, codec={self._codec_name!r}{packed})"


# The manifest


def _open(root: Path) -> None:
    """Check that ``root`` is a store, or make it one."""
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            content = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise StoreError(f"{manifest}: not valid JSON ({error.msg})") from None
        if not isinstance(content, dict) or content.get("format") != STORE_FORMAT:
            raise StoreError(f"{manifest} is not an SSAPling store manifest")
        version = content.get("version")
        if not isinstance(version, int) or version > STORE_VERSION:
            raise StoreError(
                f"the store at {root} has version {version!r}, but this version of SSAPling "
                f"reads up to {STORE_VERSION}; upgrade SSAPling to read it"
            )
        return
    if root.exists() and any(root.iterdir()):
        raise StoreError(
            f"{root} is not empty and has no SSAPling manifest, so it is not a store; "
            "use a new or empty directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    content = {"format": STORE_FORMAT, "version": STORE_VERSION}
    manifest.write_text(json.dumps(content) + "\n", encoding="utf-8")


# File names


def _file_name(name: str) -> str:
    return quote(name, safe="") + _SUFFIX


def _names_in(directory: Path, suffix: str = _SUFFIX) -> tuple[str, ...]:
    if not directory.exists():
        return ()
    found = (path.name for path in directory.iterdir() if path.name.endswith(suffix))
    return tuple(sorted(unquote(name[: -len(suffix)]) for name in found))


def _compressed(path: Path) -> Path:
    """Where a record's file goes once it is compressed."""
    return path.with_name(path.name[: -len(_SUFFIX)] + _COMPRESSED)


@contextmanager
def _reading(path: Path) -> Iterator[None]:
    """Add the file being read to any error raised while its contents are rebuilt."""
    try:
        yield
    except Exception as error:
        error.add_note(f"While reading {path}.")
        raise


# Records and collections as documents, and back


def _fields(obj: Any) -> dict[str, Any]:
    return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}


def _header_doc(header: RecordHeader) -> dict[str, Any]:
    return {
        **_fields(header),
        "participants": list(header.participants),
        "reset": None if header.reset is None else _fields(header.reset),
        "provenance": None if header.provenance is None else _fields(header.provenance),
    }


def _header(doc: dict[str, Any]) -> RecordHeader:
    reset, provenance = doc["reset"], doc["provenance"]
    return RecordHeader(
        **{
            **doc,
            "participants": tuple(doc["participants"]),
            "reset": None if reset is None else ResetResult(**reset),
            "provenance": None if provenance is None else Provenance(**provenance),
        }
    )


def _step_doc(step: RecordStep) -> dict[str, Any]:
    transition = step.transition
    return {
        **_fields(step),
        "acting": list(step.acting),
        "teacher_responses": [_fields(response) for response in step.teacher_responses],
        "transition": None if transition is None else _transition_doc(transition),
        "timing": {agent: _fields(timing) for agent, timing in step.timing.items()},
    }


def _step(doc: dict[str, Any]) -> RecordStep:
    transition = doc["transition"]
    return RecordStep(
        **{
            **doc,
            "acting": tuple(doc["acting"]),
            "teacher_responses": [TeacherResponse(**r) for r in doc["teacher_responses"]],
            "transition": None if transition is None else _transition(transition),
            "timing": {agent: AgentTiming(**t) for agent, t in doc["timing"].items()},
        }
    )


def _transition_doc(result: StepResult) -> dict[str, Any]:
    responses = result.teacher_responses
    return {
        **_fields(result),
        "teacher_responses": None if responses is None else [_fields(r) for r in responses],
    }


def _transition(doc: dict[str, Any]) -> StepResult:
    responses = doc["teacher_responses"]
    return StepResult(
        **{
            **doc,
            "teacher_responses": None
            if responses is None
            else [TeacherResponse(**r) for r in responses],
        }
    )


def _drawn_doc(drawn: Drawn) -> dict[str, Any]:
    return {"instance": _fields(drawn.instance), "provenance": _fields(drawn.provenance)}


def _drawn(doc: dict[str, Any]) -> Drawn:
    return Drawn(Instance(**doc["instance"]), Provenance(**doc["provenance"]))
