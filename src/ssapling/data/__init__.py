"""SSAPling data.

Responsibility: Sources, samplers, disjoint collections, records, datasets and
streams, storage, and batching.

Not its responsibility: What a problem means or how it is scored.

May import: core, tasks, and the result types of environments and teachers.
Enforced by tests/test_layering.py.
"""

from .batching import batches, collators
from .collections import Collections, Drawn
from .encoding import Codec, JsonCodec, codecs
from .errors import (
    HeldOutError,
    SourceExhaustedError,
    StoreError,
    UnknownCollectionError,
    UnknownDocumentError,
    UnknownRecordError,
)
from .provenance import Provenance
from .records import (
    RECORD_SCHEMA_VERSION,
    AgentTiming,
    Clock,
    LateOutput,
    Record,
    RecordBuilder,
    RecordHeader,
    RecordStatus,
    RecordStep,
    RecordSummary,
    TeacherPlacement,
)
from .sampling import (
    GeneratedSampler,
    GeneratedSettings,
    Instances,
    Sample,
    Sampler,
    StratifiedSampler,
    StratifiedSettings,
    UniformSampler,
    samplers,
)
from .storage import STORE_FORMAT, STORE_VERSION, RecordStore, Store

__all__ = [
    "RECORD_SCHEMA_VERSION",
    "STORE_FORMAT",
    "STORE_VERSION",
    "AgentTiming",
    "Clock",
    "Codec",
    "Collections",
    "Drawn",
    "GeneratedSampler",
    "GeneratedSettings",
    "HeldOutError",
    "Instances",
    "JsonCodec",
    "LateOutput",
    "Provenance",
    "Record",
    "RecordBuilder",
    "RecordHeader",
    "RecordStatus",
    "RecordStep",
    "RecordStore",
    "RecordSummary",
    "Sample",
    "Sampler",
    "SourceExhaustedError",
    "Store",
    "StoreError",
    "StratifiedSampler",
    "StratifiedSettings",
    "TeacherPlacement",
    "UniformSampler",
    "UnknownCollectionError",
    "UnknownDocumentError",
    "UnknownRecordError",
    "batches",
    "codecs",
    "collators",
    "samplers",
]
