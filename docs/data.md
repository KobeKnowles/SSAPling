# Data

`ssapling.data` manages instances and records: where they come from, which
are selected, how they are stored, and how they are delivered. It does not
know what a problem means, how it is scored, or how a model encodes it.

| Part | What it is |
| --- | --- |
| [Records](#records) | The log of one attempt at one instance: a header, steps, and a summary. |
| [Building a record](#building-a-record) | `RecordBuilder` appends steps and checks each one. A record is valid before it has a summary. |
| [Teacher placement](#teacher-placement) | Executed actions and teacher responses read the same wherever the teacher sits. |
| [Static solving](#static-solving) | Interaction with no environment uses the same record structure. |
| [The clock](#the-clock) | Which clock an attempt was timed with. |
| [Sources](#sources) | Anything that generates instances or lists them, as a task does. |
| [Samplers](#samplers) | Pick instances from a source, reproducibly from a seed: `uniform`, `stratified`, `generated`, or your own. |
| [Collections](#collections) | Named sets drawn from one source that never share a problem; the remainder streams everything else; held-out sets. |
| [Datasets and streams](#datasets-and-streams) | A collection is a tuple, the remainder an iterator: both iterate the same way. |
| [Provenance](#provenance) | Where an item came from, in plain values: enough to trace it or make it again. |
| [Storage](#storage) | `RecordStore`: records written step by step and collections saved, as JSON-lines files. |
| [Codecs](#codecs) | How payloads are written: plain values by default, your own codec for anything else. |
| [Batching](#batching) | `batches(items, collate, ...)`, with collation optionally in worker processes. |

## Records

A **record** is the log of one attempt at one instance. SSAPling fixes its
structure, so Data, Train, and Evaluate can handle any project's records.
What the payloads contain is downstream. Observations, actions, predictions,
teacher contributions and diagnostics pass through uninterpreted.

| Part | Type | Contents |
| --- | --- | --- |
| Header | `RecordHeader` | What is fixed when the attempt starts: ids, participants, the reset, provenance, seeds, the clock, the teacher's placement. |
| Steps | `RecordStep` | One per environment step (or per presentation, in static solving): who acted, what they proposed, the teacher's responses, what was executed, and the environment's transition result. |
| Summary | `RecordSummary` | How the attempt ended: terminated, truncated, or failed. |

A step holds the environment's `StepResult` and the teacher's
`TeacherResponse` objects themselves, rather than copying their fields, so
each shape is defined once. The header does the same with the
environment's `ResetResult`. Per-agent values are keyed by agent id, even
with one agent, as everywhere in SSAPling.

### The header

```python
RecordHeader(
    id="ep-1",                        # the record's id, chosen by the coordinating code
    instance_id="add-2d-7",           # the instance attempted
    participants=("student",),        # every agent id taking part
    reset=reset_result,               # the environment's ResetResult; None for static solving
    provenance=Provenance(...),       # optional: where the instance came from
    seeds={"reset": 7, "model": 3},   # the seeds the coordinating code used, by name
    clock="simulated",                # "simulated", "wall", or None (no timing)
    teacher_placement="coordinator",  # "coordinator", "environment", "participant", or None
    metadata={"run": "baseline"},     # downstream fields
)
```

- **The record refers to its instance by id.** It never holds the instance
  itself, so the instance's reference never travels with its records.
- **The reset is kept whole.** `header.initial_observations` reads the
  participants' first observations from it. The reset's `info` and action
  restrictions are kept too.
- **`schema_version`** defaults to `RECORD_SCHEMA_VERSION` (currently 1).
  A record with a newer version than this SSAPling knows is refused by name,
  so it is never misread.

### Steps

```python
RecordStep(
    index=0,                                   # position in the record, from 0
    acting=("student",),                       # the agents that acted
    proposals={"student": "9"},                # what they proposed
    teacher_responses=[response],              # the teacher's responses at this step
    executed_actions={"student": "2"},         # what was actually executed
    transition=step_result,                    # the environment's StepResult
    metadata={...},                            # downstream fields
)
```

`proposals`, `timing`, `diagnostics` and, in static solving, `presented` and
`predictions` may only name agents in `acting`. Executed actions may be the environment's own
report, so they are not checked against `acting`, and the environment's
results are not checked against the participants: which agents exist is the
environment's rule.

`transition` is `None` when the step did not complete, for example when a
remote environment could not be reached. The step is still recorded, and
the summary says the attempt failed.

### The summary

```python
RecordSummary.terminated()                     # a terminal state: won, lost, solved
RecordSummary.truncated("step limit")          # cut off before one
RecordSummary.failed("connection lost")        # could not continue; the reason is kept in detail
RecordSummary.truncated(metadata={"steps": 50})  # metadata: a shortcut for filtering stored records
```

The summary is not checked against the last transition. The coordinating
code may cut an attempt off with its own limit while the environment says
it continues.

### What a record does not hold

- **Judgements.** Scores and other measurements are Evaluate results that
  refer to the record.
- **The instance's reference.** The header names the instance by id.
- **Teacher requests.** Each response carries its request's id.

A record keeps what each participant observed (in `header.reset` and each
step's `transition`, or `presented` in static solving) apart from
everything else: teacher contributions, diagnostics, metadata. So a model's
inputs can be built from observations alone. Using any other field, such as
a teacher's supervision as a learning target, is an explicit downstream
choice.

## Building a record

`RecordBuilder` builds a record step by step. Here a student enters 45 + 57
= 102 one digit at a time, units first:

```python
from ssapling.data import RecordBuilder, RecordHeader, RecordStep, RecordSummary

env = DigitPadEnv()
reset = env.reset(options={"a": 45, "b": 57})
builder = RecordBuilder(
    RecordHeader(id="pad-1", instance_id="add-45-57", participants=("student",), reset=reset)
)

for index, digit in enumerate("201"):
    result = env.step({"student": digit})
    builder.append(RecordStep(
        index=index, acting=("student",),
        proposals={"student": digit}, executed_actions={"student": digit},
        transition=result,
    ))
    builder.record             # the record so far: valid, summary None

record = builder.finish(RecordSummary.terminated())
```

**A record is valid before it has a summary.** `builder.record` can be read
at any time, to stream the attempt, to learn from it online, or to keep it
when an episode is interrupted. Each read returns an immutable snapshot, so
a snapshot taken earlier does not change when more steps are appended.
`builder.next_index` is the index the next step must have.

Each step is checked as it is appended, so a mistake is reported where it
is made:

| Check | Why |
| --- | --- |
| Steps arrive in order: `index` equals `next_index`. | A step's index identifies it when it is streamed alone. |
| Every agent in `acting` is a participant. | Catches a wrong agent id in the coordinating code. |
| No step follows one whose transition ended the episode. | A record holds one attempt. The next episode gets a new record. |
| A static record's steps have no proposals, executed actions or transitions, and an interactive record's steps have no presentations or predictions. | Static and interactive steps share one type but keep to their own fields. |
| Nothing is appended after `finish`, and `finish` is called once. | A finished record is final. |

A `Record` made directly, for example when a store reads one back, is
checked by the same rules as a whole.

## Teacher placement

A teacher can sit in the coordinating code, inside the environment, or
control a participant. **Executed actions and teacher responses read the
same in the record whatever the placement**, so code that reads records
never needs to know where the teacher sat.

| Placement | Who consults the teacher | Where the step's `executed_actions` and `teacher_responses` come from |
| --- | --- | --- |
| `"coordinator"` | The coordinating code, between the model and the environment. | The coordinating code: what it submitted, and its teacher's responses. |
| `"environment"` | The environment, inside its step. | The environment's `StepResult`, passed on unchanged by the coordinating code. |
| `"participant"` | Nobody: the teacher acts through the ordinary agent interface. | The coordinating code. |

A step holds what it is given. **The environment's report is never
overridden:** when the environment owns the teacher, the coordinating code
copies the executed actions and teacher responses from its `StepResult`
into the step as they are:

```python
result = env.step(actions)          # the environment consults its own teacher
RecordStep(index=i, acting=acting, proposals=actions,
           executed_actions=result.executed_actions,
           teacher_responses=result.teacher_responses or (),
           transition=result)
```

The [episode loop](interaction.md#the-episode-loop) does exactly this, so downstream code only writes
it when building records by hand. The `StepResult` is kept whole in
`transition` too, so the environment's own report can always be read there.
The header's `teacher_placement` says where the teacher sat.

Responses of every status are recorded as they are. A `stale` answer keeps
its contributions, so the record shows what the teacher said, but only an
intervention changes an executed action, and only an answered one is ever
applied.

## Static solving

Static solving is interaction with no environment: the coordinating code
presents an instance's inputs to a model, all at once or in parts, and
reads its predictions. A static record has no reset (`header.static` is
`True`), and each step holds what was presented and what was predicted:

```python
builder = RecordBuilder(RecordHeader(id="static-1", instance_id="p1", participants=("solver",)))
builder.append(RecordStep(index=0, acting=("solver",),
                          presented={"solver": puzzle.inputs},
                          predictions={"solver": first_attempt}))
builder.append(RecordStep(index=1, acting=("solver",),
                          presented={"solver": puzzle.inputs},
                          predictions={"solver": second_attempt}))
```

Several attempts at one instance are several steps. Presenting an instance
in parts is one step per part, with a prediction on the step where the
model gives one. Static and interactive records are the same types, so
storage, batching, Train and Evaluate handle both alike.

## The clock

The header's `clock` names the clock the attempt was timed with:
`"simulated"` (advanced only by the coordinating code, so runs are
reproducible and time is counted in model steps) or `"wall"` (real time, in
seconds). `None` means no timing was applied, as in a record built by hand or
a teacher's demonstration.

Each step's `timing` holds every acting agent's `AgentTiming`: how long it
took (`elapsed`, in the clock's units), whether a default action was used
for want of an output (`defaulted`), and whether a late output was
`"discarded"` or `"used"` at that step (`late`). The episode loop fills it
in (see [Interaction: timing](interaction.md#timing)); a record built by
hand may leave it empty.

Each step's `diagnostics` holds what each acting agent's model reported
about its decision, such as the tokens or compute it used, passed through
uninterpreted (see [Models: diagnostics](models.md#diagnostics)). The
environment's own diagnostics stay in `transition.info`.

## Sources

A **source** provides instances. Data needs no new interface for it: a
source is anything that offers instances the way a task does, by
**generating** them (`generate(params, seed)`, the task feature
`"generate"`) or by **listing** them (`ids()` and `get(id)`, the task feature
`"enumerate"`). A task is a source. So is a stored collection, a teacher
proposing problems, or a service that supplies game ids, once it offers one
of those methods; it needs no `score`. `tasks.supports(source, "generate")`
says which a source offers.

Sources that arrive already divided, such as separately published training
and evaluation sets, are separate sources.

For a few instances made in code, `Instances([...])` is a listed source: its
`ids()` are theirs, in order, and `get(id)` gives one back (an unknown id
raises `UnknownInstanceError`, as a task does):

```python
lesson = Instances([task.problem(47, 5)])           # one problem, again and again
UniformSampler(source=lesson).draw(seed=3).instance
```

## Samplers

A **sampler** picks instances from a source. `draw(seed=...)` returns a
`Sample`: the instance, plus what is needed to make it again (`params` for a
generated source, `position` in `ids()` for a listed one). **The same seed
gives the same sample**, so every draw is reproducible; Data never chooses
seeds itself.

Samplers are components, registered in `ssapling.data.samplers` (also
`ssapling.registries.samplers`) and built through Core with their source:

```python
from ssapling import registries as r

uniform = r.samplers.create("uniform", source=puzzles)
by_level = r.samplers.create("stratified", {"by": "difficulty"}, source=puzzles)
curriculum = r.samplers.create(
    "generated", {"vary": "digits", "values": [1, 2, 3], "weights": {"1": 3, "2": 1}},
    source=addition,
)
sample = curriculum.draw(seed=7)   # Sample(instance=..., params={"digits": 1})
```

| Built-in | Source | Picks |
| --- | --- | --- |
| `uniform` | Listed | Any instance, each equally likely. No settings. |
| `stratified` | Listed | A stratum (a value of the descriptor `by`) by `weights`, then one of its instances uniformly. With no weights, every stratum is equally likely however many instances it has. |
| `generated` | Generated | Calls `generate(params, seed)` with the fixed `params`, plus one parameter `vary` taking one of `values` by `weights`. With no `vary`, just the fixed params. |

**Weights** are keyed by the value as a string (`{"easy": 1, "hard": 3}`,
`{"1": 3, "2": 1}`), because settings files have string keys. No weights
means equal weights; a value left out of non-empty weights is never drawn;
a weight for a value that does not exist is an error.

**Samplers can change over time.** A training schedule or a teacher can
call `set_weights(...)`, for example to move towards harder problems as the
student improves. The sampler's `config` always says what it uses now, and
`export_state()` / `restore_state(state)` save and restore the weights with a
checkpoint. Draws stay reproducible: a draw depends on the seed and the
weights at that moment, and the item's provenance keeps the parameters it
was made with.

**Your own sampler** needs only `draw(*, seed) -> Sample`. Give it a
`source` attribute, and `Collections` checks it draws from the right source;
give it `config` (as a registry does), and its settings appear in provenance;
give it `export_state`/`restore_state` if it has state that changes.

## Collections

A **collection** is a named set of instances drawn from a source. One
source can supply many collections, and **they never share a problem**:
each collection reserves the `problem_key` of every instance it holds (the
equivalence key, or the id when there is none), and every later draw
rejects reserved problems.

```python
from ssapling.data import Collections

collections = Collections(addition, source_name="addition")
collections.build("validation", sampler=uniform_digits, size=1000, seed=1)
collections.build("extrapolation", sampler=long_problems, size=1000, seed=2)
collections.build("test", sampler=uniform_digits, size=1000, seed=3, held_out=True)
training = collections.remainder(sampler=curriculum, seed=4)    # drawn on the fly

for drawn in collections.get("validation"):
    drawn.instance, drawn.provenance
```

- **Distinct problems.** A collection holds at most one instance per
  problem, so equivalent instances (the same puzzle under another seed, say)
  are never split between collections, nor repeated within one.
- **Reproducible.** Each draw gets its own seed, derived from the
  collection's `seed`. The same sampler, seed and earlier collections build
  the same collection.
- **Restricted by descriptors.** `where={"digits": 3}` keeps only instances
  with those descriptor values. A list equals a tuple with the same items, so
  `where={"lengths": [2, 1]}`, as a YAML or JSON file gives it, matches the
  descriptor `(2, 1)`.
- **Never loops forever.** If `max_draws` draws in a row (1000 by default)
  are rejected, `SourceExhaustedError` names the sampler and the reserved
  collections. A failed build reserves nothing.
- **The remainder** is a stream of instances outside every collection,
  drawn on the fly, so a training set need not be stored. It checks the
  collections as they are at each draw, so a collection built later is
  excluded from then on. It reserves nothing, so it can repeat a problem.
- **Held out.** A collection built with `held_out=True` is served by `get`
  only to a consumer that declares `purpose="evaluation"`; anything else
  raises `HeldOutError`. How strictly to use this is the project's choice.

Every item served is a **`Drawn`**: the instance and its provenance, which
`Collections` fills in: the source (`source_name`, or the source's class
name), the source's settings (`source_settings`, or the source's `config`),
the collection (`None` for the remainder), the sampler's class name and
settings, the parameters, and the draw's seed and position. Pass the
registered `source_name` to make every item rebuildable through the
registry. [Storage](#saving-and-loading-collections) saves collections and loads
them back with their reservations.

## Datasets and streams

A **dataset** is finite and re-readable; a **stream** may be unbounded or
live. Data uses Python's own types for them: a built collection is a
`tuple` of `Drawn` (a dataset), and the remainder is an iterator (a
stream). Consumers iterate both the same way, and check before counting or
indexing:

```python
from collections.abc import Sequence, Sized

for drawn in items: ...                 # a dataset or a stream
if isinstance(items, Sized): len(items)
if isinstance(items, Sequence): items[0]
```

## Provenance

Data does not choose seeds. The coordinating code supplies them, and Data
records them. `Provenance` says where an instance or record came from, in
plain values:

```python
Provenance(
    source="addition",                        # the source's name
    source_settings={"max_digits": 3},        # its settings: settings_to_dict(task.config)
    collection="validation",                  # optional
    sampler="StratifiedSampler",              # optional
    sampler_settings={"by": "digits"},        # optional, only with a sampler
    params={"digits": 2},                     # what a generated item was made with
    seed=41,                                  # the seed it was drawn or generated with
    position=12,                              # its index in the source, or place in a stream
)
```

At least one of `seed` and `position` is required. Every value must be plain:
`None`, booleans, numbers, strings, lists, and mappings with string keys. So
provenance can be stored whatever codec the payloads use. As with
`settings_to_dict`, tuples become lists, and the values are copied when the
provenance is made.

`Collections` fills it in for every item it serves (see
[Collections](#collections)). Provenance is enough to **make the item again**. For a generated task,
`params` and `seed` are passed back to `generate`:

```python
from ssapling import registries as r

task = r.tasks.create(origin.source, origin.source_settings)
assert task.generate(params=origin.params, seed=origin.seed) == instance
```

`params` are kept because a stateful sampler, whose weights a training
schedule changes over time, would not choose the same parameters again from
its settings alone. For an enumerated task, `position` indexes `task.ids()`,
and the record's `instance_id` names the instance directly.

## Storage

`Store` is the storage interface the rest of SSAPling uses: the episode
loop writes records through it, Train's `learn_from_data` stage can read them
back, and assessments and training runs save results, settings and
checkpoints in it as documents. `RecordStore` is the built-in store, using only the standard library.
A project that needs a database or a cloud bucket implements `Store`
itself.

```python
from ssapling.data import RecordStore

store = RecordStore("runs/baseline")       # created if missing
store.begin(header)                        # the header line
store.append(header.id, step)              # one line per step, written straight away
store.finish(header.id, RecordSummary.terminated())
store.read(header.id)                      # a Record: finished or not
store.write(record)                        # a whole record at once
store.record_ids()                         # every record's id, sorted
```

```text
runs/baseline/
    manifest.json                  {"format": "ssapling-store", "version": 1}
    records/<record id>.jsonl      a header line, one line per step, a summary line
    records/<record id>.jsonl.gz   the same, compressed, with compress=True, once finished
    collections/<name>.jsonl       a header line, one line per item
    documents/<name>.jsonl         a header line, then the document
```

- **Every line is written as it happens**, so a record survives an
  interrupted run, and reading back ignores a last line that was only partly
  written. A record can be read while it is still being written.
- **`RecordStore(root, compress=True)`** gzip-compresses each record's file
  when the record finishes (a long record becomes about thirty times
  smaller); until then it is plain, as above. Every store reads plain and
  compressed records alike, and a record id is used once across both.
  Collections and documents are never compressed.
- **Each step is checked as it is appended**, by the same rules as
  `RecordBuilder`. A step that breaks them, or whose payloads the codec
  cannot encode, is refused and nothing is written.
- **Records come back exactly**: `store.read(record.header.id) == record`,
  tuples and bytes in payloads included.
- **Only the store that began a record writes to it.** Another
  `RecordStore` on the same directory, after a restart say, can read the
  record but not append to it or finish it. An episode interrupted mid-way
  is run again as a new record; the unfinished one stays as it was written.
- **Any id or name is safe**: it is written into the file name with
  percent-encoding (`run/1` is stored as `run%2F1.jsonl`).
- **The store owns the structure, not the payloads.** The manifest says what
  the directory is and its layout version; a newer version, or a non-empty
  directory that is not a store, is refused. Each record's header holds its
  record schema version.

### Saving and loading collections

```python
store.save_collection(collections, "validation")
store.save_collection(collections, "test")         # held out: saved as held out
collections = store.load_collections(task, source_name="addition")
```

Collections are saved one at a time, held out or not, with each item's
instance and provenance, and a saved name is never overwritten. Loading
gives a `Collections` for the source with every saved collection in the
order it was saved, whether it is held out, and **its reservations**: the
remainder and any new collection still avoid every saved problem. Tuples in
equivalence keys come back as tuples, so keys still match.

### Saving and loading documents

```python
store.save_document("results/assess-1", {"accuracy": 0.83, ...})
store.load_document("results/assess-1")
store.document_names()        # ("results/assess-1", ...), sorted
```

A **document** is anything else saved whole under a name, as a mapping of
payloads the codec can encode: assessment results (see
[Evaluate](evaluate.md#results)), and a training run's settings and
checkpoints (see [Train](train.md#the-output)). A saved name is never
overwritten, and loading a name that was never saved raises
`UnknownDocumentError`.

## Codecs

The store writes the structure; a **codec** writes the payloads inside it:
observations, actions, predictions, teacher contributions, an instance's
inputs. It turns them into JSON-ready values (`None`, booleans, numbers,
strings, lists, and dicts with string keys) and back, so every file is
readable JSON lines whatever the payloads are. **Each file names the codec
it was written with** on its first line, and is read back with that codec,
whichever codec the store writes new files with.

The built-in `"json"` codec handles plain Python values. It tags a tuple as
`{"__tuple__": [...]}` and bytes as `{"__bytes__": "<base64>"}`, so they
come back as they were; anything else raises `TypeError` naming where it is
(`observations.student[1]: the json codec cannot encode ndarray`).

A project whose payloads hold other types, such as arrays, registers its own
codec: an object with `encode(value)` and `decode(value)`, usually handling
its own types and passing everything else to `JsonCodec`:

```python
class ArrayCodec:
    def __init__(self):
        self.json = JsonCodec()

    def encode(self, value):
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
        ...                                   # recurse into lists and dicts
        return self.json.encode(value)

    def decode(self, value): ...

r.codecs.register("arrays", ArrayCodec)
store = RecordStore("runs/baseline", codec="arrays")
```

## Batching

`batches` groups items (records, drawn instances, anything) into batches
for a consumer. **Collation**, turning a list of items into one batch, for
example by padding and stacking, depends on the model's representation, so
the project supplies it. Data does not require torch.

```python
from ssapling.data import batches

for batch in batches(records, pad_and_stack, batch_size=32):
    model.learn("learn_from_batch", batch=batch)
```

- `items` may be a dataset or a stream, even an unbounded one: it is read
  only as far as batches are taken. Batches keep the items' order.
- The last batch may be short; `drop_last=True` drops it instead.
- `collate` is a function, or a `ComponentSpec` naming a collator registered
  in `ssapling.data.collators` (also `ssapling.registries.collators`), built
  with its settings.

**Workers.** `workers=4` collates in four worker processes, a few batches
ahead of the consumer. They are started with `spawn`, which works on every
platform, so each worker builds its collator afresh: a module-level function
or a picklable object is sent to it, and a `ComponentSpec` is rebuilt from
the registry after `setup` (usually the project's `register_components`)
has run in that worker:

```python
batches(records, ComponentSpec("pad", {"width": 30}), batch_size=32,
        workers=4, setup=register_components)
```

An error in `collate` or in `setup` reaches the caller as itself, raised
where that batch would have been. Closing the iterator stops the workers.

## Errors

| Error | Means |
| --- | --- |
| `TypeError` from the record types | A value of the wrong type: per-agent values not keyed by agent id, agent ids that are not strings, a reset, transition, response, step or summary of the wrong type, or an integer field given something else. |
| `ValueError` from the record types | An empty or repeated id; a per-agent value naming an agent that did not act; an unknown clock, placement or status; a teacher placed in the environment of a static record; a schema version newer than this SSAPling reads. |
| `ValueError` from `RecordBuilder` | A step that breaks one of the [checks](#building-a-record), or a finished record changed. |
| `TypeError` from `Provenance` | Values that are not plain (every one is listed, with its path), a settings instance instead of a mapping, or a seed or position that is not an integer. |
| `ValueError` from `Provenance` | No seed and no position, an empty name, sampler settings without a sampler, or a negative position. |
| `TypeError` from a sampler | A listed source given to `generated`, or a generated one to `uniform` or `stratified`. |
| `ValueError` from a sampler | A source with no instances, an instance without the `by` descriptor, or weights naming a value that does not exist. |
| `ConfigError` from a sampler | Invalid settings or weights (negative, all zero, `vary` without `values`, ...), whether given through the registry, made directly, or passed to `set_weights`. |
| `SourceExhaustedError` (a `LookupError`) | `max_draws` draws in a row were rejected while building a collection or drawing the remainder. Names the sampler and the reserved collections. |
| `HeldOutError` (a `LookupError`) | A held-out collection was asked for without `purpose="evaluation"`. |
| `UnknownCollectionError` (a `LookupError`) | No collection has that name; the error lists those that exist. |
| `StoreError` (a `ValueError`) | A store's files are not what SSAPling writes: a non-empty directory that is not a store, a newer store version, an invalid line (named with its file and line), or a file whose first line names no codec. |
| `UnknownRecordError` (a `LookupError`) | The store holds no record with that id. |
| `UnknownDocumentError` (a `LookupError`) | The store holds no document with that name. |
| `ValueError` from `RecordStore` | A record id, collection name or document name already in the store, a step that breaks a [builder check](#building-a-record), or a finished record changed. |
| `TypeError` from a codec | A payload the codec cannot encode, named with where it is. Nothing is written. |
| `UnknownComponentError` from `RecordStore` or `batches` | A codec or collator name that is not registered. |
| `ValueError`, `TypeError` from `batches` | `batch_size` below 1, negative `workers`, or a `collate` or `setup` of the wrong type. |
| `ValueError`, `TypeError` from `Collections` | A taken or empty name, a negative size, `max_draws` below 1, a seed that is not an integer, a sampler drawing from another source, a sampler whose `draw` does not return a `Sample`, or a source that neither generates nor lists. |
