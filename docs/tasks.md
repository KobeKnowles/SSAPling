# Tasks

`ssapling.tasks` defines **problems**: what is to be solved and how success is
judged. A task holds no data pipeline, teaching strategy, or world simulation.
Those belong to Data, Teachers, and Environments.

| Part | What it is |
| --- | --- |
| [Instance](#instances) | One concrete problem: id, inputs, optional reference, descriptors. |
| [Task](#the-task-contract) | A family of instances, with `score`. Generated from a seed, or enumerated from a fixed set. |
| [Score](#scoring) | The judgement of one outcome: local, external, or unavailable. |
| [Partial checks](#partial-checks) | Optional: judging an intermediate state. |
| [Teaching resources](#teaching-resources) | Optional: task-level material for teachers. |
| [Features](#features) | How a consumer finds out what a task offers. |
| [Registry](#tasks-as-components) | Tasks are components, built through Core. |

SSAPling ships no tasks. A downstream project writes a class with the methods
below and registers it. The tests use two toy tasks, `AdditionTask` and
`PuzzleBook` in `tests/fixtures/tasks.py`, which are good examples to read.

## Instances

An **instance** is one concrete problem, for example one ARC puzzle with its
demonstrations and test input, or one addition.

```python
from ssapling.tasks import Instance

Instance(
    id="add-2d-17",                 # stable identity
    inputs={"a": 42, "b": 17},      # what the student is shown
    reference=59,                   # the answer, for teachers and evaluators
    descriptors={"digits": 2},      # task-defined attributes, passed through
    equivalence_key=("add", 42, 17) # optional: "which problem is this"
)
```

| Field | Meaning | Who uses it |
| --- | --- | --- |
| `id` | A stable identifier. Records, scores, and teacher requests refer back to it. | Everyone |
| `inputs` | Everything presented for solving. **The only part a student sees.** Required: write `inputs=None` explicitly for a remote instance with no local inputs, such as an ARC-AGI-3 game id, so inputs are never left out by accident. | The environment reset, the static solver |
| `reference` | Optional. An expected output, a reference solution (such as a sequence of steps), or both. | Teachers, `score` |
| `descriptors` | Optional, task-defined, uninterpreted by SSAPling. | Data samples by them; Evaluate groups results by them |
| `equivalence_key` | Optional. Instances with the same key are the same problem to the task. Must be hashable, such as a string or a tuple of strings and numbers; anything else is rejected when the instance is made. | Data keeps them in one collection |

**Visibility follows from the structure.** The student is given
`instance.inputs` and nothing else. Downstream code that resets an environment
passes only the inputs, and the coordinating code decides what enters teacher
context and records. The reference reaches the student only if a downstream
project explicitly chooses so.

**`problem_key`** is the equivalence key, or the id when the task declares
none. Data uses it when building non-overlapping collections. Two seeds of
an addition task can produce the same pair of numbers; with
`equivalence_key=("add", a, b)` they count as one problem and never land in
both a training and a validation set.

An instance's fields cannot be reassigned. Its payloads and `descriptors` are
ordinary Python objects, so SSAPling cannot stop them being changed, but it
treats them as read-only: do not change them once the instance exists.
Instances compare by value and hash by `id` alone, so they work in sets even
when their payloads are lists. They pickle, so they can be sent to worker
processes.

## The task contract

A **task** is a family of instances. It is any object with a `score` method;
it needs no base class. Two kinds of task produce instances:

```python
class AdditionTask:                                   # generated
    def generate(self, *, params=None, seed: int) -> Instance: ...
    def score(self, instance, outcome) -> Score: ...

class PuzzleBook:                                     # enumerated
    def ids(self) -> Sequence[str]: ...
    def get(self, instance_id: str) -> Instance: ...  # raises UnknownInstanceError
    def score(self, instance, outcome) -> Score: ...
```

- **Generated:** `generate(params=..., seed=...)` makes an instance from
  task-defined parameters and a seed. The same parameters and seed must give
  an equal instance, and the id should follow from them (`"add-2d-17"`).
- **Enumerated:** a fixed set. `ids()` lists them in a stable order and
  `get(id)` returns one. An unknown id raises `UnknownInstanceError`.

A task produces instances. It does not decide which are used, in what order,
or how often. That belongs to Data, a Train schedule, or a teacher. A task is
also independent of any environment: turning an instance into an
environment's reset options is downstream code for that task–environment pair.

The protocols (`Task`, `GeneratedTask`, `EnumeratedTask`, `PartialChecker`,
`TeachingResourceProvider`) are `typing.Protocol`s, so a type checker verifies
a downstream task structurally. Nothing inherits from them.

## Scoring

A task scores **one outcome of one instance**, the smallest unit of
measurement. Aggregation belongs to Evaluate.

```python
score = task.score(instance, outcome)
```

A `Score` has a status, and the status says where the value came from:

| Status | Meaning | `value` |
| --- | --- | --- |
| `local` | Computed by the task. | The score (task-defined; often a number in [0, 1] or a bool). |
| `external` | Reported by a service, such as an ARC-AGI-3 scorecard, and recorded as reported. | What the service reported. |
| `unavailable` | No score can be given: the reference is hidden, or the service has not reported yet. | Always `None`. The reason goes in `detail`. |

```python
Score.local(1.0)
Score.external(0.75, detail={"source": "scorecard"})
Score.unavailable("the reference output is hidden")
score.available     # True unless unavailable
```

**A score is never invented.** A `local` or `external` score must have a
value, and an `unavailable` score must not. [Evaluate](evaluate.md#aggregation)
carries this through: an unavailable score is excluded from aggregates, never counted as 0.

`detail` is for task-defined extras: per-cell correctness, the raw service
report, the reason a score is unavailable.

## Partial checks

A task may optionally judge an **intermediate state**, for example whether a
half-finished answer is still valid, how far along it is, or where the first
mistake is. Teachers use this to intervene mid-episode.

```python
if tasks.supports(task, "check_partial"):
    report = task.check_partial(instance, state)     # state is task-defined
    report.valid, report.progress, report.first_error, report.detail
```

Every field of a `PartialReport` is optional, because tasks differ in what they
can judge part-way. `progress` is a number in [0, 1]; any real number is
accepted, such as a NumPy float, but not a `bool`.

## Teaching resources

A task may offer task-level material for teachers, such as a worked example or
a solving procedure, keyed by name:

```python
teaching_resources(task)            # {"procedure": "Add the units, carry, ..."} or {}
teaching_resource(task, "procedure")
# MissingTeachingResourceError: The task has no teaching resource 'worked_example';
# it offers: 'procedure'.
```

A teacher that needs a resource declares the requirement, and the
coordinating code checks it against `teaching_resources(task)` before the
run, so a mismatch is reported up front rather than mid-episode:

```python
teachers.check_resources(teacher, tasks.teaching_resources(task))
```

See [Teachers: teaching resources](teachers.md#teaching-resources).

## Features

A task declares an optional feature by **implementing its methods**:

| Feature | Methods |
| --- | --- |
| `generate` | `generate` |
| `enumerate` | `ids`, `get` |
| `check_partial` | `check_partial` |
| `teaching_resources` | `teaching_resources` |

Consumers ask before using one:

```python
from ssapling import tasks

tasks.supports(task, "check_partial")   # True or False
tasks.supports(task, "solve")           # ValueError: not a task feature (a bug in the caller)
```

Write it as `tasks.supports(...)`, not a bare `supports(...)`. Environments,
teachers and models have a `supports` of their own (`environments.supports`,
`teachers.supports`, `models.supports`), and code such as the interaction
loop uses several at once.

A task can also **opt out** of a feature its class implements, for example
when a setting turns partial checks off, by defining a `supports(feature)`
method and answering `False`. That method only switches features off: it
answers `False` for those and `True` for every other feature.

```python
class MyTask:
    def supports(self, feature):
        return self.config.partial_checks if feature == "check_partial" else True
```

- A task can never claim a feature it lacks the methods for: `supports`
  returning `True` is not enough.
- A task's own `supports` must not call `tasks.supports`, because
  `tasks.supports` calls it, and the two would call each other forever.
- `supports` must be a method. A task with any other attribute of that name,
  such as a tuple of feature names, is rejected with a `TypeError`.

## Tasks as components

Tasks are registered and built through Core like everything else. The task
registry is `ssapling.tasks.registry`, also reachable as
`ssapling.registries.tasks`:

```python
from ssapling import registries as r

r.tasks.register("addition", AdditionTask)        # AdditionTask.Settings is picked up
task = r.tasks.create("addition", {"max_digits": 3})
```

`ssapling.registries` collects every area's registry, so a downstream
`register_components()` needs one import.

## Static tasks and remote instances

**Static tasks** (ARC-AGI-2 style, input → output, no environment) need nothing
beyond this contract. A solver receives `instance.inputs`, produces a
prediction, and `task.score(instance, prediction)` judges it where a reference
is available:

```python
instance = task.generate(params={"digits": 2}, seed=42)
prediction = my_solver(instance.inputs)    # downstream code
score = task.score(instance, prediction)
```

[Interaction's `solve`](interaction.md#static-solving) is SSAPling's own
static solving: it presents the inputs to a model and records its
predictions.

**Remote instances** (ARC-AGI-3 style) can be identifiers only:
`Instance(id="game-a", inputs=None)` with no inputs and no reference. An environment
adapter resolves the id against the service, and `score` reports the service's
result as `external`, or `unavailable` until it arrives.

## Roles, not objects

Task, Data source, and Teacher are contracts, and one downstream object may
fulfil several. A teacher that creates practice problems can act as an
instance source directly. Nothing in SSAPling assumes a fixed dataset or test
set exists; collections, including held-out ones, belong to Data.

## Errors

| Error | Means |
| --- | --- |
| `UnknownInstanceError` (a `LookupError`) | An enumerated task has no instance with that id. |
| `MissingTeachingResourceError` (a `LookupError`) | The task offers no resource of that name; the error lists what it offers. |
| `ValueError` from `Instance`, `Score`, `PartialReport` | An invalid value: an empty id, a score with no value, progress outside [0, 1]. |
| `TypeError` from `Instance`, `PartialReport` | A value of the wrong type: descriptors that are not a mapping with string names, an unhashable equivalence key, progress that is not a number. |
| `TypeError` from `tasks.supports` | The task has a `supports` attribute that is not a method. |
