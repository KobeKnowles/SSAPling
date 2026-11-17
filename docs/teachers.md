# Teachers

`ssapling.teachers` defines how components **ask a teacher for help and
receive it**. A teacher can give supervision, feedback, explanations,
demonstrations, interventions, or proposed practice problems, and it need
not give all of these. Human, LLM, neural, and scripted teachers all
implement the same contract. SSAPling contains no teaching strategies, human
interfaces, or LLM clients. Those belong downstream.

| Part | What it is |
| --- | --- |
| [Request](#requests) | One request for help: the operation, the context the teacher may use, and the proposals to review. |
| [Response](#responses) | The teacher's answer, or the reason there is none: a status, plus contributions keyed by kind and agent. |
| [Teacher](#the-teacher-contract) | Any object with `operations` and `respond`. `respond` may be `def` or `async def`. |
| [Runner](#asking-a-teacher-the-runner) | One awaitable entry point. It checks the request, calls the teacher, and turns any problem with the answer into a response. |
| [Teaching resources](#teaching-resources) | A teacher declares what it needs from the task, and this is checked before the run. |
| [Lifecycle](#lifecycle) | Optional `reset`, `close`, `export_state`, `restore_state`. |
| [Registry](#teachers-as-components) | Teachers are components, built through Core. |

SSAPling ships no teachers. The tests use toy teachers in
`tests/fixtures/teachers.py`. `DigitTutor` is the one to read: it reviews an
addition answer digit by digit, using the task's teaching resource.

## A worked example

A student is adding 45 + 57 and is about to write the units digit. They
propose 9. The coordinating code asks the tutor to review the proposal:

```python
from ssapling import tasks, teachers
from ssapling.teachers import TeacherRequest, TeacherRunner

task = AdditionTask()
teachers.check_resources(DigitTutor, tasks.teaching_resources(task))  # before the run
tutor = DigitTutor(task=task)

instance = task.generate(params={"digits": 2}, seed=7)       # 45 + 57, reference 102
request = TeacherRequest(
    id="review-1",
    operation="review_action",
    context={"instance": instance, "state": ""},  # what the teacher may use
    proposals={"student": "9"},                   # what the student wants to do
    episode="ep-1", participant="student", step=0,
)
response = await TeacherRunner().request(tutor, request)
# TeacherResponse(request_id="review-1", status="answered",
#     contributions={"supervision": {"student": "2"},
#                    "feedback": {"student": "Not quite. Add the units, carry, then add the tens."}})
```

The tutor saw the reference (102) because it was in the context. The student
never sees the context. What reaches the student is decided by the
coordinating code, such as the [episode loop](interaction.md#the-episode-loop).

## Requests

```python
TeacherRequest(
    id="review-12",                    # matches the response to the request
    operation="review_action",         # what is asked
    context={"instance": ..., "state": "4"},   # what the teacher may use
    proposals={"student": "7"},        # proposed actions, by agent id
    episode="ep-3", participant="student", step=12,   # optional: where it came from
)
```

| Field | Meaning |
| --- | --- |
| `id` | Identifies the request. The runner checks that the response carries the same id. |
| `operation` | What is asked, such as `"review_action"`, `"explain"`, or `"propose_instances"`. The names are downstream choices. |
| `context` | What the teacher may use, keyed by name. It need not match the student's view: a teacher may be given the reference, the full state, or the task's resources. |
| `proposals` | Proposed actions to consider, keyed by agent id. Empty for operations that review nothing. |
| `episode`, `participant`, `step` | Optional. For live interaction, they say which episode, agent, and step the request is about, so its response can be associated with the right moment. |

**A request captures its context when it is made.** It keeps its own copy of
`context` and `proposals`. If the episode moves on and the coordinating code
changes its own dictionaries, a teacher that answers late still reads the
state it was asked about. The values inside are treated as read-only, like an
instance's payloads. Pass values that will not change afterwards: for
example, copy a buffer that the environment reuses.

## Responses

Every response has a **status**, so a missing answer is recorded as missing,
never as silence:

| Status | Meaning | Contributions |
| --- | --- | --- |
| `answered` | The teacher answered. | Any, possibly none. |
| `abstained` | The teacher chose not to contribute. | None. The reason goes in `detail`. |
| `failed` | No answer: the teacher raised an error or gave an invalid response. | None. |
| `cancelled` | No answer: the request was cancelled first, for example when no answer came in time. | None. |
| `stale` | An answer that arrived after the state it was about had moved on. | Kept for the record, **never applied automatically**. |

```python
TeacherResponse.answered(request, {"feedback": {"student": "Carry the 1."}})
TeacherResponse.abstained(request, "the answer is already complete")
TeacherResponse.failed(request, "the model returned no answer")
TeacherResponse.cancelled(request, "the human closed the dialog")
response.as_stale()        # an answer, marked as too late to apply
```

These constructors copy the request's id, so the response always matches its
request.

**Contributions** are keyed by kind, and then by agent id:

```python
{
    "supervision": {"student": "2"},             # a target, not a command
    "feedback": {"student": "Carry the 1."},     # not a reward
    "intervention": {"student": "2"},            # change the executed action, where supported
}
```

Kinds are open-ended, so a downstream project can add a form of teaching
without changing SSAPling. Common ones are `supervision`, `feedback`,
`intervention`, `demonstration`, `explanation`, and `instances` (proposed
practice problems). Payloads pass through uninterpreted. The distinctions
matter:

- **Supervision describes a target.** It does not make the environment do
  anything.
- **An intervention is the only thing that changes an executed action.** It
  is applied only through mechanisms the environment supports, and the
  episode loop applies one only when told to (`intervene`).
- **Feedback is not a reward.** It becomes one only if a downstream project
  chooses so.

**A missing answer is never approval.** Only an `answered` response can
approve or change anything. `abstained`, `failed`, and `cancelled` responses
have no contributions at all; constructing one with contributions is an
error. So nothing in them can be applied by mistake. A `stale` response keeps
its contributions so the record shows what the teacher said, but they are
never applied automatically. The coordinating code marks an answer stale
with `response.as_stale()` when it arrives after the state it was about has
moved on, for example in real-time play. Only an answer can be
marked stale; the other statuses have nothing to apply.

## The teacher contract

A **teacher** is any object with two members. It needs no base class:

```python
class DigitTutor:
    operations = frozenset({"review_action"})

    def respond(self, request):                 # or: async def respond(self, request)
        ...
        return TeacherResponse.answered(request, {...})
```

- **`operations`**: the names of the operations it supports. A teacher whose
  operations depend on its settings can set them in `__init__`.
- **`respond(request)`**: returns a `TeacherResponse` to that request. Write
  it as `async def` for a teacher that waits on something, such as a remote
  model or a human; the runner awaits either kind the same way.

Consumers ask before requesting an operation:

```python
from ssapling import teachers

teachers.supports(tutor, "review_action")   # True
teachers.supports(tutor, "explain")         # False
```

Operations are an **open set**: a name the teacher has never heard of is
simply unsupported, not an error. This differs from `tasks.supports`, whose
features are a fixed list.

**`operations` is the only declaration.** To switch an operation off, for
example by a setting, leave it out of `operations`. Unlike a task, a teacher
has no `supports` method of its own; if it has one, it is not consulted. That
keeps one object that is both a task and a teacher (see
[Roles, not objects](tasks.md#roles-not-objects)) from mixing the two.
`operations` must be a collection of names: a single string is rejected,
because `"review" in "review_action"` would be true.

A teacher may also have these members, each optional:

| Member | Meaning |
| --- | --- |
| `required_context` | For each operation, the context names a request must give: `{"review_action": ("instance", "state")}`. The runner checks them before asking. Every key must be one of `operations`, so a typo is caught. |
| `required_resources` | The teaching resources it needs from the task: `("procedure",)`. See [Teaching resources](#teaching-resources). |
| `reset()`, `close()` | Forget episode state; release what it holds. See [Lifecycle](#lifecycle). |
| `export_state()`, `restore_state(state)` | Save and restore its persistent state, for checkpoints. |

`Teacher` is a `typing.Protocol`, so a type checker verifies a downstream
teacher structurally. Nothing inherits from it.

## Asking a teacher: the runner

The runner is the one entry point for asking any teacher, synchronous or
asynchronous:

```python
runner = TeacherRunner()                       # sync teachers run in the loop's thread
runner = TeacherRunner(placement="thread")     # sync teachers run in a worker thread

if teachers.supports(teacher, "review_action"):
    response = await runner.request(teacher, request, timeout=2.0)
```

It checks that a request can be asked and that the response answers it. It
does not judge whether the advice is correct.

**Before asking**, the runner checks the request against the teacher. A
failure here is a mistake in the calling code, so it raises, and the teacher
is not asked:

| Problem | Error |
| --- | --- |
| The teacher does not list the operation. | `UnsupportedOperationError`, listing what it does support. |
| The request lacks context the teacher requires for the operation. | `MissingContextError`, listing what is missing and what was given. |

**After asking**, anything that goes wrong with the teacher becomes a
response, so it is recorded and the run goes on:

| What happened | Response |
| --- | --- |
| `respond` raised an error. | `failed`, with `"RuntimeError: boom"` as the reason. The error and its traceback are also logged to `ssapling.teachers`. |
| `respond` returned something other than a `TeacherResponse`. | `failed` |
| The response is for a different request id. | `failed`, naming both ids. |
| No response within `timeout` seconds. | `cancelled`, with `"no response within 2.0 s"`. |
| The teacher cancelled the request itself, for example a human closing the dialog. | `cancelled` |
| The teacher's own timeout fired, such as its HTTP client's. | `failed`: the teacher failed; the runner did not give up on it. |

None of these responses has contributions, so none can be taken for approval.

### Where a synchronous teacher runs

| `respond` | `placement="direct"` (default) | `placement="thread"` |
| --- | --- | --- |
| `def` | Runs in the event loop's thread, blocking the loop until it returns. | Runs in a worker thread. The loop keeps going. |
| `async def` | Awaited in the loop. | The same: awaited in the loop. |

A synchronous teacher is not made non-blocking by being awaited. A slow one,
such as one that makes a blocking HTTP call, should use `placement="thread"`.
Pass `executor=ThreadPoolExecutor(max_workers=1)` for a teacher that must
always be called from the same thread. To run a teacher in a separate
process, build it with `in_process` (see
[Interaction](interaction.md#running-a-component-in-another-process)).

### Timeouts and cancellation

These use standard `asyncio`. Retry and timeout policies are the caller's
choice:

- **`timeout=`** gives up waiting after that many seconds and returns a
  `cancelled` response, which the caller can record. It cannot interrupt a
  synchronous teacher run directly, because that teacher blocks the loop, so
  combining the two is rejected with a `ValueError`. Use
  `placement="thread"`, where the runner stops waiting and the thread
  finishes on its own.
- **Cancelling the caller** works as usual: if the task awaiting
  `runner.request(...)` is cancelled by `asyncio.wait_for`, a `TaskGroup`, or
  shutdown, `CancelledError` (or `TimeoutError` from `wait_for`) reaches the
  caller. The runner never swallows it, because that would leave the rest of
  the program unaware that it was cancelled. A caller that keeps a record
  writes `TeacherResponse.cancelled(request, reason)` itself.

## Teaching resources

A task may offer task-level material such as a solving procedure (see
[Tasks: teaching resources](tasks.md#teaching-resources)). A teacher that
needs some declares them:

```python
class DigitTutor:
    required_resources = ("procedure",)

    def __init__(self, config=None, *, task):
        self.procedure = tasks.teaching_resource(task, "procedure")
```

The coordinating code checks the requirement **before the run**, so a
mismatch is reported up front rather than mid-episode:

```python
teachers.check_resources(DigitTutor, tasks.teaching_resources(task))
# MissingResourcesError: Teacher DigitTutor needs teaching resources that are
# not available: 'procedure'; none are available.
```

- `check_resources` takes the teacher or its class, so a teacher can be
  checked before it is built. To check a class, `required_resources` must be
  a class attribute. It lists every missing resource at once.
- It works on plain names, so `teachers` never imports `tasks`. The caller
  passes `tasks.teaching_resources(task)`, whose keys are the names.
- How the teacher *gets* the material is up to it. A teacher built with the
  task as a live object (`task=task`) fetches what it needs with
  `tasks.teaching_resource`. That call also fails clearly if a resource is
  missing, so a teacher built without the check still fails at construction,
  not mid-episode.

## Lifecycle

A teacher may have any of these methods, each optional, as `def` or
`async def`:

| Method | When it is called |
| --- | --- |
| `reset()` | Before a new episode, to forget episode state. |
| `close()` | When the teacher is no longer needed, to release what it holds, such as a connection. |
| `export_state()` | At a checkpoint, to save persistent state. Returns plain data. |
| `restore_state(state)` | On resume, to restore what `export_state` returned. |

**One owner** calls these methods: whichever component owns the teacher. That
is the coordinating code, or the environment when the teacher sits inside it.
So a teacher is never reset or closed twice by different layers. The
episode loop resets a teacher it consults at the start of each episode, but
does not close it: whoever built the teacher closes it when it is no longer
needed, such as a training run. [Checkpoints](train.md#checkpoints-and-resuming)
export and restore. State export is optional; a human or remote teacher need
not be exactly resumable, and when a run resumes, a warning names any
component without it rather than silently starting it fresh.

## Teachers as components

Teachers are registered and built through Core like everything else. The
registry is `ssapling.teachers.registry`, also reachable as
`ssapling.registries.teachers`. Live objects, such as the task, are passed at
construction:

```python
from ssapling import registries as r

r.teachers.register("digit_tutor", DigitTutor)   # DigitTutor.Settings is picked up
tutor = r.teachers.create("digit_tutor", {"intervene": True}, task=task)
```

Which live objects a teacher takes is up to the teacher. A teacher that
needs the task, as `DigitTutor` does, declares a `task` parameter.

## Where teachers sit

A teacher can sit in an external coordinator that calls the environment,
inside the environment around its dynamics, or in control of a participant.
Placement is independent of teaching strategy, and the episode loop supports
each (see [Interaction](interaction.md#teachers)). An environment that owns a teacher calls it directly, using
the same request and response types.

That is why `ssapling.teachers.types` is special: `environments` and `data`
may import it, and nothing else from `teachers`. A transition result can
report teacher responses, and a record step holds them. So the types module
holds plain data and imports nothing else from `teachers`;
`tests/test_layering.py` checks this. Requests and responses also pickle, so
they can be sent to a teacher running in another process.

## Errors

| Error | Means |
| --- | --- |
| `UnsupportedOperationError` (a `LookupError`) | The teacher does not list the requested operation; the error lists what it does. |
| `MissingContextError` (a `LookupError`) | The request lacks context the teacher requires for that operation; the error lists what is missing and what was given. |
| `MissingResourcesError` (a `LookupError`) | The teacher requires teaching resources that are not available; the error lists every missing one and what is available. |
| `ValueError` from `TeacherRequest`, `TeacherResponse` | An invalid value: an empty id, an unknown status, contributions on a response without an answer, a negative step. |
| `TypeError` from `TeacherRequest`, `TeacherResponse` | A value of the wrong type: context or contributions that are not mappings with string keys, a step that is not an integer. |
| `TypeError` from `teachers.supports`, the runner | The object is not a teacher: no `operations` or `respond`, or `operations` is a single string. |
| `ValueError` from the runner | A timeout that is not a positive number, a timeout for a synchronous teacher run directly, or `required_context` naming an operation that is not in `operations`. |

A teacher's own errors are never raised by the runner: they become `failed`
responses.
