# Interaction

`ssapling.interaction` runs attempts: episodes with an environment, and
static solving without one. It is the one loop that Train's interact stage
and Evaluate's assessments both use, so an attempt behaves the same in
training and in assessment apart from the settings passed to it. It drives
an environment, models and teachers, whether each is written with `def` or
`async def` and wherever it runs, and writes the record step by step.

| Part | What it is |
| --- | --- |
| [The episode loop](#the-episode-loop) | `run_episode`: reset, act, consult the teacher, step, record. |
| [Timing](#timing) | How long a model has to act: turn-based, deadline or real-time, on a simulated or wall clock. Late outputs and default actions. |
| [Lanes](#lanes) | Several episodes at once that share a model. |
| [Static solving](#static-solving) | `solve`: present an instance's inputs to a model, with no environment. |
| [Running a component in another process](#running-a-component-in-another-process) | `in_process`: build a component in a worker process and use it through a stand-in. |

## The episode loop

```python
from ssapling.interaction import run_episode

record = await run_episode(env, model, instance, record_id="ep-1", seed=7)

record = await run_episode(
    env, {"player_1": alice, "player_2": bob}, instance, record_id="ep-2",
    teacher=tutor, store=store, settings={"intervene": True, "max_steps": 50},
)
```

`run_episode` runs one episode and returns its record. It is `async`; from
synchronous code, call it with `asyncio.run(run_episode(...))`.

1. **Start.** The environment is reset with `options=instance.inputs` (and
   `seed=`, if given). The record's header is written. Each model's episode
   state is reset (`reset_episode`), and so is the teacher's (`reset`).
2. **Each step.**
   - The agents whose turn it is (`environments.next_agents`) each give an
     action. Their model perceives the agent's latest observation (`None` if
     it has had none), and the loop waits for its output, as the
     [timing mode](#timing) allows.
   - If there is a teacher, it is asked to review the proposals.
   - The environment steps, and the step is appended to the record.
3. **End.** When the environment says the episode is over, the summary is
   terminated or truncated, as the environment reports. With `max_steps`, an
   episode still going after that many steps is truncated, with
   `"step limit"` as the detail.

**Only the instance's inputs reach the environment**, and so the models. The
reference and teaching resources never do, unless a downstream project
passes them on itself. If the inputs are `None`, as for a remote game
identified by its id, the reset gets no `options`.

### Models

`model` is one model controlling every participant, or a mapping from each
participant's agent id to its model. A model that controls several
participants, as in self-play, gets each one's agent id as `key`, so it can
keep their episode state apart. A model controlling one participant is
called without a key.

The loop drives every model through its buffers (`as_buffered`, see
[Models](models.md#buffered-models)): it perceives, then the loop calls
`poll` and `advance` until there is an output or time runs out. An
act-only model needs one `advance`, which calls its `act`. A buffered model
with no `advance`, one that works on its own in the background, is only
polled, so it needs the wall clock.

### Teachers

| Where the teacher sits | How to run it | What the record shows |
| --- | --- | --- |
| In the coordinating code | Pass `teacher=`. The loop asks it once per step, before the environment steps. | Its response, and the executed actions: the proposals and any default actions, with its interventions applied if `intervene` is on. |
| In the environment | Pass `teacher_placement="environment"`. The environment asks it inside `step`. | What the environment reports in its `StepResult`, copied unchanged. |
| Controlling a participant | It is one of the models; pass `teacher_placement="participant"`. | Its actions, as for any model. |

So the record reads the same whatever the placement: the tests run the same
episode with the teacher in the loop and in the environment, and compare.

The loop's request for each step has the id `"<record id>:<step>"`, the
`teacher_operation` from the settings (`"review_action"` by default), the
acting agents' proposals, and the episode and step. Its context is
`{"instance": instance, "observations": {...}}`: the whole instance,
reference included, since a teacher may use it, and the acting agents'
observations. A teacher that needs another context gets it from a
`teacher_context` function:

```python
def digits_so_far(instance, observations):
    return {"instance": instance, "state": observations["student"]["entered"]}

await run_episode(env, model, instance, record_id="ep-1",
                  teacher=tutor, teacher_context=digits_so_far)
```

**A teacher observes unless `intervene` is on.** Its responses are recorded
either way, but only with `intervene` do the interventions of an answered
response replace the proposals. No other status ever changes an action.

The loop waits for the teacher's response, up to `teacher_timeout`; after
that the response is `cancelled`. The loop resets a teacher it consults, but
does not close it: whoever built it closes it.

### Settings

`settings` is an `EpisodeSettings`, or a mapping of the same values, so a
run file can set them.

| Setting | Meaning | Default |
| --- | --- | --- |
| `max_steps` | Truncate the episode after this many steps. | none |
| `intervene` | Apply the teacher's interventions. | `False`: the teacher only observes |
| `teacher_operation` | The operation the teacher is asked for. | `"review_action"` |
| `threads` | Which components run their synchronous calls in a worker thread of their own: any of `"environment"`, `"model"`, `"teacher"`. | none: all run directly |
| `environment_timeout` | Seconds to wait for the environment's reset or step. | none |
| `teacher_timeout` | Seconds to wait for the teacher. | none |
| `timing`, `clock`, `budget`, `late_outputs`, `default_actions` | See [Timing](#timing). | turn-based, simulated clock |

A synchronous component runs directly in the event loop unless it is listed
in `threads`. A time limit can only interrupt a synchronous call that runs
in a thread, so a limit for one that runs directly is refused before the
episode starts. Each threaded component gets one thread of its own, so its
calls never overlap. For a model, the thread runs `advance` and
`reset_episode`; `perceive` and `poll` are always called directly, since
they return at once. A component in another process (see below) is used
like any other; give it a thread too if it is synchronous.

### When something goes wrong

| What happens | The record | The caller |
| --- | --- | --- |
| The environment cannot be reached during a step (`EnvironmentCommunicationError`) | The step is kept with no transition, and the summary is `failed`, with the error as the detail. | Gets the record. |
| Anything else: a model's or teacher's bug, the environment's own error, a cancelled task | Kept, with the steps so far, and the summary is `failed`, with the error as the detail. | Gets the error, as itself. |
| A mistake in the call: a participant with no model, a model with no `advance`, an unusable timeout, a teacher with a placement other than `"coordinator"` | None is begun. | Gets a `ValueError` or `TypeError`. |

With `store=`, the record is written to the store step by step as well, so
a failed record is kept there too.

**Each step, as it happens.** With `on_step=`, each step is also passed to a
function of yours once it is recorded, before the next step begins. Train's
interact stage learns online this way. The function may be `async def`, and
an error it raises ends the episode like any other.

### What the loop leaves to others

- **Closing** the environment, the teacher or the models: their owner does it.
- **`set_evaluating`** and learning: [Evaluate](evaluate.md) and [Train](train.md).
- **Stale responses:** the loop waits for its teacher, so it never marks one
  stale.
- **Natively batched environments** (`step_lanes`): the loop runs one
  episode, with one environment. Several episodes at once are separate
  calls (see [Lanes](#lanes)).

## Timing

How long the loop waits for a model's output is the **timing mode**,
measured on a **clock**. Both are settings:

```python
settings = {"timing": "deadline", "clock": "simulated", "budget": 3,
            "late_outputs": "discard", "default_actions": {"student": "0"}}
```

| `timing` | The model acts... |
| --- | --- |
| `"turn_based"` (default) | ...when it has an output, however long that takes. |
| `"deadline"` | ...as soon as it has an output, within `budget`. |
| `"real_time"` | ...at the end of a tick, `budget` long, with the output it has by then. |

| `clock` | Time is counted in... | Use it for |
| --- | --- | --- |
| `"simulated"` (default) | calls to the model's `advance`, made by the loop. Nothing else takes time. | Reproducible runs: "the model had 3 steps to think" means the same on every machine. Repeated runs give identical records. |
| `"wall"` | seconds. | Real-time and remote environments, and models that work on their own. |

`budget` is required for `"deadline"` and `"real_time"`, and not allowed for
`"turn_based"`. On the simulated clock it is a whole number of calls to
`advance`.

**No output in time.** The agent's entry in `default_actions` is executed.
An agent without one is left out of the step, so the environment applies
its own default. Either way, the step has no proposal for the agent.

**Late outputs.** An output that arrives after its deadline answers an
earlier observation. The loop tells, because it counts each agent's
observations perceived and outputs read, and every output answers one
observation, in order. With `late_outputs="discard"` (the default) it is
thrown away and the loop keeps waiting for the answer to the latest
observation. With `"use"`, it becomes the agent's action at its next turn.

**On the wall clock**, a call to `advance` still running when time runs out
keeps running in the background, and its output arrives late. A synchronous
model needs `threads=["model"]` for that, so a wall-clock budget for one
that runs directly is refused before the episode starts.

**The record** shows the timing of every acting agent at every step, in
`step.timing`:

```python
step.timing["student"]   # AgentTiming(elapsed=3, defaulted=True, late="discarded")
```

| Field | Meaning |
| --- | --- |
| `elapsed` | From perceiving to the output: calls to `advance`, or seconds. The whole budget when no output came. |
| `defaulted` | No output came in time, so the default (or the environment's own) was used. |
| `late` | A late output arrived at this step: `"discarded"` or `"used"`. `None` otherwise. |

After each decision, the loop also asks the model for its own report, if it
has a `diagnostics()` method, and keeps it in `step.diagnostics` (see
[Models: diagnostics](models.md#diagnostics)).

The header's `clock` names the clock. Agents acting in the same step decide
at the same time, each with its own budget.

## Lanes

Several episodes can run at once, each with its own environment. When they
share a model, each episode's **lane** keeps its episode state apart: the
lane is the model's `key` (or `(lane, agent)` when the model also controls
several participants).

```python
records = await asyncio.gather(*(
    run_episode(make_env(), model, instance, record_id=f"ep-{lane}", lane=lane)
    for lane, instance in enumerate(instances)
))
```

## Static solving

`solve` is the episode loop without an environment, for static tasks such
as ARC-AGI-2: it presents an instance's inputs to a model and records its
predictions.

```python
from ssapling.interaction import solve

record = await solve(model, puzzle, record_id="p-1", attempts=2)
record.steps[0].presented     # {"solver": puzzle.inputs}
record.steps[0].predictions   # {"solver": the model's first answer}
```

- **Each attempt is one step.** The model perceives the whole of
  `instance.inputs`, never the reference, and the loop waits for its
  prediction under the timing settings. A prediction that does not come in
  time is left out, and the step's timing says so.
- **The record is static:** no reset, and the steps have `presented` and
  `predictions` instead of proposals and transitions. The summary is
  `terminated`. The participant is `"solver"`, or the `participant` you
  name.
- The timing settings and `threads` apply; the others are for episodes with
  an environment. `lane`, `provenance`, `store` and `on_step` work as for
  `run_episode`.

Presenting an instance in parts is left to downstream code, which knows how
its inputs divide; it can build such records with `RecordBuilder`.

## Running a component in another process

Every component can run in one of three places:

| Where | How | When to use it |
| --- | --- | --- |
| Directly in the event loop | The runners' default, `placement="direct"`. | Fast components, and every `async def` one. |
| In a worker thread | `placement="thread"` on a runner. | A slow `def` component, such as one making a blocking HTTP call. |
| In its own process | Build it with `in_process`. | A CPU-bound component, one that needs its own GPU context, or one that might crash. |

`in_process` builds the component in a long-lived worker process and gives
you a **stand-in** for it until the `with` block ends:

```python
from ssapling.interaction import in_process

with in_process(DigitPadEnv) as env:                # DigitPadEnv() is built in the worker
    first = await runner.reset(env, options=instance.inputs)
    environments.supports(env, "snapshot")          # True: its methods are visible
    env.observation_descriptions                    # read from the worker

with in_process(build_model, "checkpoints/7") as model:
    ...                                             # build_model("checkpoints/7"), in the worker
```

**The stand-in behaves like the component**, so everything that uses a
component (the runners, `supports`, the episode loop) uses it unchanged:

- **Methods.** A public method, looked up through the stand-in, comes back
  as a stand-in method. It is `def` or `async def` exactly as the
  component's method is, and takes the same parameters, so checks such as
  "does `reset` take a seed?" give the same answer. The call runs in the
  worker; the caller waits for the result. Methods a wrapper forwards
  through `__getattr__` work the same way.
- **Attributes.** Any other public attribute is read from the worker each
  time, so one that changes, such as `next_agents`, is never out of date.
  Attributes cannot be set through the stand-in: change them through a
  method.
- **State** stays in the worker between calls, for as long as the block lasts.
- **Errors** the component raises arrive as themselves, with a note saying
  they were raised in the worker. Every SSAPling error survives the journey
  with its attributes.

The stand-in's class is named after the component's, so messages name the
component ("Environment NimEnv does not support 'seeded_reset'"), but it is
not an instance of the component's class.

### Rules

- **Everything crosses by pickling.** The worker is started with `spawn`,
  which works on every platform, so the factory (a class, or a function
  defined at module level), its arguments, and every argument and result of
  a call must be picklable.
- **A factory can prepare the worker.** The worker is a fresh Python
  process, so the project's components may not be registered there yet. A
  module-level factory function can register them, then build:

  ```python
  def build_tutor():
      register_components()
      return registries.teachers.create("stepwise", {"intervene": True})

  with in_process(build_tutor) as tutor:
      ...
  ```
- **A synchronous method blocks the thread that calls it** until the worker
  answers, just as a synchronous component run directly does. To keep the
  event loop running meanwhile, and to allow a timeout, give the component
  `placement="thread"` as well. A timeout on a synchronous component run
  directly is refused, in a process or not.
- **One call at a time.** Calls from several threads are answered one after
  another, in the order they reach the worker.
- **Lifetime.** The worker stops when the block ends, once any call in
  progress has finished, or by force after 5 seconds. The stand-in cannot be
  used after that. Calling the component's own `close`, if it has one, is up
  to its owner, before the block ends.

### Errors

| Error | Means |
| --- | --- |
| Whatever the factory raises | Building the component failed. Raised when the block is entered, with a note. |
| Whatever the component raises | Raised by the call that caused it, as itself. |
| `ComponentProcessError` (a `RuntimeError`) | The worker has exited (it crashed, or the block has ended), or a result or error could not be sent back, for example a function. The worker carries on after the latter. |
| `AttributeError` | The component has no such attribute, or one was set through the stand-in. |
