# Extending SSAPling

Everything a project brings to SSAPling is a **component**: an object with
the right methods, registered by name. Nothing inherits from a SSAPling
class. Each contract is a `typing.Protocol`, checked by its shape, with a
small required part and optional extras. SSAPling never calls a method a
component does not have, and asks before using an optional one.

This page lists what each kind of component needs, with a checklist, and
then how to change SSAPling's own defaults where a project needs something
else. [The usage guide](usage.md) shows the kinds working together, and the
area guides have the details.

| Kind | Registry (`ssapling.registries`) | It needs at least | Details |
| --- | --- | --- | --- |
| Task | `tasks` | `score`, and `generate` or `ids` and `get` | [Tasks](tasks.md) |
| Environment | `environments` | `reset`, `step` | [Environments](environments.md) |
| Teacher | `teachers` | `operations`, `respond` | [Teachers](teachers.md) |
| Model | `models` | `act` | [Models](models.md) |
| Sampler | `samplers` | `draw(seed=...)` | [Data: samplers](data.md#samplers) |
| Collator | `collators` | a call that turns a list of items into a batch | [Data: batching](data.md#batching) |
| Measurement | `measurements` | a function of a `Trial` | [Evaluate: measurements](evaluate.md#measurements) |
| Aggregator | `aggregators` | a function of a list of values | [Evaluate: aggregation](evaluate.md#aggregation) |
| Assessment | `assessments` | `async run(model, *, run_id, seed, model_info=None)` | [Evaluate: assessments](evaluate.md#assessments) |
| Stage | `stages` | `run(context)` | [Train: stages](train.md#stages) |
| Controller | none: given to the `Runner` | `decide(context, last_error)` | [Train: controllers](train.md#controllers) |

## What every kind shares

**Settings.** A component with settings declares them as a frozen dataclass,
names it as its `Settings` attribute, and takes the checked settings as
`config`. Live objects, such as the task a teacher needs, come as keyword
arguments:

```python
from dataclasses import dataclass
from ssapling.core import setting

@dataclass(frozen=True)
class TutorSettings:
    hints: int = setting(3, ge=0, description="How many hints per problem.")

class Tutor:
    Settings = TutorSettings

    def __init__(self, config: TutorSettings, *, task) -> None:
        self.config, self.task = config, task
```

Settings are plain values (numbers, strings, lists, dicts, nested settings),
so they can come from a file and be saved with a run. A rule across several
settings goes in a `validate(self)` method, which runs once every setting
has passed its own checks. See [Core: settings](core.md#settings).

**Registration.** A project registers everything in one function and calls
it at start-up. Registering the same thing twice is harmless; registering
something else under a taken name is refused unless `replace=True`:

```python
from ssapling import registries as r

def register_components() -> None:
    r.teachers.register("tutor", Tutor, description="Gives up to three hints.")
```

**Sync or async.** Any method SSAPling calls may be `def` or `async def`. A
slow synchronous one, such as a large model's forward pass, can run in a
thread so it does not hold up the rest (see
[Interaction: settings](interaction.md#settings)).

**State.** A component whose state should survive a checkpoint has
`export_state()`, returning plain data, and `restore_state(state)`. Without
them it starts afresh when a run resumes, and a warning names it.

## Tasks

A task makes problems and scores answers. It is **generated**,
`generate(*, params, seed)`, or **listed**, `ids()` and `get(id)`, and
always has `score(instance, outcome)`.

- [ ] The same `params` and `seed` give the same instance; the same id gives the same instance.
- [ ] `inputs` holds only what the student may see. The answer goes in `reference`.
- [ ] `equivalence_key` says which instances are the same problem, such as `12 + 34` and `34 + 12`, so they are never split between collections.
- [ ] `descriptors` holds what results are grouped by, such as `{"digits": 3}`.
- [ ] `score` returns `Score.local(value)`, `Score.external(value)` for a score a service reports, or `Score.unavailable(reason)`, never a made-up 0.

Optional: `check_partial(instance, state)` to judge an unfinished answer,
and `teaching_resources()` for what a teacher may use, such as the
procedure as text.

## Environments

An environment is the world agents act in: `reset()` returns a
`ResetResult`, `step(actions)` a `StepResult`. A toy or service with its own
interface gets an **adapter**, which converts its results; see the adapter
checklist in [Environments](environments.md).

- [ ] Everything is keyed by agent id, even with one agent: `{"student": ...}`.
- [ ] `reset` takes the instance's inputs as `options`, and a `seed` if it can reproduce an episode from one.
- [ ] `terminated` (a natural end) and `truncated` (cut off) stay apart. The environment's own flags decide when the episode ends.
- [ ] `rewards=None` when there are none: no rewards is not a reward of zero.
- [ ] A lost connection to a remote environment raises `EnvironmentCommunicationError` and is never retried, because the step may already have happened.

Optional: `agents`, `active_agents` and `next_agents` for several agents or
turns; descriptions of observations and actions; `close()`; and the
capabilities `seeded_reset`, `render`, `state`, `snapshot` and `batched`.

## Teachers

A teacher answers requests: `operations`, the set of names it can be asked,
and `respond(request)`, returning a `TeacherResponse`.

- [ ] Answer with `TeacherResponse.answered(request, contributions)`, or `.abstained(request, reason)`. Contributions are keyed by kind and then by agent: `{"supervision": {"student": "7"}}`.
- [ ] Use the kinds a model or the loop will look for: `supervision`, `feedback`, `intervention` (a replacement action, applied only where the episode settings say `intervene`), or your own.
- [ ] List the context each operation needs in `required_context`, and the task's teaching resources in `required_resources`, so a mistake is caught before the run.
- [ ] Don't catch your own errors to hide them: an error, a wrong response, or no answer in time becomes a `failed` or `cancelled` response, which never changes an action.

Optional: `reset()` for episode state, `close()`, and
`export_state()`/`restore_state(state)`.

## Models

A model acts: `act(observations)` returns an action, or a prediction when
there is no environment.

- [ ] A model that learns lists `learning_operations` and has `learn(operation, **payload)`, returning statistics, which are published.
- [ ] It lists the learning `signals` it takes, such as `{"supervision"}` or `{"reward"}`. Only those reach it; teacher feedback never becomes a reward unless the model asks for feedback and treats it as one.
- [ ] Its parameters or memory survive a checkpoint through `export_state`/`restore_state`. For large weights, save them to a file of your own and return the path: a checkpoint is a document, and nothing in a store is overwritten, so use a new file each time.

Optional: `perceive`, `poll` and `advance` for a model that takes its own
time; a `key=None` parameter for one model serving several participants or
lanes; `reset_episode`; `set_evaluating`; `diagnostics()`; and descriptions
of what it accepts.

## Samplers, measurements, aggregators and the rest

- **A sampler** has `draw(*, seed)`, returning a `Sample`, and usually
  `source`, the task it draws from. The same seed must give the same
  sample. To change over time, it has `set_weights(weights)` and saves its
  weights with `export_state`.
- **A measurement** is a function from a `Trial` (the instance, its record,
  the outcome, the task) to one value. Return `None` or
  `MeasuredValue.unavailable(reason)` when it cannot be computed. A variant
  is a function of its own, under its own name.
- **An aggregator** is a function from the list of contributing values to
  one value. It never sees unavailable or failed values, and is not called
  when there are none.
- **An assessment** of your own has `async run(model, *, run_id, seed,
  model_info=None)` and returns `AssessmentResults`. The built-in
  `standard` assessment covers most needs from its settings.
- **A stage** has `run(context)` and declares what it `requires`, `reads`
  and `publishes`, so a procedure that cannot work is refused before it
  starts. It keeps nothing between runs; what lasts goes in the context.
- **A controller** has `decide(context, last_error)` and returns the next
  entry, or `None` to stop. Every decision is checked like a written
  procedure, and logged.
- **A collator** is built from its settings and then called with a list of
  items, returning a batch, such as padded tensors. To collate in worker
  processes, give it as a registered name, which each worker builds again
  after running your `register_components`.

## Changing SSAPling's defaults

SSAPling has two kinds of rule. Most are **defaults**: the parts a project
plugs in, which it can replace. A few are **guarantees**, which every
project relies on to mean the same thing, and which a project uses rather
than changes:

| Guarantees: the same in every project | Defaults: replace them when you need to |
| --- | --- |
| A record's structure, and that a stored record comes back exactly | Every component above, built-in ones included |
| Nothing in a store is overwritten | Which collections exist, and what they serve or reserve |
| Only an `answered` response changes an action; a `stale` one never does | What the loop does each episode, by writing your own stage |
| The timing modes (turn-based, deadline, real time) | The outcome a task scores, from a record |
| Settings are checked strictly, with no silent conversion | What a measurement or an aggregate means |
| The whole procedure is checked before a run; resuming continues exactly | Which stage runs next, through a controller |

If a guarantee itself is wrong for a project, change SSAPling: a guarantee
loosened downstream would no longer be one.

### Replace a built-in

Built-in components are registered like any other, so a project can put
its own under the same name. Every run file that names it then gets the
project's version:

```python
r.stages.register("interact", BatchedInteract, replace=True)
r.samplers.register("generated", MixingSampler, replace=True)
```

Giving yours a new name is usually clearer. Keep `replace=True` for when
existing run files must pick up the change.

### Write your own stage

The built-in `interact` stage runs one episode at a time, always with the
model acting. A project that needs something else, such as a teacher that
demonstrates whole solutions while the student only watches, or many
episodes at once for a GPU, writes its own stage. To keep the guarantee
that training never sees a reserved problem, draw from what the
collections leave, as `interact` does:

```python
class Demonstrate:
    Settings = DemonstrateSettings
    requires = ("model", "sampler", "collections", "teacher")

    def __init__(self, config) -> None:
        self.config = config

    async def run(self, context) -> None:
        collections, sampler = context.components["collections"], context.components["sampler"]
        for _ in range(self.config.episodes):
            number = context.counters["episodes"]
            drawn = next(collections.remainder(sampler=sampler, seed=context.seed(f"draw-{number}")))
            ...                                    # the teacher's demonstration of drawn.instance
            context.counters["episodes"] += 1
```

Name each instance's seed from the counters, as above, and the run stays
reproducible and resumable.

To learn from what a stage gathered, call `ssapling.train.learn(context,
operation, **payload)`, as the built-in stages do: it calls the model's
`learn` (sync or `async def`), counts the update, publishes the statistics
under the operation's name, and returns them:

```python
stats = await learn(context, "learn_from_batch", batch=batch)
```

To call any other component method that may be sync or `async def`, use
`await Execution().call(method, *args)` from `ssapling.core`.

### Change what collections serve and reserve

A collection reserves its problems, so training never draws them, and holds
each problem once. A project that needs something else, such as a fixed
set of elementary problems that is both monitored and trained on, can
subclass `Collections`. The stages only ever ask a collections object for
a collection, with `get`, and for training draws, with `remainder`:

```python
from ssapling.data import Collections

class WithMemorisation(Collections):
    """Serves "memorisation" for assessment, without reserving it from training."""

    def __init__(self, source, *, memorisation) -> None:
        super().__init__(source)
        self.memorisation = tuple(memorisation)    # Drawn items; a problem may appear twice

    def get(self, name, *, purpose=None):
        if name == "memorisation":
            return self.memorisation
        return super().get(name, purpose=purpose)
```

Give it to the run as a component, `Runner(..., components={"collections":
WithMemorisation(task, memorisation=items)})`. SSAPling's checks no longer
cover what the subclass serves itself: here, that no other collection holds
a memorisation problem. Say why that holds where you build it.

### Score what a record shows

An assessment scores the **outcome** of each attempt. By default that is
the last prediction, or the final observation. When the answer has to be
read from the record, give your own function, as the example does for the
digit pad:

```python
def entered_number(record):
    entered = record.steps[-1].transition.observations["student"]["entered"]
    return int(entered[::-1])

assessments.create("standard", settings, collections=collections, task=task,
                   env=pad, outcome=entered_number)
```

In a run file, give the function to the `Runner` as a component and pass
it with `uses: {outcome: <its role>}`.

### Summaries the built-ins don't give

An aggregator of your own covers any summary of the values themselves. A
summary across groups needs no aggregator: with `group_by`, the
`Summary`'s `groups` hold each group's value, and `weighting: equal` makes
the value the mean of those groups' values. `where` restricts any
aggregation to the instances with given descriptor values.
