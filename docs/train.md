# Train

`ssapling.train` composes training runs from interchangeable parts. A run
does **stages**, such as interacting, learning, assessing, or a project's
own, in the order its **procedure** says, until its **stop condition**
holds. Stages never talk to each other directly. They read from, and
publish to, one shared **run context**. SSAPling provides the machinery and
a few generic stages built only on the other areas' contracts. Learning
methods and domain-specific stages are downstream.

| Part | What it is |
| --- | --- |
| [Procedures](#procedures) | A list of stages, each with its settings and an optional trigger, and a stop condition. Python or YAML. |
| [Triggers and conditions](#triggers-and-conditions) | When an entry runs: once, every N of a counter, until or when a condition holds. |
| [Stages](#stages) | The contract a stage implements, and how to write your own. |
| [Built-in stages](#built-in-stages) | `interact`, `learn_from_data`, `learn_operation`, `update_sampler`, `assess`, `checkpoint`. |
| [The run context](#the-run-context) | What the stages share: components, counters, seeds, published values. |
| [Runs](#runs) | `Runner(...)`, `Runner.from_config(...)`, `Runner.resume(...)`, then `await runner.run()`. What is checked before the run. |
| [Run files](#run-files) | One file for the whole run: its components, procedure and stop condition. |
| [The output](#the-output) | What a run writes to its store, so it can be inspected and rebuilt. |
| [Checkpoints and resuming](#checkpoints-and-resuming) | Saving a run between stages, and continuing it from the same position. |
| [Controllers](#controllers) | What decides the next stage: the built-in `FixedController` follows the list; your own can decide as the run goes. Every decision is checked, logged, and can be replayed. |

## A worked example

A run on additions entered on a digit pad. It learns online, consolidates
every 500 episodes, assesses every 2000 updates, moves on to longer
problems once its accuracy reaches 0.9, checkpoints every 10000 updates,
and stops at 100000 episodes or once its accuracy reaches 0.98, whichever
comes first:

```yaml
# run.yaml
components:
  task: {kind: tasks, name: addition, settings: {max_digits: 3}}
  every_digit:
    kind: samplers
    name: generated
    settings: {vary: digits, values: [1, 2, 3]}
    uses: {source: task}
  sampler:                                  # what training draws: short problems first
    kind: samplers
    name: generated
    settings: {vary: digits, values: [1, 2, 3], weights: {"1": 1}}
    uses: {source: task}
  collections:
    kind: collections
    settings:
      source: task
      build: {validation: {sampler: every_digit, size: 1000, seed: 1}}
  environment: {kind: environments, name: digit_pad}
  model: {kind: models, name: my_net, settings: {width: 256}}
  monitoring:
    kind: assessments
    name: standard
    settings:
      collections: [validation]
      aggregations: {accuracy: {measurement: score, aggregator: mean}}
    uses: {collections: collections, task: task}
procedure:
  - stage: interact
    settings: {episodes: 32, learn: {after: step}}
  - stage: learn_operation
    every: {episodes: 500}
    settings: {operation: consolidate, payload: {limits: {seconds: 600}}}
  - stage: assess
    every: {updates: 2000}
    settings: {assessment: monitoring}
  - stage: update_sampler
    when: {value: monitoring.accuracy, at_least: 0.9}
    settings: {weights: {"1": 1, "2": 1, "3": 1}}
  - stage: checkpoint
    every: {updates: 10000}
stop: {episodes: 100000, value: monitoring.accuracy, at_least: 0.98}
```

```python
from ssapling.train import Runner

register_components()      # the project's own: addition, digit_pad, my_net
runner = Runner.from_config("run.yaml", seed=7, output="runs/baseline")
context = await runner.run()
context.counters           # {"episodes": 41216, "steps": ..., "updates": ..., "stages": ...}
context.read("monitoring.accuracy")

context = await Runner.resume("runs/baseline").run()   # after a crash: from its last checkpoint
```

`run` is `async`; from synchronous code, call it with
`asyncio.run(runner.run())`.

## Procedures

A **procedure** is a list of entries and a stop condition. Each entry names
a registered stage, gives its settings, and may have a trigger:

```yaml
procedure:
  - stage: interact                 # the registered stage
    settings: {episodes: 32}        # its settings, checked against its settings class
  - stage: learn_operation
    every: {episodes: 500}          # an optional trigger
    settings: {operation: consolidate}
stop: {episodes: 100000}
```

The same procedure in Python, as a mapping of the same values or with the
classes:

```python
from ssapling.train import Condition, Entry, Every, Procedure

procedure = Procedure(
    [
        Entry("interact", {"episodes": 32}),
        Entry("learn_operation", {"operation": "consolidate"}, every=Every(episodes=500)),
    ],
    stop=Condition(episodes=100000),
)
```

`Procedure`, `Entry`, `Every` and `Condition` are settings classes (see
[Core: settings](core.md#settings)), so a procedure from a file and one
from Python are checked the same way, and every problem is reported with
its path, such as `procedure[1].every.episodes: must be >= 1, got 0`.

The run goes through the list again and again (each time through is a
**pass**), running each entry whose trigger fires. The stop condition is
checked before every stage, so a run stops as soon as it holds, even in the
middle of a pass.

## Triggers and conditions

An entry has at most one trigger:

| Trigger | The entry runs... |
| --- | --- |
| none | each time the list reaches it. |
| `once: true` | the first time the list reaches it, only. |
| `every: {episodes: 500}` | each time the counter has grown by 500 since the entry last ran, counting from the start of the run the first time. |
| `until: <condition>` | each time the list reaches it, until the condition holds. |
| `when: <condition>` | once: the first time the list reaches it while the condition holds. For a curriculum's next step, say. |

A **condition** names counters, a published value, or both, and holds when
any of them is reached:

```yaml
{episodes: 100000}                                      # a counter has reached this
{value: monitoring.accuracy, at_least: 0.95}            # a published value, by its path
{value: consolidate.loss, at_most: 0.1}
{episodes: 100000, value: monitoring.accuracy, at_least: 0.95}   # whichever comes first
```

- The counters are `episodes`, `steps`, `updates` and `stages` (see [the
  run context](#the-run-context)), and `seconds`: how long the run has
  been running. Time is not reproducible, so a run that uses it can differ
  from one run to the next.
- `value` is a published value, by its name or a dotted path into it.
  `at_least` and `at_most` bound it; with both, it must be within both.
- **A value not yet published leaves the condition unmet**, and so does an
  undefined one (`None`), such as an aggregate over no values. A path
  that leads nowhere in a published value, or a value that is not a
  number, is an error, so a misspelt name does not leave a run going
  forever.
- `every` names exactly one counter.

**The stop condition is also the run's budget.** It is checked before
every stage, whatever the controller decides, so `stop: {stages: 200,
seconds: 3600}` limits a run however it is controlled.

## Stages

A **stage** does one kind of work. It is a registered component: anything
with `run(context)`, written as `def` or `async def`, registered in
`ssapling.train.stages` (also `ssapling.registries.stages`). Built-in stages
and your own are treated exactly alike:

```python
from dataclasses import dataclass
from ssapling.core import setting
from ssapling.train import stages

@dataclass(frozen=True)
class ReplayAuditSettings:
    sample: int = setting(100, ge=1, description="Records to audit.")

class ReplayAudit:
    """Checks a sample of the stored records and publishes what it found."""

    Settings = ReplayAuditSettings
    requires = ("model", "store")
    publishes = ("audit",)

    def __init__(self, config):
        self.config = config

    async def run(self, context):
        store = context.components["store"]
        ...
        context.publish("audit", {"bad": 3, "checked": self.config.sample})

stages.register("replay_audit", ReplayAudit)
```

A stage may say what it needs, so a procedure that cannot work is refused
before the run starts:

| Declaration | Meaning |
| --- | --- |
| `requires` | The components it uses, by role, such as `("model", "store")`. |
| `reads` | The published values it reads, by name. |
| `publishes` | The values it publishes, by name. |
| `check(context)` | Anything else, such as a learning operation the model must support. It raises `ValueError` saying what is wrong. |

They are read from the stage once it is built, so they may depend on its
settings: a stage that runs a named assessment can publish under that
name. A stage is built each time it runs, and once before the run to check
it, so building one should do nothing but keep its settings. It keeps
nothing between runs: what lasts belongs in the context.

## Built-in stages

SSAPling's stages are built only on the other areas' contracts. Each finds
its components by **role**: by default `model`, `environment`, `sampler`
and `collections`, and a setting names any other role, such as the
assessment to run or the teacher to consult. They have no special status:
a project's own stage plugs in the same way.

| Stage | Work | Requires | Publishes |
| --- | --- | --- | --- |
| [`interact`](#interact) | Runs episodes of the model on training instances, optionally with a teacher, and optionally learning as they happen. | `model`, the sampler, and the environment, collections, teacher and its context, unless set to `null` | The learning operation, if it learns. |
| [`learn_from_data`](#learn_from_data) | Draws batches from a store or a collection of items and learns from each. | `model`, the source, the collator if set | The learning operation. |
| [`learn_operation`](#learn_operation) | Calls one learning operation, such as a night-cycle `consolidate`. | `model` | The learning operation. |
| [`update_sampler`](#update_sampler) | Gives the sampler new weights, for a curriculum. | the sampler | nothing |
| [`assess`](#assess) | Runs an assessment, saves its results and publishes its aggregates. | `model`, the assessment | The assessment's role. |
| [`checkpoint`](#checkpoints-and-resuming) | Saves the run, so it can be resumed from here. | `store` | nothing |

**Learning.** Every call to a learning operation counts one update, and
what the model's `learn` returns is published under the operation's name,
so a condition can read it: `{value: consolidate.loss, at_most: 0.1}`.
Before the run, each stage checks that the model supports the operations it
will call (see [Models: learning](models.md#learning)).

### `interact`

Runs episodes through the [episode loop](interaction.md#the-episode-loop),
each on a new instance.

| Setting | Meaning | Default |
| --- | --- | --- |
| `episodes` | How many episodes. | 1 |
| `environment` | The environment's role. `null`: [static solving](interaction.md#static-solving), with no environment. | `"environment"` |
| `teacher` | The role of a teacher to consult at each step. | `null`: no teacher |
| `teacher_context` | The role of a function that builds each request's context, as for `run_episode`. | `null`: the default context |
| `sampler` | The role of the sampler that picks each instance. | `"sampler"` |
| `collections` | The role of the run's collections. | `"collections"`; `null` if it has none |
| `episode` | How each episode runs: the [episode loop's settings](interaction.md#settings), such as `timing`, `max_steps` and `intervene`. | the loop's defaults |
| `learn` | Learn as the episodes happen: `{operation, after}`. | `null`: no learning |

- **Training never sees a reserved problem.** Each instance is drawn from
  the remainder of the run's collections, so no problem of any collection,
  held out or not, is trained on (see [Data: collections](data.md#collections)).
- **Seeds follow the episode's number.** Episode *n*'s instance is drawn
  with the seed `draw-<n>`, and the environment, if it has seeded reset,
  is reset with `episode-<n>`. So episode 41 gets the same instance however
  the episodes were split between stages, and in a resumed run.
- **Records** go to the run's store, if it has one (the component
  `store`), as `"episode-<n>"`.
- It counts `episodes`, and `steps`: the steps of its records, which in
  static solving are attempts.

**Learning online.** `learn: {operation: learn_from_experience, after:
step}` (the defaults) calls the operation after each step, once it is
recorded, before the next:

```python
model.learn("learn_from_experience", step=step, signals={"supervision": {"student": "7"}})
```

`signals` holds only the [learning signals](models.md#learning-signals) the
model declares, out of the step's: `reward` (the environment's rewards) and
each kind of contribution in the teacher's answers (`supervision`,
`feedback`, ...), each keyed by agent. `after: episode` calls it once per
episode, with the record: `learn(operation, record=record)`. The record
holds everything; using any of it is the model's choice, as in
[Evaluate](evaluate.md#assessments).

### `learn_from_data`

Draws batches of items and calls a learning operation with each.

| Setting | Meaning | Default |
| --- | --- | --- |
| `source` | The role of what the items come from. A store: its finished records, except attempts at instances of a collection, such as an assessment's, which training never learns from. Anything else: its items, such as a list of records or a collection's instances. | required |
| `operation` | The learning operation called with each batch. | `"learn_from_batch"` |
| `batches` | How many batches: one update each. | 1 |
| `batch_size` | Items per batch. | 32 |
| `collator` | The role of a [collator](data.md#batching), which turns a list of items into a batch. | `null`: the list itself |
| `workers` | Worker processes that collate batches. | 0: collate here |

Items are drawn at random, with replacement, with the seed
`batches-<updates>`, and each batch is passed as
`learn(operation, batch=batch)`. The run's own records are the store
`store`: `{source: store}`.

### `learn_operation`

Calls one learning operation once: `learn(operation, **payload)`.

| Setting | Meaning | Default |
| --- | --- | --- |
| `operation` | The operation, such as `"consolidate"`. | required |
| `payload` | Its arguments, as plain values, such as `{limits: {seconds: 600}}`. | none |

### `update_sampler`

Calls the sampler's `set_weights(weights)`, which the built-in `stratified`
and `generated` samplers have (see [Data: samplers](data.md#samplers)).

| Setting | Meaning | Default |
| --- | --- | --- |
| `weights` | The new weights, keyed by value or stratum as a string. | required |
| `sampler` | The sampler's role. | `"sampler"` |

A curriculum moves on with a `when` trigger:

```yaml
- stage: update_sampler
  settings: {weights: {"1": 1, "2": 3}}
  when: {value: monitoring.accuracy, at_least: 0.9}
```

### `assess`

Runs an [assessment](evaluate.md#assessments) of the model.

| Setting | Meaning | Default |
| --- | --- | --- |
| `assessment` | The role of the assessment component. Its aggregates are published under this name. | required |

- Its run id is `"<role>-<stages>"`, such as `"monitoring-12"`, and its
  seed is `assessment-<role>`: the same each time, so the assessments
  during a run are of the same attempts, and comparable.
- The results' model information (`provenance["model"]`) is the run's
  counters, such as `{"episodes": 4096, "steps": ..., "updates": 128,
  "stages": 12}`: how much training the model had had.
- The results are saved in the run's store, if it has one, as
  `"results/<run id>"`, and the aggregates' values are published under the
  role: `{"accuracy": 0.83, "by_digits": 0.8}`. The assessment's own
  records go to the store it was given, if any.
- Given the same model state, instances and seed, the results are the same
  as the assessment run standalone.

## The run context

The run context is what the stages of a run share. A stage never hands
anything to another stage directly; it reads from the context and
publishes values to it.

| Part | What it is |
| --- | --- |
| `context.components` | The run's components, keyed by role: `"model"`, `"environment"`, and whatever else the run was given. The role `"store"` is the run's store, where records and results go. |
| `context.counters` | `episodes`, `steps`, `updates` and `stages`, from 0. The stages that do the work count it: the interact stage counts episodes and steps, a learning stage counts updates, and the runner counts the stages it starts. |
| `context.seed(name)` | A seed derived from the run's root seed. |
| `context.publish(name, value)` and `context.read(path)` | The table of published values. |
| `context.seconds` | How long the run has been running. |
| `context.decisions` | Every decision so far, with what the controller saw (see [the decision log](#the-decision-log)). |

**Published values** are plain data (`None`, booleans, numbers, strings,
and lists and string-keyed dicts of these), keyed by name, so they can be
saved with a checkpoint and shown to a controller. Each is stamped with the
counters at which it was published, and their meaning belongs to the
stages that publish and read them:

```python
context.publish("monitoring", {"accuracy": 0.83, "by_digits": 0.8})
context.read("monitoring.accuracy")    # 0.83
context.values["monitoring"]           # Published(value={...}, counters={"episodes": 4096, ...})
```

**Seeds.** `context.seed(name)` derives a seed from the root seed and a
name by a fixed rule, `random.Random(f"{root seed}:{name}").getrandbits(32)`:
the same root seed and name give the same seed on every machine, and
different names give unrelated seeds. The built-in stages use the names
`draw-<n>` and `episode-<n>` for episode *n*, `batches-<updates>`, and
`assessment-<role>`. The rule is public as `derive_seed(root, name)`
(`from ssapling.train import derive_seed`), so a project can derive its own
named seeds from a run's root seed the same way, outside a run.

## Runs

```python
runner = Runner(procedure, components, seed=7, output="runs/baseline")
runner = Runner.from_config("run.yaml", seed=7, output="runs/baseline")
runner = Runner.resume("runs/baseline")
context = await runner.run()
```

| Argument | Meaning |
| --- | --- |
| `procedure` | A `Procedure`, or a mapping of its settings, such as `load_config("procedure.yaml")`. It may list the components to build: see [run files](#run-files). |
| `components` | Components given as they are, keyed by role: a model built in Python, say, or a function such as an assessment's `outcome`. |
| `seed` | The root seed. Every seed in the run is derived from it. |
| `output` | Where the run writes: a directory, for a [`RecordStore`](data.md#storage), or any `Store`. It is the component `store`. Optional, but checkpoints and the run's saved settings need it. |
| `controller` | What decides the next stage. By default the [fixed controller](#controllers). |

When the runner is made, the procedure's settings and its components'
settings are checked, and every problem is reported together. Then `run`:

1. **Builds** the procedure's components, in order.
2. **Checks** every entry of the procedure, before anything runs, and
   reports every problem together in one `ConfigError`:

   ```text
   ConfigError: Invalid procedure:
     - procedure[0].stage: Unknown stage 'interac'. Did you mean 'interact'? ...
     - procedure[1].settings.batches: must be >= 1, got 0
     - procedure[2]: stage 'interact' requires 'teacher', which the run does not have (its components: 'model', 'environment', 'sampler', 'collections')
     - procedure[3].until.value: reads 'monitoring', which no stage in the procedure publishes
   ```

   So an unknown stage, invalid settings, a missing component, a failed
   `check`, and a value read (by a stage or a condition) that no stage in
   the procedure publishes are all found before training starts. A
   condition on an assessment the procedure never runs is the last of
   these.
3. **Saves the run's settings** to its store, if it has one (see [the
   output](#the-output)), or, when resumed, restores its last checkpoint.
4. **Runs:** asks the controller for the next entry, checks it the same
   way, counts it in `stages`, and runs it, until the stop condition holds
   or the controller stops. A stage's own error ends the run, as itself.
5. **Closes** the components it built (their `close()`, if they have one),
   in reverse order. Components given as they are are left to whoever made
   them.

SSAPling logs what the run does through Python's standard `logging`, under
the name `ssapling.train`, and installs no handlers.

## Run files

A procedure can also list the components the run builds, so that one file
describes the whole run: `components`, `procedure` and `stop`. Each
component has a role, its key, and they are built in order:

```yaml
components:
  task: {kind: tasks, name: addition, settings: {max_digits: 3}}
  uniform: {kind: samplers, name: uniform, uses: {source: task}}
  collections:
    kind: collections
    settings:
      source: task
      build:
        validation: {sampler: uniform, size: 1000, seed: 1}
        test: {sampler: uniform, size: 1000, seed: 2, held_out: true}
  monitoring:
    kind: assessments
    name: standard
    settings: {collections: [validation]}
    uses: {collections: collections, task: task, store: store}
```

| Key | Meaning |
| --- | --- |
| `kind` | Where it comes from: `tasks`, `teachers`, `models`, `environments`, `samplers`, `collators` or `assessments`, the registries of the same names in `ssapling.registries`; or `collections`. |
| `name`, `settings` | The registered component and its settings, as for `registry.create`. |
| `uses` | What it is built with: for each parameter, the role of a component built before it. |

- **Only what `uses` names is passed.** Nothing is wired by its name
  alone, so an assessment, say, gets no teacher unless `uses` gives it one.
  `uses` can name components given as they are, which come first, and the
  run's store, `store`.
- **Seeds.** A component whose constructor takes `seed` is given one,
  derived from the root seed as `component-<role>`, and saved with the
  run.
- **Collections** are drawn from a source: `settings: {source, build}`.
  `source` is a role, such as the task, and each entry of `build` is a
  collection, built in order, with the keys of `Collections.build`:
  `sampler` (a role), `size`, `seed`, `held_out`, `where` and
  `max_draws`. The source's registered name goes into each item's
  provenance, so items can be made again through the registry.

What a file holds beyond one run, such as sweeps, presets, includes and
command-line overrides, is the project's own. Every problem in a run file
is reported together, with the file:

```text
ConfigError: Invalid run file run.yaml:
  - components.task.name: Unknown task 'additon'. Did you mean 'addition'? ...
  - components.sampler.uses.source: 'tsk' is not a component built before it
```

## The output

A run writes to its store:

| What | Where |
| --- | --- |
| Its settings, root seed, component seeds, the roles it was given as they are, and its controller's class, if it was given one | the document `run` |
| Training records | `episode-<n>` |
| Assessment results | the document `results/<role>-<stages>`; the assessment's records go to the store it was given, if any |
| Checkpoints | the document `checkpoints/<stage>` |
| The decision log (see [controllers](#the-decision-log)) | the document `decisions`, when the run ends |
| Each resumption | the document `resumes/<k>`, naming the checkpoint it continued from |

The document `run` is enough to rebuild the run: `Runner.resume` does, and
`Runner(saved["procedure"], seed=saved["seed"], output=...)` runs it again
from the start. A second run into the same output is refused.

## Checkpoints and resuming

The **`checkpoint`** stage saves the run, as the document
`checkpoints/<stage>`: the state of every component with `export_state()`,
the controller's, and the run context's (its counters, running time,
published values and decision log). What each component saves is its own
business; the store's codec writes it. Checkpoints are taken between stages, never in the
middle of an episode:

```yaml
- stage: checkpoint
  every: {updates: 10000}
```

**`Runner.resume(output)`** continues a run from its latest checkpoint:

- The run is rebuilt from its saved settings. Components given as they
  are, and a controller of its own, are given again:
  `Runner.resume(output, components={"model": model}, controller=...)`;
  an error names any that are missing.
- Each component is given back its state (`restore_state`), and so are the
  controller and the context. The run continues from the same position,
  with the same episodes, instances and seeds, so it ends as if it had
  never stopped.
- A component that does not export its state starts as built, and is named
  in a warning logged when the run resumes.
- Without a checkpoint, the run starts again from the beginning.

**Nothing is overwritten.** A run that failed wrote on after its last
checkpoint, and that stays as a record of what happened. So each
resumption is a new session, `k`, recorded as the document `resumes/<k>`,
and everything it writes is named with `context.output_name(name)`, which
adds `resume-<k>/`: `resume-1/episode-41`, `resume-1/checkpoints/120`, and
assessment results as `results/resume-1/<role>-<stages>`. A stage of your
own that writes to the store names what it writes the same way.

## Controllers

A **controller** decides what the run does next: which stage to run, with
which settings, or to stop. The built-in **`FixedController`** follows the
procedure's list, as described above. It is the default.

If a whole pass through the list finds nothing to run, and the stop
condition does not hold, the fixed controller raises `ProcedureStuckError`
rather than loop forever: for example, every entry is `once`, and the stop
condition asks for more episodes than they run.

### Deciding as the run goes

Any object with `decide(context, last_error)` can be a controller, and
decide while the run is going: for example a language model choosing to
interact, then to learn with a particular operation, then to assess, then
to change the curriculum.

```python
class Tutor:
    """Interacts, then assesses, in turn, until the assessment is perfect."""

    def __init__(self):
        self.turns = 0

    def decide(self, context, last_error):
        if "monitoring" in context.values and context.read("monitoring.accuracy") == 1.0:
            return None                                          # stop
        self.turns += 1
        if self.turns % 2:
            return {"stage": "interact", "settings": {"episodes": 32}}
        return {"stage": "assess", "settings": {"assessment": "monitoring"}}

runner = Runner({"procedure": [], "stop": {"stages": 200}}, components, seed=7,
                controller=Tutor())
```

- **What it returns.** An `Entry`, or a mapping of one such as
  `{"stage": "interact", "settings": {"episodes": 8}}`, or `None` to stop.
  `decide` may be `async def`, as a call to a language model would be.
  Settings name components by role, so a decision can pick the teacher, the
  learning operation or the sampler, or run without a teacher.
- **What it sees.** The run context: `context.export_state()` gives the
  counters, running time and published values as plain data, ready to show
  a language model. The stages it can choose from are
  `stages.catalogue()`: each registered stage's name, description, and
  settings as JSON Schema; `settings_schema(Entry)` describes a decision.
  They can be offered to a language model as tool descriptions, through
  MCP for example; the model client and any MCP server are the project's
  own.
- **Every decision is checked** exactly like an entry of a procedure, and a
  value it reads must have been published already. A decision that cannot
  run does not run: the error goes back to the controller as `last_error`
  on its next call, so it can choose again. Three refused decisions in a
  row end the run with the last error.
- **The stop condition is the budget.** It is checked before every stage,
  whatever the controller decides, so `stop: {stages: 200, seconds: 3600}`
  limits a run however it is controlled.
- **Entries the controller offers are checked first.** A controller that
  knows every entry it can decide offers them as `procedure()`. Given one,
  the procedure may leave out its list (`{"stop": {...}}`): the runner takes
  the controller's entries and checks them all before the run, as it checks
  a procedure's own list.

### The decision log

Every decision is recorded, in order, in `context.decisions`: the stage and
settings chosen, what the controller saw when it chose (the counters,
running time and published values), and, for a refused one, why:

```python
context.decisions[2]
# {"stage": "interact", "settings": {"episodes": 32},
#  "seen": {"counters": {...}, "seconds": 41.2, "values": {"monitoring": {...}}},
#  "refused": None}
```

The log is saved with each checkpoint, and at the end of the run, also when
it fails, as the store document `decisions`. When an error ended the run, the
document also holds it, as `"error": {"type": "RuntimeError", "message":
"..."}`, so a stopped run's reason stays with its decisions. So a run can be inspected and
**replayed exactly** from its decisions, even when the controller was a
language model that would choose differently another time:

```python
from ssapling.train import ReplayController

decisions = RecordStore("runs/llm").load_document("decisions")["decisions"]
runner = Runner(procedure, components, seed=7, controller=ReplayController(decisions))
```

`ReplayController` makes the logged decisions again, leaving out the
refused ones, and stops after the last. Given the same seed and
components, the replayed run does the same.

**A controller's state** is saved with each checkpoint, like a component's,
if it has `export_state()` and `restore_state(state)`, as the fixed and
replay controllers do. A run with a controller of its own is resumed with
that controller given again: `Runner.resume(output, controller=...)`.

## Errors

| Error | Means |
| --- | --- |
| `ConfigError` | Invalid procedure settings (raised when the runner is made), a procedure that cannot run (raised by `run` before anything runs, listing every problem), or a third refused decision in a row. |
| `ProcedureStuckError` | Nothing in a fixed procedure can run, and the stop condition does not hold. `.counters` says where the run got to. |
| `LookupError` from `context.read`, or a condition | Nothing is published under the name, or the path leads nowhere in the value. |
| `TypeError` from `context.publish` | Part of the value is not plain data. |
| `TypeError` from a condition | The value it bounds is not a number. |
| `ValueError` from `Runner` | A component given as `"store"` (the store is the `output`); or, from `resume`, a component or controller the run was given and is not given again. |
| `ValueError` from `run` | The output holds a run already: resume it, or give a new output. |
| `UnknownDocumentError` from `resume` | No run was written there. |
| `TypeError` from `run`, when resuming | A component saved its state but has no `restore_state`. |
