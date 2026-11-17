# SSAPling overview

SSAPling is a pure-Python library of **contracts and machinery** for
System–Specification Alignment Problem projects, such as one that teaches
arithmetic or one on ARC. It defines how environments, tasks, teachers, and
models plug together, and it runs the parts that are the same in every
project: building components from settings, running episodes, storing records,
training procedures, and assessments.

It contains **no concrete** environments, tasks, teachers, models, or learning
methods. A downstream project writes those and registers them; everything else
then works.

## Package areas

| Area | Responsibility |
| --- | --- |
| `core/` | Registration, configuration (Python, YAML or JSON), construction, extension. |
| `tasks/` | Problems: instances, references, scoring, teaching resources. |
| `teachers/` | Requests to, and responses from, user-defined teachers. |
| `models/` | How a model acts, keeps state, and learns. |
| `environments/` | Agents, actions, observations, transitions, turn order. |
| `data/` | Sources, samplers, collections, records, storage, batching. |
| `interaction/` | The loop that runs episodes (and static solving). |
| `evaluate/` | Assessments, measurements, aggregation. |
| `train/` | Training procedures built from stages. |

Each area's `__init__.py` docstring states what it is, and is not, responsible
for.

## Import rules

An area may import only the areas its arrows come from, and the last three
list theirs in brackets. That prevents circular imports and keeps each area
usable on its own:

```text
core ─┬─> tasks ─────────────┐
      ├─> teachers ──(types)─┼─> environments ──(types)─┐
      ├─> models             │                          │
      └──────────────────────┴──> data <────────────────┘
                     interaction  (core, tasks, environments, teachers, models, data)
                     evaluate     (everything except train)
                     train        (everything)
```

`(types)` means only the result-type module of that area. For example,
`environments` may use a teacher's *response type*, but not the teacher runner.

`tests/test_layering.py` enforces these rules. It reads every module's import
statements (without running them) and fails, naming the module and line, if
an area imports something it may not.

## Packaging commitments

- Python 3.12 and newer; Linux, macOS, and Windows.
- **No required third-party dependencies.** `pip install ssapling` pulls in
  nothing else. `pip install "ssapling[yaml]"` adds YAML configuration files.
- Ships type information (`py.typed`).
- Public interface: names exported from `ssapling`, from
  `ssapling.registries`, and from each area's `__init__`. Anything else,
  including names starting with `_`, is internal.
- Semantic versioning. Before 1.0, a minor release may change the public
  interface, and every such change is listed in `CHANGELOG.md`.

`tests/packaging/` checks the second and third: the declared dependencies are
empty, importing every area loads only standard-library modules, and
`py.typed` ships. CI checks the first, testing Python 3.12 to 3.14 on all
three platforms.

## Where to start

1. [Using SSAPling](usage.md): a whole project, from components to an
   assessed run, following [`examples/arithmetic/`](../examples/arithmetic/).
2. [Extending SSAPling](extending.md): what each kind of component needs,
   and how to change SSAPling's defaults.
3. The area guides, in the order the areas build on each other:
   [Core](core.md), [Tasks](tasks.md), [Teachers](teachers.md),
   [Models](models.md), [Environments](environments.md), [Data](data.md),
   [Interaction](interaction.md), [Evaluate](evaluate.md), [Train](train.md).
4. [Development](development.md), to work on SSAPling itself.

## Glossary

| Term | Meaning |
| --- | --- |
| Aggregate | One value summarising a measurement over many instances, such as mean accuracy. Unavailable and failed values are left out and counted, never taken as zero. See [Evaluate](evaluate.md#aggregation). |
| Assessment | Runs a model over collections, measures each attempt, and aggregates. See [Evaluate](evaluate.md#assessments). |
| Async | A method written `async def`, which can pause while it waits, for a network reply say, so other work goes on. SSAPling calls sync and async methods alike. See [Core](core.md#execution-helper). |
| Capability, feature | An optional extra a component offers just by having its methods, such as an environment's `render`. A component can switch one off, never claim one it lacks. |
| Checkpoint | Everything a run needs to continue from where it was: every component's state, the counters, published values and decisions. See [Train](train.md#checkpoints-and-resuming). |
| Collection | A named set of problems drawn from a source, which training never draws from. A held-out collection is served only to assessments. See [Data](data.md#collections). |
| Condition | When a trigger fires or a run stops: a counter reached, or a published value within bounds. See [Train](train.md#triggers-and-conditions). |
| Contribution | What a teacher's answer offers, by kind and then by agent: `{"supervision": {"student": "7"}}`. The kinds are open; `intervention` replaces an action, where the episode settings allow it. See [Teachers](teachers.md). |
| Controller | Decides which stage a run does next. The default follows the procedure's list. See [Train](train.md#controllers). |
| Episode | One attempt at one instance in an environment, from reset to its end. |
| Equivalence key, problem key | Which instances are the same problem, such as `12 + 34` and `34 + 12`. Collections never share a problem key. See [Tasks](tasks.md). |
| Given as it is | A component a run is handed already built, by role, such as a model made in Python or an assessment's `outcome` function, rather than built from the run file. The run file can `use` it; the run never closes it; and `Runner.resume` must be given it again. See [Train](train.md#runs). |
| Held out | A collection served only for evaluation. |
| Instance | One problem: the `inputs` the student sees, the `reference` answer, and `descriptors`. See [Tasks](tasks.md). |
| Lane | One of several copies of an environment stepped together, so one model can act in many episodes at once. |
| Learning operation | A named way a model learns, such as `learn_from_batch` or `consolidate`, called through `learn`. See [Models](models.md#learning). |
| Learning signal | What a model may learn from after a step: the environment's `reward`, or a kind of teacher contribution. A model receives only the signals it declares. See [Models](models.md#learning-signals). |
| `NOTHING` | What a model's `poll` returns when it has no output yet. It is not `None`, which can be a real output. |
| Outcome | What a task scores: by default the last prediction or the final observation; a project can read it from the record instead. |
| Participant | An agent in an episode, played by a model, a teacher, or the environment's own logic. |
| Pickling | Python's way of turning an object into bytes, to send it to another process or save it. SSAPling's values and errors survive it, so they arrive from a worker process as themselves. |
| Placement | Where a component's synchronous calls run: `direct`, in the event loop's thread, or in a `thread` of its own, so the loop keeps going and a timeout can stop waiting. A component can also run in a separate process. See [Interaction](interaction.md#running-a-component-in-another-process). |
| Procedure | The list of stages a run goes through, with their triggers, and the stop condition. See [Train](train.md#procedures). |
| Provenance | How an instance was made, so it can be made again: its source, seed or position, and sampler. See [Data](data.md). |
| Record | Everything about one attempt: a header, each step (proposals, executed actions, teacher responses, observations, timing), and a summary of how it ended. See [Data](data.md#records). |
| Registry | Maps names to implementations of one kind of component. `ssapling.registries` holds them all. See [Core](core.md#registry). |
| Remainder | The problems no collection holds. Training draws from it. |
| Role | A component's name within a run, such as `model` or `tutor`. Stages find components by role. |
| Root seed | The one seed a run is given. Every other seed is derived from it by name, so a run is reproducible. See [Train](train.md#the-run-context). |
| Run context | What a run's stages share: components, counters, seeds and published values. See [Train](train.md#the-run-context). |
| Setting | One option of a component, declared as a field of its settings class, with a type, a default and constraints. See [Core](core.md#settings). |
| Stage | One kind of work in a run, such as `interact` or `assess`. See [Train](train.md#stages). |
| Stale | A teacher's answer that arrived after the state it was about had moved on. Kept for the record, never applied. |
| Static solving | Attempting an instance with no environment: the model is given the inputs and predicts. See [Interaction](interaction.md#static-solving). |
| Store | Where records and documents are written, such as a run's output. Nothing in it is overwritten. See [Data](data.md#storage). |
| Terminated, truncated | How an episode ended: naturally, or cut off, such as by a step limit. They are never merged. |
