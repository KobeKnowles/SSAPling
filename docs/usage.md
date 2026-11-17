# Using SSAPling

This guide goes from nothing to a finished, assessed training run. It follows
the project in [`examples/arithmetic/`](../examples/arithmetic/): a learner
that knows no arithmetic learns to add on a digit pad, from a tutor. Every
code block here comes from that example, which CI runs on every change, so
you can run it, change it, and compare.

| You write | SSAPling does |
| --- | --- |
| The task, environment, teacher and model, registered by name | Builds them from checked settings |
| A run file: what to build, the training procedure, when to stop | Checks the whole run before it starts, then runs it |
| Assessments: which problems, what to measure | Runs them, aggregates without inventing values, saves the results |
| Nothing more | Seeds, reserved problems, records, checkpoints, resuming |

## 1. Install

SSAPling needs Python 3.12 or newer, and nothing else. Run files in YAML
need the `yaml` extra:

```bash
pip install "ssapling[yaml]"
```

While SSAPling and a project change together, install it from a clone in
editable mode, so changes to SSAPling show up at once:

```bash
uv add --editable ../SSAPling          # or: pip install -e "../SSAPling[yaml]"
```

## 2. Write your components

A project supplies four kinds of component. None of them inherits from
anything: SSAPling only calls the methods it needs, and asks before using an
optional one. [Extending SSAPling](extending.md) covers each kind in full;
here are the example's, from
[`components.py`](../examples/arithmetic/components.py).

**The task** makes problems and scores answers. An `Instance` keeps the
`inputs`, which are all the student ever sees, apart from the `reference`
answer. Its `equivalence_key` says which instances are the same problem, so
that `12 + 34` and `34 + 12` never end up in two different collections:

```python
class AdditionTask:
    Settings = AdditionSettings                    # max_digits, checked when built

    def __init__(self, config: AdditionSettings) -> None:
        self.config = config

    def generate(self, *, params, seed) -> Instance:
        ...                                        # the same params and seed give the same problem
        return Instance(
            id=f"{a}+{b}",
            inputs={"a": a, "b": b},
            reference=a + b,
            descriptors={"digits": digits},        # for grouping results
            equivalence_key=("+", min(a, b), max(a, b)),
        )

    def score(self, instance, outcome) -> Score:
        return Score.local(1.0 if outcome == instance.reference else 0.0)
```

**The environment** is the world the student acts in. It is set up from an
instance's inputs, and everything it reports is keyed by agent, even with
one agent:

```python
class DigitPad:
    def reset(self, *, options=None) -> ResetResult:          # options: the instance's inputs
        ...
        return ResetResult(observations={"student": {"a": a, "b": b, "entered": ""}})

    def step(self, actions) -> StepResult:
        entry = actions["student"]
        ...                                        # a wrong digit, or the last one, ends it
        return StepResult(
            observations={"student": {"a": a, "b": b, "entered": entered}},
            terminated={"student": done},
            info={"correct": correct},
        )
```

**The teacher** answers requests. `ColumnTutor` reviews each digit the
student proposes. Its answer carries **contributions**, by kind and then by
agent: a `supervision` saying how the column works out, and, when the
student is wrong, an `intervention` with the right digit:

```python
class ColumnTutor:
    operations = frozenset({"review_action"})
    required_context = {"review_action": ("instance", "observations")}

    def respond(self, request: TeacherRequest) -> TeacherResponse:
        ...
        contributions = {"supervision": {"student": lesson}}
        if request.proposals.get("student") != str(write):
            contributions["intervention"] = {"student": str(write)}
        return TeacherResponse.answered(request, contributions)
```

**The model** acts, and may learn. Only `act` is required. A model that
learns lists its learning operations and the learning signals it takes;
only those signals ever reach it:

```python
class FactLearner:
    learning_operations = frozenset({"learn_from_experience"})
    signals = frozenset({"supervision"})

    def act(self, observations) -> str:
        ...                                        # one digit, from the facts it has memorised

    def learn(self, operation, **payload) -> dict[str, int]:
        lesson = payload["signals"].get("supervision", {}).get("student")
        ...                                        # memorise the fact
        return {"facts": len(self.facts)}          # published as the value "learn_from_experience"

    def export_state(self) -> dict:                # saved with each checkpoint
        return {"facts": dict(self.facts)}

    def restore_state(self, state) -> None:        # given back when the run resumes
        self.facts = dict(state["facts"])
```

## 3. Register them

A project registers its components by name in one function, which it calls
at start-up. A run file then refers to them by those names:

```python
from ssapling import registries as r

def register_components() -> None:
    r.tasks.register("addition", AdditionTask)
    r.environments.register("digit_pad", DigitPad)
    r.teachers.register("column_tutor", ColumnTutor)
    r.models.register("fact_learner", FactLearner)
```

A component's `Settings` class, if it has one, is checked when it is
registered, so a mistake in declaring it shows up at once rather than in
the middle of a run.

## 4. Describe the run

One file describes the whole run: the components to build, the procedure,
and when to stop. This is [`run.yaml`](../examples/arithmetic/run.yaml),
slightly shortened:

```yaml
components:                           # built in this order; each key is the component's role
  task: {kind: tasks, name: addition, settings: {max_digits: 3}}
  sampler:
    kind: samplers
    name: generated
    settings: {vary: digits, values: [1, 2, 3]}
    uses: {source: task}
  collections:                        # reserved problems: training never draws them
    kind: collections
    settings:
      source: task
      build:
        validation: {sampler: sampler, size: 30, seed: 1}
        test: {sampler: sampler, size: 30, seed: 2, held_out: true}
  environment: {kind: environments, name: digit_pad}
  tutor: {kind: teachers, name: column_tutor}
  model: {kind: models, name: fact_learner}
  monitoring:
    kind: assessments
    name: standard
    settings:
      collections: [validation]
      aggregations: {accuracy: {measurement: score, aggregator: mean}}
    uses: {collections: collections, task: task, env: environment, outcome: outcome}

procedure:
  - stage: interact
    settings:
      episodes: 20
      teacher: tutor
      episode: {intervene: true}
      learn: {operation: learn_from_experience, after: step}
  - stage: assess
    every: {episodes: 100}
    settings: {assessment: monitoring}
  - stage: checkpoint
    every: {episodes: 200}

stop: {episodes: 2000, value: monitoring.accuracy, at_least: 1.0}
```

- **Components.** Each has a **role**, its key, and is built from a
  registry (`kind`), by `name`, with checked `settings`. `uses` passes it
  components built before it, by role; nothing is passed by name alone.
- **Collections** are drawn from the task, and reserve their problems.
  Training draws only from what is left, so no problem of any collection is
  ever trained on. A `held_out` collection is served only to assessments.
- **The procedure** is a list of stages, gone through again and again. The
  `interact` stage runs episodes with the tutor and learns after each step;
  `assess` runs the monitoring assessment and publishes its accuracy;
  `checkpoint` saves everything needed to resume. `every` triggers an entry
  each time a counter has grown by that much.
- **The stop condition** is checked before every stage. Here: 2000
  episodes, or validation accuracy 1.0, whichever comes first.

[Train](train.md) describes the built-in stages, triggers and conditions,
and run files in full.

## 5. Run it

```python
from ssapling.train import Runner

register_components()
given = {"outcome": entered_number}        # built in Python: a function, given by role
runner = Runner.from_config("run.yaml", seed=7, output="runs/arithmetic", components=given)
context = await runner.run()

context.counters                           # {"episodes": 400, "steps": ..., "updates": ..., "stages": ...}
context.read("monitoring.accuracy")        # 1.0
```

`run` is `async`. From ordinary code, call it as
`asyncio.run(runner.run())`, as [`run.py`](../examples/arithmetic/run.py)
does.

- **The seed** is the run's root seed. Every other seed is derived from it
  by name, so the same file and seed give the same run on any machine.
- **Components given as they are**, such as the `outcome` function here,
  join the run under their role, and the run file can `use` them.
- **`output`** is where the run writes. A second run into the same output
  is refused.

**Everything is checked before anything runs.** When the runner is made,
the run file's components and their settings are checked. When `run`
starts, before any stage runs, every stage's settings are checked, and so
are the components each stage needs and the values each condition reads.
Each time, all the problems are reported together:

```text
ConfigError: Invalid run file run.yaml:
  - components.model.name: Unknown model 'fact_lerner'. Did you mean 'fact_learner'? ...

ConfigError: Invalid procedure:
  - procedure[0].settings.episodes: must be >= 1, got 0
  - procedure[1]: stage 'assess' requires 'monitor', which the run does not have (its components: ...)
  - stop.value: reads 'monitoring', which no stage in the procedure publishes
```

## 6. Read what it wrote

The run's output is a store, readable while the run is still going:

| What | Where |
| --- | --- |
| The run's settings and seeds, enough to rebuild it | the document `run` |
| Every training episode, step by step | the records `episode-0`, `episode-1`, ... |
| Each assessment's results | the documents `results/monitoring-6`, ... |
| Checkpoints | the documents `checkpoints/13`, ... |
| Every decision the run made | the document `decisions` |

```python
from ssapling.data import RecordStore
from ssapling.evaluate import load_results

store = RecordStore("runs/arithmetic")
record = store.read("episode-41")              # a Record: header, steps, summary
record.steps[0].teacher_responses              # what the tutor said at the first step
results = load_results(store, "monitoring-25")
results.aggregates["accuracy"].value
```

A record holds every step: what the model proposed, what was executed, the
teacher's responses, the observations, and how the episode ended. See
[Data: records](data.md#records).

## 7. Assess on held-out problems, once

The validation problems decided when to stop, so they cannot also give a
fair measure of what was learned. The held-out `test` collection is used
once, at the end. The run file builds a second assessment, `final`, and
[`run.py`](../examples/arithmetic/run.py) runs it after training:

```python
final = context.components["final"]
results = await final.run(
    context.components["model"],
    run_id=context.output_name("final"),
    seed=7,
    model_info=dict(context.counters),          # how much training the model had
)
save_results(context.components["store"], results)
results.aggregates["accuracy"].value          # 0.9666...: one column fact was never taught
results.aggregates["by_digits"].groups["2"].value
```

A missing value is never counted as zero, and an aggregate over no values
is `None`. See [Evaluate](evaluate.md).

## 8. Resume after a crash

```python
runner = Runner.resume("runs/arithmetic", components=given)
context = await runner.run()
```

The run is rebuilt from its saved settings, every component gets back the
state it saved at the last checkpoint, and the run continues with the same
episodes, problems and seeds, so it ends as if it had never stopped. What
it writes from then on is named with `resume-1/`, such as
`resume-1/episode-201`, so nothing is overwritten.
Components given as they are must be given again. See
[Train: checkpoints and resuming](train.md#checkpoints-and-resuming).

## Without a run file

Everything above also works from Python, with no file:

```python
from ssapling.train import Condition, Entry, Every, Procedure, Runner

procedure = Procedure(
    [
        Entry("interact", {"episodes": 20, "teacher": "tutor",
                           "episode": {"intervene": True}, "learn": {}}),
        Entry("assess", {"assessment": "monitoring"}, every=Every(episodes=100)),
    ],
    stop=Condition(episodes=2000, value="monitoring.accuracy", at_least=1.0),
)
components = {"model": FactLearner(), "environment": DigitPad(), "tutor": ColumnTutor(), ...}
context = await Runner(procedure, components, seed=7, output="runs/arithmetic").run()
```

A procedure from Python is checked exactly as one from a file.

## Next

- [Extending SSAPling](extending.md): each kind of component in full, and
  how to change SSAPling's defaults where a project needs something else.
- [Overview](overview.md): the areas, how they depend on each other, and a
  glossary.
- The area guides, for the details: [Core](core.md), [Tasks](tasks.md),
  [Teachers](teachers.md), [Models](models.md),
  [Environments](environments.md), [Data](data.md),
  [Interaction](interaction.md), [Evaluate](evaluate.md), [Train](train.md).
