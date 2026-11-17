# Models

`ssapling.models` defines how SSAPling uses a model or agent: how it **acts**,
how its **state** is managed, and how it **learns**. A model can be a neural
network, a reinforcement learning agent, an LLM call, or a hand-written rule.
SSAPling requires no machine learning framework.

| Part | What it is |
| --- | --- |
| [The simplest model](#the-simplest-model) | One method, `act`. Everything else is optional. |
| [Buffered models](#buffered-models) | `perceive` and `poll`, for a model that takes its own time. `as_buffered` gives any act-only model this interface. |
| [Participants and lanes](#participants-and-lanes) | An optional `key`, so one model can serve several participants or batch lanes. |
| [State](#state) | Optional: episode resets, saving and loading persistent state, the evaluation flag. |
| [Diagnostics](#diagnostics) | Optional: what a model reports about each decision, such as the tokens or compute it used, kept in the record. |
| [Learning](#learning) | Optional: the learning operations a model supports, and `learn`. |
| [Learning signals](#learning-signals) | Optional: which signals a model learns from; only those reach it. |
| [Descriptions](#descriptions) | Optional: what a model can handle, checked against the environment. |
| [Registry](#models-as-components) | Models are components, built through Core. |

SSAPling ships no models. The tests use toy models in
`tests/fixtures/models.py`: `LookupModel` (act only), `SlowThinker` (buffered,
with internal steps), `AsyncEcho` (async), and `CountingLearner` (which
declares every optional part).

## The simplest model

A model needs one method:

```python
class LookupModel:
    def act(self, observations):
        return self.table.get(observations, "?")
```

`act` maps observations to an output: an action during interaction, or a
prediction in static solving. It may be `def` or `async def`, for example
for a model behind a remote API. That is a complete model. Every other part
on this page is optional, and **a model that leaves a part out simply does
not have it**. SSAPling never calls a method a model does not have, and a
procedure that genuinely needs one reports it by name.

| A model that... | ...writes |
| --- | --- |
| Acts | `act` |
| Takes its own time to think | `perceive`, `poll`, and optionally `advance` |
| Serves several participants or lanes | a `key=None` parameter on those methods |
| Has memory within an episode | `reset_episode` |
| Has parameters or long-term memory to save | `export_state`, `restore_state` |
| Behaves differently under evaluation | `set_evaluating` |
| Reports on its own decisions, such as compute used | `diagnostics` |
| Learns | `learning_operations`, `learn`, and optionally `signals` |
| Can only handle certain inputs | `observation_description`, `action_description`, or `accepts_description` |

## Buffered models

Some models need time between seeing and acting, and the coordinating code
may not wait for them. Acting then goes through an input and an output
buffer:

```python
model.perceive(observations)   # put observations in the input buffer; returns at once
output = model.poll()          # the next output, or NOTHING if none is ready yet
```

What happens in between is the model's own business. It may process in the
background, for example in its own thread. Or it may offer **`advance()`**,
which does one internal step of processing. Then the coordinating code
drives it, and runs in simulated time are reproducible: "the model had 3
steps to think" means the same thing on every machine.

```python
thinker = SlowThinker(think_steps=3)
thinker.perceive("frame-1")
thinker.advance(); thinker.poll()   # NOTHING
thinker.advance(); thinker.poll()   # NOTHING
thinker.advance(); thinker.poll()   # "frame-1#1"
```

**`NOTHING`** means no output is ready. It is not `None`, because `None` can
be a real output, so compare with `is`: `if output is NOTHING`. It stays the
same object when sent to or from another process.

### `as_buffered`: simple models never deal with buffers

`as_buffered(model)` gives any model the buffered interface. A model that
already has `perceive` and `poll` is returned unchanged. An act-only model is
wrapped:

| Wrapper method | What it does |
| --- | --- |
| `perceive(observations)` | Queues the observations. |
| `advance()` | Calls the model's `act` on the oldest queued observations and queues the output. An act-only model needs exactly one step per output. Does nothing if nothing is queued. |
| `poll()` | Returns the oldest output not yet read, or `NOTHING`. |
| `act(observations)` | The model's own `act`, called directly. |
| `reset_episode()` | Empties the queues, then calls the model's own `reset_episode`, if it has one. |
| `export_state()` | The model's own state (`None` if it keeps none) with the queues: `{"model": ..., "queues": [[key, observations, outputs], ...]}`. |
| `restore_state(state)` | Returns to an exported state: the queues, then the model's own. |

```python
model = as_buffered(LookupModel(...))
model.perceive("2+2")
model.poll()       # NOTHING: perceived, not yet processed
model.advance()
model.poll()       # "4"
```

- **Nothing is dropped.** The queues keep every observation and output, in
  order. If an output arrives after the coordinating code has moved on, it
  is still there, and what to do with it (discard it, or use it next turn)
  is the coordinating code's choice (see [Interaction: timing](interaction.md#timing)).
- **Sync or async is kept.** `advance` and `act` are `async def` exactly
  when the model's `act` is.
- **A checkpoint keeps the step.** A state exported between `perceive` and
  `poll` holds what is queued, so a model restored from it goes on exactly
  as the original would. The state methods are `async def` exactly when the
  model's are.
- **Everything else is the model's own.** `learn`, `learning_operations`,
  and the rest are reached through the wrapper, so the wrapper can stand in
  for the model everywhere.

How long the coordinating code waits for an output is not the model's
decision. That is the timing mode, chosen in the [episode loop](interaction.md#timing):

| Mode | The coordinating code... |
| --- | --- |
| Turn-based | Waits for an output. |
| Deadline | Waits up to a limit: a number of `advance` steps on a simulated clock, or seconds on a wall clock. |
| Real-time | Gives the model a tick, a fixed time long, and acts at its end with the output it has by then. |

When no output is ready in time, the environment or the coordinating code
applies its own default, and the record marks that a default was used.

## Participants and lanes

One model can control several participants (for example both players in
self-play) or run several independent episodes at once in batch lanes. Each
needs its own episode state. Methods then take an optional **`key`** that
says which participant or lane is meant:

```python
model.perceive(observations, key="player_1")
model.poll(key="player_1")
model.reset_episode(key="lane-3")
```

**A key is passed only when there is one.** With a single participant and a
single lane, SSAPling calls `act(observations)` with no key, so a simple
model never needs the parameter. A model used with keys must accept one,
because otherwise it could mix up participants silently. Giving it a key
without that raises an error that says to add a `key=None` parameter.

## Diagnostics

Some things only the model knows: the tokens it used, its FLOPs or GPU
time, its confidence, a reasoning trace. A model reports them with an
optional **`diagnostics()`** method, which the episode loop calls after each
decision and keeps in the record, in `step.diagnostics[agent]`:

```python
class TokenCountingModel:
    def act(self, observations):
        answer, self.used = self.client.complete(observations)   # downstream code
        return answer

    def diagnostics(self):
        return {"tokens": self.used}                          # about the decision just made
```

- What it returns is the model's own, passed through uninterpreted, like an
  environment's `info`. Return `None` to report nothing for a decision.
- It is called once per agent per step, also when no output came in time,
  and it takes `key` like the other methods when the model serves several
  participants or lanes.
- For a buffered model, a step's report covers everything since that step's
  observation: every `advance` until its output, or until the deadline. The
  loop calls `diagnostics()` once per step, not once per `advance`, so a
  model that thinks in several steps keeps what each did and reports them
  together.
- It works the same in training and in assessment, since both use the
  episode loop. A measurement can then sum it, for example compute per
  solved puzzle (see [Evaluate](evaluate.md#measurements)). Totals over a
  model's whole life belong in its persistent state, as counters it keeps.

## State

A model may hold two kinds of state. Both are optional, and what each
contains is up to the model.

| Kind | Examples | Method | Called |
| --- | --- | --- | --- |
| **Episode state** | Activations, recurrent memory, context | `reset_episode(key=None)` | At each episode boundary, per participant or lane. |
| **Persistent state** | Parameters, long-term memory, a replay buffer | `export_state()` and `restore_state(state)` | At checkpoints and on resume. |
| **Memory across episodes** | A short-term memory that lasts a day of problems | `reset_memory(key=None)` | When the coordinating code decides, such as after a night of learning. |

A model whose memory lasts across episodes clears it in `reset_memory`, not in
`reset_episode`. SSAPling never calls `reset_memory` itself: the downstream
project's stages and controllers decide when a memory ends. Its state is part
of the model's persistent state, so `export_state` holds it.

`export_state` returns plain data, or data a [codec](data.md#codecs) can save, and
`restore_state` takes it back. A model can keep its own lifetime counters,
such as updates applied, in its persistent state, so they travel with it.
SSAPling does not require or define them.

**`set_evaluating(evaluating)`** tells a model whether it is being evaluated.
By default, evaluation leaves persistent state unchanged: Evaluate calls no
learning operation unless the assessment enables learning during evaluation,
for example for continual learning. SSAPling cannot check that a model
ignores the flag, so assessment results record whether learning was
enabled: the `learning_operation` setting is in their provenance.

The coordinating code calls these at episode boundaries, checkpoints and
assessments. The [episode loop](interaction.md#the-episode-loop) calls
`reset_episode` at the start of each episode; [Evaluate](evaluate.md) and
[Train](train.md) call the others.

## Learning

There is no single training method. A model **lists the learning operations
it supports** and performs them with `learn`:

```python
class CountingLearner:
    learning_operations = frozenset({"learn_from_batch", "consolidate"})

    def learn(self, operation, **payload):
        ...
        return {"loss": 0.31, "updates": 120}      # statistics, keyed by name
```

| Example operation | Typical use |
| --- | --- |
| `learn_from_batch` | Supervised or imitation learning on records from Data. |
| `learn_from_experience` | Online or reinforcement learning from transitions as they happen. |
| `consolidate` | An offline phase between episodes, such as replaying internal memory. |

Consumers ask before calling one, and Train checks, before a run, that the
model supports every operation its procedure calls:

```python
from ssapling import models

if models.supports(model, "consolidate"):
    stats = model.learn("consolidate", limits={"seconds": 60})
```

- Operation names and payloads are an **open set**, so a downstream project
  defines new forms of learning without changing SSAPling. A payload can
  include limits, such as a time limit. Enforcing them is downstream.
- A model without `learning_operations` learns nothing. That is allowed: a
  fixed rule or a frozen network is a model.
- The learning can live inside the model, for an agent that learns by
  itself, or in a separate downstream learner that holds the loss and
  optimiser and updates the model. Both satisfy this contract.
- `learning_operations` must be a collection of names (a single string is
  rejected), and a model that lists operations must have `learn`.

## Learning signals

A model may learn from environment rewards, teacher supervision or feedback,
recorded demonstrations, or examples it generates itself. It **declares the
signals it uses**, and only those are passed to it:

```python
class CountingLearner:
    signals = frozenset({"reward", "supervision"})

models.filter_signals(model, {"reward": 1.0, "supervision": "7", "feedback": "Carry the 1."})
# {"reward": 1.0, "supervision": "7"}
```

- **Teacher feedback is not turned into reward** unless a downstream project
  chooses to. A model that wants feedback declares `"feedback"`.
- A model that declares no signals receives none.
- Signal names are an open set. The common ones match the environment's
  `reward` and the teacher's contribution kinds (`supervision`, `feedback`,
  `demonstration`), so a teacher's response can be passed on by kind.

## Descriptions

Environments report their own descriptions of observations and actions, such
as a library's space objects or an ARC action list, and SSAPling defines no
space types of its own. A model may declare what it handles, and
`check_descriptions` reports a mismatch clearly before the run:

```python
class CountingLearner:
    observation_description = "a string of digits"
    action_description = "one digit"

models.check_descriptions(model, observations=env_observations, actions=env_actions,
                          participant="student")
# DescriptionMismatchError: Model CountingLearner cannot handle the environment's
# descriptions as participant 'student':
#   - actions: the model handles 'one digit', the environment gives 'a grid'
```

For each kind both sides describe:

- If the model has **`accepts_description(kind, description)`**, it decides.
  Use this when the model handles a range of descriptions, such as any grid
  up to 30 cells across. `kind` is `"observation"` or `"action"`. It answers
  true to accept, and false, or a **`Refusal(reason)`** that says why, to
  refuse. A `Refusal` is false wherever a bool is read, and its reason is in
  the mismatch: `observations: the model does not accept {...}: <reason>`.
- Otherwise the model's `observation_description` or `action_description`
  must **equal** the environment's.

Where either side gives no description, there is nothing to compare, so
nothing is checked. Every mismatch is reported at once.

## Models as components

Models are registered and built through Core like everything else. The
registry is `ssapling.models.registry`, also reachable as
`ssapling.registries.models`:

```python
from ssapling import registries as r

r.models.register("lookup", LookupModel)          # LookupModel.Settings is picked up
model = r.models.create("lookup", {"table": {"2+2": "4"}})
```

`ActModel` and `BufferedModel` are `typing.Protocol`s, so a type checker
verifies a downstream model structurally; `Model` means either. Nothing
inherits from them.

## Errors

| Error | Means |
| --- | --- |
| `DescriptionMismatchError` (a `ValueError`) | The model cannot handle the environment's descriptions; the error lists every mismatch. |
| `TypeError` from `as_buffered` | The object has neither `act` nor both `perceive` and `poll`. |
| `TypeError` from a wrapper method | A key was given to a model whose method does not take one. |
| `TypeError` from `models.supports`, `filter_signals` | Declarations that are not a collection of names, or learning operations without a `learn` method. |
| `TypeError` from `check_descriptions` | The model's `accepts_description` is not a method. |
