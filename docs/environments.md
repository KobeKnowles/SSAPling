# Environments

`ssapling.environments` defines the contract through which SSAPling uses an
environment: who takes part, who acts next, what each agent observes, and
what happens when they act. A new environment implements it directly. An
existing one, from a library or a remote service, is wrapped by an
**adapter** written downstream. The environment keeps its own state,
dynamics, and design.

| Part | What it is |
| --- | --- |
| [The simplest environment](#the-simplest-environment) | Two methods, `reset` and `step`. Everything else is optional. |
| [Results](#results) | `ResetResult` and `StepResult`: per-agent values, and separate termination and truncation. |
| [Participants and turn order](#participants-and-turn-order) | Who takes part, who is active, who acts next. |
| [Capabilities](#capabilities) | Optional extras: seeded reset, rendering, state, snapshots, batched lanes. |
| [The runner](#the-runner) | One awaitable entry point for sync and async environments, with communication failures kept apart from episode ends. |
| [Adapters](#adapters) | How to wrap an existing environment, with a checklist. |
| [A teacher inside the environment](#a-teacher-inside-the-environment) | Reporting executed actions and teacher responses. |
| [Registry](#environments-as-components) | Environments are components, built through Core. |

SSAPling ships no environments and no adapters. The tests use four toy
"native" environments, each with its own interface, and an adapter for each,
in `tests/fixtures/environments.py`: digit entry (one agent), Nim (two
players taking turns), matching pennies (two players at once), and an
ARC-AGI-3-like remote game.

## The simplest environment

An environment needs two methods:

```python
class DigitPadEnv:
    def reset(self):
        return ResetResult(observations={"student": {"a": 45, "b": 57, "entered": ""}})

    def step(self, actions):
        digit = actions["student"]
        ...
        return StepResult(observations={"student": ...}, terminated={"student": done})
```

Either may be `async def`, for example for a remote environment. That is a
complete single-agent environment: with one agent, it needs no turn logic,
no descriptions, and no capabilities. Every other part on this page is
optional.

**Per-agent values are keyed by agent id, even with one agent:**
`{"student": action}`, never a bare `action`. That keeps single-agent and
multi-agent records the same shape. The ids are the environment's choice.

```python
result = env.step({"student": action})                                # one agent
result = env.step({"player_1": action_1, "player_2": action_2})       # simultaneous
result = env.step({"agent": {"action": "ACTION6", "x": 12, "y": 7}})  # structured
```

Agent ids name participants, not the models or teachers controlling them.

`reset` may also take two keyword arguments, and SSAPling passes each only
when there is one:

- **`seed=`** makes the episode reproducible (the `seeded_reset`
  capability).
- **`options=`** is what the episode is set up from. The
  [episode loop](interaction.md#the-episode-loop) passes an instance's inputs here, so a task's instance and
  an environment meet without either knowing the other.

## Results

```python
ResetResult(observations={...}, info={...}, action_restrictions={...})

StepResult(
    observations={"student": ...},   # each agent's observation (final, if its part ended)
    rewards={"student": 1.0},        # optional; None means no rewards, not zero
    terminated={"student": False},   # optional; per agent: reached a terminal state
    truncated={"student": False},    # optional; per agent: cut off, e.g. by a time limit
    episode_terminated=None,         # optional; see below
    episode_truncated=None,
    info={...},                      # diagnostics, passed through
    executed_actions=None,           # optional; what actually ran, if it can differ
    teacher_responses=None,          # optional; from a teacher the environment owns
    action_restrictions=None,        # optional; what each agent may do next
)
```

- **No rewards is not zero rewards.** `rewards=None` means the environment
  has none. An adapter never invents them.
- **Termination and truncation are separate.** Terminated means a terminal
  state (won, lost, solved). Truncated means cut off before one, for example
  by a step limit. `None` means not reported. The flags must be `True` or
  `False`, so an adapter converts values such as NumPy booleans with
  `bool(...)`.
- **An agent finishing need not end the episode.** One player can be out
  while the others play on.
- **The episode's end is the environment's rule.** Set `episode_terminated`
  or `episode_truncated` and SSAPling uses exactly what you say, whatever the
  per-agent flags show: the game ends as soon as anyone wins, the last player
  standing wins while still playing, an episode-wide time limit with no
  per-agent flags at all, or both signals at once. Set either one and you
  own both; the other counts as `False`.
- **Only when the environment says nothing** does SSAPling fill the gap: the
  episode ends once every agent the result reports on has ended, terminated
  if all of them terminated and truncated otherwise. That makes the
  single-agent case automatic (`terminated={"student": True}` ends the
  episode) and makes the adapter example below work as written. An
  environment that reports on only some of its agents in a step should give
  the episode signals itself. `result.episode_over` and
  `result.ended_agents` read the outcome.
- **Action restrictions** say what an agent may do *now*, such as ARC's
  available actions or a mask, in the environment's own form. Whether an
  action fits the interface at all is a separate question, answered by the
  action descriptions.

Payloads (observations, actions, info) pass through uninterpreted. The
results copy their mappings when made and pickle, so they can come from an
environment in another process.

## Participants and turn order

These optional members are attributes, read without waiting. An adapter for
a remote environment keeps them from the latest response.

| Member | Meaning | When the environment leaves it out |
| --- | --- | --- |
| `agents` | Every agent id that can take part. | The agents observed at reset. |
| `active_agents` | The agents taking part now. | Those just given an observation whose part has not ended. |
| `next_agents` | The agents that act in the next step. | Every active agent, all at once. |

Consumers read them through helpers that apply the defaults:

```python
from ssapling import environments

first = env.reset()
environments.next_agents(env, first)     # ("student",)
last = env.step({"student": "2"})
environments.next_agents(env, last)      # ("student",), until the episode ends
```

- A **single-agent** environment needs none of these.
- A **simultaneous** environment needs none of them either: by default,
  everyone active acts at once.
- A **sequential** environment says whose turn it is with `next_agents`, as
  `NimEnv` does. An environment that gives an observation only to the agent
  whose turn is next gets the right order even without it.

The environment defines how many agents there are, their roles, and when
they enter or leave.

**Descriptions** of observations and actions are optional attributes too:
`observation_descriptions` and `action_descriptions`, keyed by agent id, in
the environment's own form (a library's space objects, an ARC action list).
SSAPling defines no space types. It passes them on unchanged, and a model
checks them against its own (see [Models: descriptions](models.md#descriptions)):

```python
models.check_descriptions(
    model,
    observations=env.observation_descriptions["student"],
    actions=env.action_descriptions["student"],
    participant="student",
)
```

## Capabilities

Optional extras an environment may offer. Each is a fixed name that means
particular methods:

| Capability | Methods | What it gives |
| --- | --- | --- |
| `seeded_reset` | `reset` with a `seed` parameter | Reproducible episodes. |
| `render` | `render()` | A view of the current state, such as an image or text. |
| `state` | `state()` | The full state, including what agents cannot observe, for inspection. |
| `snapshot` | `snapshot()`, `restore(saved)` | Save a state and return to it later. |
| `batched` | `reset_lanes(lanes)`, `step_lanes(actions)`, and `lanes` | Several independent copies stepped together. |

**An environment offers a capability by having its methods.** It writes
nothing else. Consumers ask before using one:

```python
environments.supports(env, "snapshot")    # True: it has snapshot and restore
environments.supports(env, "teleport")    # ValueError: not a capability (a bug in the caller)
```

**An environment can switch a capability off** with a `supports(capability)`
method of its own. An adapter needs this when its class implements a
capability that the environment it wraps cannot honour. `NimEnv`'s `reset`
takes a seed, but the native game ignores it, so the adapter says so:

```python
class NimEnv:
    def supports(self, capability):
        return capability != "seeded_reset"   # don't promise reproducibility we can't keep
```

The rules are the same as for tasks:

- An environment's own `supports` only switches capabilities off. It answers
  `False` for those and `True` for every other.
- It can never claim a capability it lacks the methods for.
- It must not call `environments.supports` on itself, because
  `environments.supports` calls it. A wrapper may call it on the environment
  it wraps, which is how `AsyncProxy` in the fixtures reports what its inner
  environment supports.
- `supports` must be a method. Any other attribute of that name is rejected
  with a `TypeError`.

**Batched lanes.** A batch is several independent copies of an environment,
called lanes, stepped together, natively or by a downstream vectorising
wrapper. SSAPling includes no vectoriser. `lanes` is how many there are.
`reset_lanes(lanes)` resets the listed lanes (optionally with `seeds=` and
`options=`, keyed by lane). `step_lanes(actions)` steps the lanes given
actions (keyed by lane, then by agent). Both return a result per lane. **A
lane that ends waits to be reset**: nothing resets it automatically, so its
final observation is never confused with the next episode's first. A batch
of independent environments is not a multi-agent world.

## The runner

`EnvironmentRunner` is one awaitable entry point for synchronous and
asynchronous environments, like the teacher runner:

```python
runner = EnvironmentRunner()                     # sync envs run in the loop's thread
runner = EnvironmentRunner(placement="thread")   # sync envs run in a worker thread

first = await runner.reset(env, seed=7, options=instance.inputs)
result = await runner.step(env, {"student": "2"}, timeout=5.0)
await runner.close(env)                          # calls env.close(), if it has one
```

It checks that each call is possible and that each result has the right type:

| Problem | What happens |
| --- | --- |
| A seed for an environment without seeded reset | `UnsupportedCapabilityError`: check `environments.supports(env, "seeded_reset")` first. |
| `options` for a `reset` that does not take them | `TypeError`, saying to add an `options=None` parameter. |
| `step` returns the wrapped environment's own result, such as a tuple | `TypeError`: an adapter converts it to a `StepResult`. |
| Actions not keyed by agent id | `TypeError`. |
| The environment raises its own error, such as an illegal move | Raised as it is. |

**A communication failure is not the end of an episode.** An episode ends
only through the termination and truncation signals of a `StepResult`. When a
remote adapter loses its connection, it raises
`EnvironmentCommunicationError`. When an environment does not answer within
`timeout` seconds, the runner raises it too. **Nothing is retried**, because
a step that may already have executed must not be submitted twice. The error
says whether the call may have taken effect, so the caller can decide what to
do: stop the run, mark the record failed, or ask the service for the game's
state.

A timeout cannot interrupt a synchronous environment run directly, since it
blocks the loop, so that combination is rejected; use `placement="thread"`.
To run an environment in a separate process, build it with `in_process`
(see [Interaction](interaction.md#running-a-component-in-another-process)).

## Adapters

An adapter wraps an existing environment and translates calls and results
**without changing its behaviour**. A typical example, for a
single-agent environment whose interface is `reset(seed) -> (observation,
info)` and `step(action) -> (observation, done, info)`:

```python
class ExistingEnvironmentAdapter:
    def reset(self, *, seed=None):
        observation, info = self.wrapped.reset(seed)
        return ResetResult(observations={"agent": observation}, info=info)

    def step(self, actions):
        observation, done, info = self.wrapped.step(actions["agent"])
        return StepResult(
            observations={"agent": observation},
            terminated={"agent": done},
            truncated=None,  # not reported by the wrapped environment
            rewards=None,    # the wrapped environment has no rewards
            info=info,
        )
```

### Adapter checklist

- [ ] **Key everything by agent id**, even with one agent.
- [ ] **Keep termination and truncation apart.** A time limit is truncation.
      If the wrapped environment reports only "done", report it as what it
      means, and say nothing (`None`) about what it does not report.
- [ ] **Report rewards only if the environment has them.** `rewards=None`,
      never zeros.
- [ ] **Keep final observations.** Under autoreset, the step that ends an
      episode returns the final observation; the next episode's first one
      comes from `reset`. Never mix the two.
- [ ] **Preserve turn order and reward timing.** Expose `next_agents` for a
      sequential environment, and refuse actions out of turn rather than
      reorder them.
- [ ] **Report unsupported features; never invent them.** Switch off
      capabilities the wrapped environment cannot honour (such as seeds it
      ignores). Do not invent state access, rewards, or action meanings.
- [ ] **Pass descriptions through unchanged**, in the environment's own form.
- [ ] **For a remote environment:** raise `EnvironmentCommunicationError`
      when the connection fails, saying whether the call may have taken
      effect, and never retry an action blindly.
- [ ] **Convert flags to `bool`** and results to `ResetResult` and
      `StepResult`. The runner rejects anything else.
- [ ] **Compare native and adapted behaviour** in a test: the same actions in
      the same order, the same rewards, the same final observations, and the
      same completion. The tests in `tests/environments/test_adapters.py`
      show how.

### Interactive ARC

ARC-AGI-3 is interactive. Its adapter (`ArcEnv` in the fixtures shows the
shapes against a fake service) must preserve:

- **frame sequences**: an observation can hold several frames;
- **parameterised actions**: `{"action": "ACTION6", "x": 3, "y": 1}`;
- **available-action information**, as action restrictions;
- **game status**.

It must not conflate **level progress**, **episode completion**, **resets**,
and **evaluation-session accounting**. Completing a level is progress,
reported in `info`, and only a win or a game over ends the episode. Benchmark
scoring belongs to the downstream evaluation protocol, so the adapter
reports no rewards.

## A teacher inside the environment

A teacher can sit inside the transition function. SSAPling adds no hooks
inside environments. The environment owns the teacher and calls it directly
with the [Teachers](teachers.md) contract. To make its records look the same
as any other placement's, it reports what the teacher did:

```python
def step(self, actions):
    request = TeacherRequest(id=..., operation="review_action",
                             context={...}, proposals=actions)
    response = self.teacher.respond(request)
    executed = apply_intervention_if_any(actions, response)   # only an intervention changes it
    ...
    return StepResult(observations=..., executed_actions=executed,
                      teacher_responses=[response])
```

`DigitPadEnv(teacher=DigitTutor(...))` in the fixtures does exactly this: a
wrong digit proposed by the student is reviewed and, with intervention on,
replaced by the right one, and the result reports both. The environment
owns the teacher, so it is the one that resets and closes it, and the episode
loop does not consult the teacher itself. An async teacher needs an
`async def step`, or the environment can ask it through a `TeacherRunner`.

## Environments as components

Environments are registered and built through Core like everything else. The
registry is `ssapling.environments.registry`, also reachable as
`ssapling.registries.environments`. Live objects, such as an owned teacher or
a connection, are passed at construction:

```python
from ssapling import registries as r

r.environments.register("digit_pad", DigitPadEnv)
env = r.environments.create("digit_pad", teacher=tutor)
```

`Environment` is a `typing.Protocol`, so a type checker verifies an
environment or adapter structurally. Nothing inherits from it.

## Errors

| Error | Means |
| --- | --- |
| `EnvironmentCommunicationError` (a `ConnectionError`) | A remote environment could not be reached. Not the end of the episode. `may_have_executed` says whether the call may have taken effect; it was not retried. |
| `UnsupportedCapabilityError` (a `LookupError`) | A capability was used that the environment does not support. |
| `TypeError` from `ResetResult`, `StepResult` | Values not keyed by agent id, flags that are not booleans, teacher responses that are not `TeacherResponse`s. |
| `TypeError` from the runner | Options for a reset that does not take them, a result of the wrong type, actions not keyed by agent id, or an object that is not an environment. |
| `TypeError`, `ValueError` from `environments.supports` | A `supports` attribute that is not a method; an unknown capability name. |
| `ValueError` from the runner | A timeout that is not a positive number, or a timeout for a synchronous environment run directly. |
