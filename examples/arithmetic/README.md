# Arithmetic: learning addition from a tutor

A complete, small SSAPling project in two Python files and a run file. A
learner that knows no arithmetic learns to add on a digit pad, from a tutor
that explains each column, and is then assessed on problems it never
trained on. It needs no third-party library beyond PyYAML, for the run file,
and runs in about a second.

| File | What it is |
| --- | --- |
| [`components.py`](components.py) | The project's own components, registered by name: the `addition` task, the `digit_pad` environment, the `column_tutor` teacher, and the `fact_learner` model. |
| [`run.yaml`](run.yaml) | The run: which components to build, the training procedure, and when to stop. |
| [`run.py`](run.py) | Registers the components, runs the run file, then assesses the held-out problems once. |

## Running it

From the repository root, after `uv sync`:

```bash
uv run python examples/arithmetic/run.py
```

```text
trained for 400 episodes
validation accuracy: 1.0
test accuracy: 0.9666666666666667
  1-digit problems: 1.0 (9 problems)
  2-digit problems: 0.9090909090909091 (11 problems)
  3-digit problems: 1.0 (10 problems)
```

The run writes to `runs/arithmetic`. A second run into the same directory
is refused, since nothing a run writes is ever overwritten: give another
with `--output runs/again`. `--resume` continues the run at `--output` from
its last checkpoint and, because every seed is derived from the run's root
seed, ends exactly where an uninterrupted run would.

Outside this repository, install SSAPling with its YAML extra:
`pip install "ssapling[yaml]"`.

## What happens

1. **Components.** `run.py` registers the project's components, and the
   runner builds those the run file lists, in order: the task, a sampler of
   1-, 2- and 3-digit problems, two reserved collections of 30 problems
   (`validation`, and a held-out `test`), the pad, the tutor, the model, and
   two assessments.
2. **Training.** Each pass, the `interact` stage runs 20 episodes. On every
   step the tutor reviews the digit the learner proposes and supervises:
   which column it was, and the digit and carry it gives. The learner
   declares `supervision` as its one learning signal, so it receives exactly
   that, after each step, and memorises the fact. When the learner is wrong,
   the tutor also offers the right digit as an intervention; the episode
   settings say `intervene: true`, so it is applied and every episode is
   finished. No training episode ever uses a problem from a collection.
3. **Monitoring.** Every 100 episodes the `assess` stage runs the
   `monitoring` assessment on the validation problems, with no tutor, and
   publishes the accuracy. Every 200 episodes the run is checkpointed.
4. **Stopping.** The run stops when validation accuracy reaches 1.0, or
   after 2000 episodes, whichever comes first. Here it stops at 400.
5. **The held-out test.** Only then does `run.py` assess the `test`
   problems, once, and save the results to the run's store.

## Why the test score is lower

Validation accuracy reached 1.0, and the run stopped on it. One test
problem, `15 + 98`, still fails: its tens column is `1 + 9` with a carry
of 1, a fact no training problem happened to cover. The validation problems
were used to decide when to stop, so they can no longer give a fair measure
of what was learned; the held-out problems, looked at once and only at the
end, can. This is the reason SSAPling separates collections that guide
training from held-out ones.

## Where to go from here

- [The usage guide](../../docs/usage.md) walks through the same project,
  part by part.
- [Extending SSAPling](../../docs/extending.md) covers each kind of
  component in more depth.
- Change `run.yaml` and see what happens. With `intervene: false`, an
  episode ends at the first column the learner gets wrong, so it learns
  more slowly: 500 episodes instead of 400. Without `teacher: tutor`, it is
  told nothing, never improves, and the run goes on to 2000 episodes.
