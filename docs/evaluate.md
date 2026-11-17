# Evaluate

`ssapling.evaluate` defines how an assessment is composed and run: which
instances a model faces, what is measured on each, and how the measurements
are combined. SSAPling provides the machinery and a few generic parts (a
`score` measurement, `count` and `mean` aggregators, one settings-driven
assessment). Which assessments, measurements and aggregators a project uses
is its own.

| Part | What it is |
| --- | --- |
| [Assessments](#assessments) | `assessments.create("standard", settings, ...)`, then `await assessment.run(model, ...)`. |
| [Measurements](#measurements) | Functions that take a `Trial` and return one value per instance, with a status. |
| [Aggregation](#aggregation) | Combining values into a `Summary`, with exclusions counted and empty aggregates undefined. |
| [Results](#results) | `AssessmentResults`: per instance, aggregated, and how they were made. Saved and loaded through a `Store`. |

## A worked example

Assess an adding model on the `"validation"` collection, statically (no
environment), and report its accuracy, overall and by number of digits:

```python
from ssapling.evaluate import assessments

assessment = assessments.create(
    "standard",
    {
        "collections": ["validation"],
        "aggregations": {
            "accuracy": {"measurement": "score", "aggregator": "mean"},
            "by_digits": {"measurement": "score", "aggregator": "mean", "group_by": "digits"},
        },
    },
    collections=collections, task=task, store=store,
)
results = await assessment.run(model, run_id="assess-1", seed=7, model_info={"checkpoint": "ckpt-12"})

results.aggregates["accuracy"]
# Summary(value=0.83, contributed=6, excluded={}, groups={})
results.aggregates["by_digits"].groups["2"].value
results.instances[0].values["score"]        # MeasuredValue(status="available", value=1.0)
```

## Assessments

An **assessment** is a registered component: anything with
`async run(model, *, run_id, seed, model_info=None)` returning
`AssessmentResults`, registered in `assessments`. The built-in
`"standard"` assessment does everything from its settings:

| Setting | Meaning | Default |
| --- | --- | --- |
| `collections` | The collections assessed, in order. Each is requested with `purpose="evaluation"`, so held-out collections are served. | required |
| `measurements` | The registered measurements taken of each instance. | `["score"]` |
| `aggregations` | The summaries to compute, keyed by name (see [Aggregation](#aggregation)). | none |
| `episode` | How each instance is attempted: the [episode loop's settings](interaction.md#settings), such as timing, `max_steps` (a limit per instance), and whether a teacher intervenes. | the loop's defaults |
| `max_instances` | Assess at most this many instances. | all |
| `learning_operation` | For learning during evaluation: the learning operation called with each instance's record. | none: the model does not learn |
| `stop_on_failure` | Stop when a measurement raises, instead of recording its value as failed. | `False` |

The live parts are passed when it is made: `collections` and `task`, and
optionally `env`, `teacher`, `teacher_context`, `outcome` and `store`.

**What `run` does:**

1. Tells the model it is being evaluated: `set_evaluating(True)`, and
   `False` at the end, if it has the method.
2. Attempts each instance, recording it under `"<run_id>:<instance id>"`,
   with the instance's provenance in the record's header:
   - **With an environment**, through the [episode loop](interaction.md#the-episode-loop).
     Its reset seed is derived from the run's seed and the instance's id,
     so it does not depend on the order or the lanes. It is passed only to
     an environment with seeded reset.
   - **Without one**, by [static solving](interaction.md#static-solving).
3. Works out the **outcome**, what the task scores. By default it is the
   last prediction (static) or the final observation (with an
   environment): of the one participant, or keyed by agent when there are
   several. Turning a record into an outcome is often specific to a task
   and an environment, so `outcome=` takes your own function:

   ```python
   def entered_number(record):          # the digit pad: what was entered, as a number
       entered = record.steps[-1].transition.observations["student"]["entered"]
       return int(entered[::-1])
   ```
4. Takes each measurement.
5. Calls no learning operation, unless `learning_operation` is set.
6. Aggregates.

**Lanes.** Give `env` a list of environments to attempt that many instances
at once. The results are the same as with one environment, in the same
order. The assessment never closes the environments it is given.

**A teacher** observes unless the episode settings say `intervene`: its
responses are recorded, never applied. The results say which.

**When an attempt fails** because the environment could not be reached,
each of its measurements is `failed` and the assessment goes on. A model's
own error stops the assessment, as itself; its record is kept as failed.

## Measurements

A **measurement** is a function from a `Trial` to a value, registered by
name:

```python
from ssapling.evaluate import measurements

def wrong_moves(trial):                      # from the whole trajectory
    return sum(step.transition.info.get("correct") is False for step in trial.record.steps)

def tokens(trial):                           # from what the model reported
    return sum(step.diagnostics["student"]["tokens"] for step in trial.record.steps)

measurements.register("wrong_moves", wrong_moves)
measurements.register("tokens", tokens)
```

A `Trial` has the `instance` (reference included), its `record` (every
step: observations, actions, teacher responses, timing, and the model's
[diagnostics](models.md#diagnostics)), the `outcome`, and the `task`. The
built-in `"score"` is `task.score(trial.instance, trial.outcome)`.

**Every value carries a status**, so a missing value is never mistaken for
zero:

| Status | Meaning | What a measurement returns |
| --- | --- | --- |
| `available` | Computed here. | Any value, or a task's local `Score`. |
| `external` | Reported by a service, such as an ARC-AGI-3 scorecard, kept as reported. | `MeasuredValue.external(value)`, or a task's external `Score`. |
| `unavailable` | Cannot be computed, for example because no reference is accessible. | `None`, `MeasuredValue.unavailable(reason)`, or an unavailable `Score`. |
| `failed` | The measurement raised (the error is kept), or the attempt failed. | It raises. |

A measurement has no settings: a variant is a function of its own, under its
own name.

## Aggregation

An **aggregator** is a function over the list of contributing values,
registered by name in `aggregators`. `"count"` and `"mean"` are built in.
An **aggregation** names a measurement, an aggregator, and optionally a
descriptor to group by, the instances it covers, and how its groups are
weighted:

```python
{"measurement": "score", "aggregator": "mean"}                         # pooled
{"measurement": "score", "aggregator": "mean", "group_by": "digits"}   # and per digit count
{"measurement": "score", "aggregator": "mean", "exclude_truncated": True}
{"measurement": "score", "aggregator": "mean", "where": {"group": "broad"}}
{"measurement": "score", "aggregator": "mean", "group_by": "digits", "weighting": "equal"}
```

| Setting | Means | Default |
| --- | --- | --- |
| `measurement` | The measurement whose values are combined. | required |
| `aggregator` | The registered aggregator, such as `"mean"`. | required |
| `group_by` | A descriptor: also summarise each of its values separately. | none |
| `exclude_truncated` | Leave out the values of truncated attempts, counted under `"truncated"`. | false |
| `where` | Descriptor values an instance must have to be aggregated, as for `Collections.build`: it has every descriptor named, each equal to the value given (a list equals a tuple with the same items). | `{}`: every instance |
| `weighting` | `"pooled"`: every instance weighs the same. `"equal"`: the value is the mean of the groups' values, every group weighing the same. Needs `group_by`. | `"pooled"` |

The machinery, not the aggregator, decides what contributes:

- **Only instances that match `where` are aggregated**, in the value and
  in every group. The others are in no count: neither contributed nor
  excluded.
- **Unavailable and failed values are left out**, never counted as zero,
  and `excluded` says how many, by reason.
- **Truncated attempts count**, since the task scored whatever outcome was
  reached, unless the aggregation excludes them; then they are counted
  under `"truncated"`.
- **An aggregate over no values is `None`**: undefined, not 0.

The result is a `Summary(value, contributed, excluded, groups)`. With
`group_by`, `value` is still the pooled aggregate, in which every instance
weighs the same, and `groups` holds a summary for each value of the
descriptor, keyed as a string (`"1"`, `"2"`; `"None"` for instances
without it). With `weighting: equal` as well, `value` is the mean of the
groups' values instead: a group with no value (`None`) is left out of that
mean and still reported, and with no group value at all, `value` is
`None`. `contributed` and `excluded` are then the totals over the groups.
For example, 2 of 3 correct at one digit and 0 of 1 at two digits is 0.5
pooled and (2/3 + 0) / 2 = 1/3 with equal weight.

`aggregate(instances, aggregations)` needs only per-instance results, so
summaries can be recomputed, or new ones made, from stored results without
running the model again.

## Results

```python
AssessmentResults(
    run_id="assess-1",
    instances=(InstanceResult(instance_id, collection, descriptors, record_id, status, values), ...),
    aggregates={"accuracy": Summary(...)},
    provenance={"settings": {...}, "seed": 7, "model": {"checkpoint": "ckpt-12"},
                "teacher": "observe", "lanes": 1},
)
```

- Each `InstanceResult` keeps the instance's id, collection and
  descriptors, how its attempt ended (`status`: terminated, truncated, or
  failed), its measured values, and its record's id, to read the record
  back from the store.
- `provenance` says how the results were made. `model` is whatever the
  caller passed as `model_info`, such as a checkpoint or how much training
  the model had; inside a training run, Train fills it in with the run's
  counters.
- `save_results(store, results)` saves them as the store document
  `"results/<run id>"`, and `load_results(store, run_id)` loads them back
  equal.

## Errors

| Error | Means |
| --- | --- |
| `ConfigError` | Invalid settings, such as an aggregation over a measurement that is not taken. |
| `UnknownComponentError` | A measurement or aggregator that is not registered. Raised when the assessment is made. |
| `UnknownCollectionError` | A collection that does not exist. Raised by `run`. |
| `ValueError` from `run` | `learning_operation` names an operation the model does not support. |
| `UnknownDocumentError` from `load_results` | No results were saved under that run id. |
