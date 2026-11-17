"""Stages: the kinds of work a run does, each a registered component with its own settings.

A **stage** is anything with ``run(context)``, written as ``def`` or
``async def``, registered in ``stages``. It reads what it needs from the
run context and publishes values to it; it never hands anything to another
stage directly. It may also say what it needs, so a procedure that cannot
work is refused before the run starts:

- ``requires``: the components it uses, by role, such as ``("model",)``.
- ``reads``: the published values it reads, by name.
- ``publishes``: the values it publishes, by name.
- ``check(context)``: anything else, such as a learning operation the model
  must support. It raises ``ValueError`` saying what is wrong.

These are read from the stage once it is built, so they may depend on its
settings. Built-in stages and a project's own are treated exactly alike::

    class ReplayAudit:
        requires = ("model", "store")

        async def run(self, context):
            ...

    stages.register("replay_audit", ReplayAudit)

The built-in stages are ``interact``, ``learn_from_data``,
``learn_operation``, ``update_sampler``, ``assess`` and ``checkpoint``.
Each learning call counts one update, and what the model's ``learn``
returns is published under the operation's name.
"""

import random
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..core import Registry, setting
from ..core._declarations import name_of
from ..core._execution import Execution
from ..data import Collections, Record, RecordStep, batches
from ..environments import supports as environment_supports
from ..evaluate import save_results
from ..interaction import EpisodeSettings, run_episode, solve
from ..interaction.episode import OnStep
from ..models import filter_signals
from ..models import supports as model_supports
from ..teachers import supports as teacher_supports
from .context import RunContext

__all__ = ["Stage", "learn", "stages"]


class Stage(Protocol):
    """What every stage provides: ``run(context)``."""

    def run(self, context: RunContext) -> Awaitable[None] | None:
        """Do the stage's work, reading from ``context`` and publishing to it."""
        ...


stages: Registry[Callable[..., Stage]] = Registry("stage")
"""Registered stages. Also available as ``ssapling.registries.stages``."""


# Built-in stages


@dataclass(frozen=True)
class OnlineLearning:
    """Learning as the episodes happen: after each step, or after each episode."""

    operation: str = setting(
        "learn_from_experience", min_length=1, description="The learning operation called."
    )
    after: Literal["step", "episode"] = setting(
        "step",
        description=(
            "step: after each step, with the step and the learning signals the model declares. "
            "episode: after each episode, with its record."
        ),
    )


@dataclass(frozen=True)
class InteractSettings:
    """Settings for the interact stage."""

    episodes: int = setting(1, ge=1, description="How many episodes to run.")
    environment: str | None = setting(
        "environment",
        description="The environment's role. None: static solving, with no environment.",
    )
    teacher: str | None = setting(
        None, description="The role of a teacher to consult at each step. None: no teacher."
    )
    teacher_context: str | None = setting(
        None,
        description=(
            "The role of a function that builds each teacher request's context, as for "
            "run_episode. None: the default context."
        ),
    )
    sampler: str = setting(
        "sampler", min_length=1, description="The role of the sampler that picks each instance."
    )
    collections: str | None = setting(
        "collections",
        description=(
            "The role of the collections whose problems are never trained on: each instance "
            "comes from their remainder. None: the run has none."
        ),
    )
    episode: EpisodeSettings = setting(
        default_factory=EpisodeSettings,
        description="How each episode runs: timing, max_steps, intervene, threads.",
    )
    learn: OnlineLearning | None = setting(
        None, description="Learn as the episodes happen. None: no learning in this stage."
    )


class Interact:
    """Runs episodes of the model, each on a new instance from the training remainder.

    Each episode's instance is drawn from the remainder of the run's
    collections, so it is never a problem of an assessment. Its seeds are
    derived from the root seed and the episode's number, so a resumed run
    draws the same instances. Records go to the run's store, if it has one,
    as ``"episode-<number>"``. Counts episodes and steps.
    """

    Settings = InteractSettings

    def __init__(self, config: InteractSettings) -> None:
        self.config = config
        roles = (config.environment, config.teacher, config.teacher_context, config.collections)
        self.requires = ("model", config.sampler, *(role for role in roles if role))
        self.publishes = (config.learn.operation,) if config.learn else ()

    def check(self, context: RunContext) -> None:
        if self.config.learn is not None:
            _check_learns(context, self.config.learn.operation)
        if self.config.teacher is not None:
            operation = self.config.episode.teacher_operation
            if not teacher_supports(context.components[self.config.teacher], operation):
                raise ValueError(f"the teacher does not support {operation!r}")

    async def run(self, context: RunContext) -> None:
        config = self.config
        model, sampler = context.components["model"], context.components[config.sampler]
        env = _role(context, config.environment)
        collections = _role(context, config.collections)
        if collections is None:  # nothing reserved: draw from the sampler's source alone
            collections = Collections(sampler.source)
        on_step = self._on_step(context)
        for _ in range(config.episodes):
            number = context.counters["episodes"]
            seed = context.seed(f"draw-{number}")
            drawn = next(collections.remainder(sampler=sampler, seed=seed))
            common: dict[str, Any] = {
                "record_id": context.output_name(f"episode-{number}"),
                "provenance": drawn.provenance,
                "store": context.components.get("store"),
                "settings": config.episode,
                "on_step": on_step,
            }
            if env is None:
                record = await solve(model, drawn.instance, **common)
            else:
                seeded = environment_supports(env, "seeded_reset")
                record = await run_episode(
                    env,
                    model,
                    drawn.instance,
                    seed=context.seed(f"episode-{number}") if seeded else None,
                    teacher=_role(context, config.teacher),
                    teacher_context=_role(context, config.teacher_context),
                    **common,
                )
            context.counters["episodes"] += 1
            context.counters["steps"] += len(record.steps)
            if config.learn is not None and config.learn.after == "episode":
                await learn(context, config.learn.operation, record=record)

    def _on_step(self, context: RunContext) -> OnStep | None:
        online = self.config.learn
        if online is None or online.after != "step":
            return None
        model = context.components["model"]
        operation = online.operation

        async def learn_from(step: RecordStep) -> None:
            signals = filter_signals(model, _signals(step))
            await learn(context, operation, step=step, signals=signals)

        return learn_from


@dataclass(frozen=True)
class LearnFromDataSettings:
    """Settings for the learn_from_data stage."""

    source: str = setting(
        min_length=1,
        description=(
            "The role of what the items come from: a store, whose finished records are the "
            "items, except attempts at instances of a collection; or any finite collection of "
            "items, such as a list of records or a collection's instances."
        ),
    )
    operation: str = setting(
        "learn_from_batch", min_length=1, description="The learning operation called per batch."
    )
    batches: int = setting(1, ge=1, description="How many batches: one update each.")
    batch_size: int = setting(
        32, ge=1, description="Items per batch, drawn at random, with replacement."
    )
    collator: str | None = setting(
        None,
        description=(
            "The role of a collator, which turns a list of items into a batch. None: the batch "
            "is the list."
        ),
    )
    workers: int = setting(0, ge=0, description="Worker processes that collate; 0 collates here.")


class LearnFromData:
    """Draws batches from a data source and calls a learning operation with each.

    Items are drawn at random, with replacement, with a seed derived from
    the root seed and the updates so far. Each batch is passed as
    ``learn(operation, batch=...)`` and counts one update.
    """

    Settings = LearnFromDataSettings

    def __init__(self, config: LearnFromDataSettings) -> None:
        self.config = config
        self.requires = ("model", config.source, *([config.collator] if config.collator else []))
        self.publishes = (config.operation,)

    def check(self, context: RunContext) -> None:
        _check_learns(context, self.config.operation)
        source = context.components[self.config.source]
        if not (_is_store(source) or isinstance(source, Iterable)):
            raise ValueError(f"{self.config.source!r} is neither a store nor a collection of items")

    async def run(self, context: RunContext) -> None:
        config = self.config
        items = _items(context.components[config.source])
        if not items:
            raise ValueError(f"{config.source!r} has no items to learn from")
        rng = random.Random(context.seed(f"batches-{context.counters['updates']}"))
        drawn = (rng.choice(items) for _ in range(config.batches * config.batch_size))
        collate = context.components[config.collator] if config.collator else list
        for batch in batches(drawn, collate, batch_size=config.batch_size, workers=config.workers):
            await learn(context, config.operation, batch=batch)


@dataclass(frozen=True)
class LearnOperationSettings:
    """Settings for the learn_operation stage."""

    operation: str = setting(
        min_length=1, description="The learning operation, such as 'consolidate'."
    )
    payload: dict[str, Any] = setting(
        default_factory=dict,
        description="Its arguments, as plain values, such as {limits: {seconds: 600}}.",
    )


class LearnOperation:
    """Calls one learning operation, such as a night-cycle consolidation, once.

    It is called as ``learn(operation, **payload)``, and counts one update.
    """

    Settings = LearnOperationSettings
    requires = ("model",)

    def __init__(self, config: LearnOperationSettings) -> None:
        self.config = config
        self.publishes = (config.operation,)

    def check(self, context: RunContext) -> None:
        _check_learns(context, self.config.operation)

    async def run(self, context: RunContext) -> None:
        await learn(context, self.config.operation, **self.config.payload)


@dataclass(frozen=True)
class UpdateSamplerSettings:
    """Settings for the update_sampler stage."""

    weights: dict[str, float] = setting(
        description=(
            "The sampler's new weights, keyed by value or stratum as a string, as for the "
            "built-in samplers."
        )
    )
    sampler: str = setting("sampler", min_length=1, description="The role of the sampler.")


class UpdateSampler:
    """Gives a sampler new weights, for a curriculum's next step.

    Usually triggered with ``when``, by a condition on a published value.
    """

    Settings = UpdateSamplerSettings

    def __init__(self, config: UpdateSamplerSettings) -> None:
        self.config = config
        self.requires = (config.sampler,)

    def check(self, context: RunContext) -> None:
        if not callable(getattr(context.components[self.config.sampler], "set_weights", None)):
            raise ValueError(f"the sampler {self.config.sampler!r} has no set_weights method")

    def run(self, context: RunContext) -> None:
        context.components[self.config.sampler].set_weights(self.config.weights)


@dataclass(frozen=True)
class AssessSettings:
    """Settings for the assess stage."""

    assessment: str = setting(
        min_length=1,
        description=(
            "The role of the assessment to run. Its aggregates are published under this name."
        ),
    )


class Assess:
    """Runs an assessment of the model, saves the results, and publishes the aggregates.

    The assessment's run id is ``"<role>-<stages>"``, and its seed is
    derived from the root seed and the role, so each time it runs it is
    given the same seed. The results' model information is the run's
    counters, and they are saved in the run's store, if it has one. The
    aggregates' values are published under the role, as
    ``{"accuracy": 0.83, ...}``.
    """

    Settings = AssessSettings

    def __init__(self, config: AssessSettings) -> None:
        self.config = config
        self.requires = ("model", config.assessment)
        self.publishes = (config.assessment,)

    async def run(self, context: RunContext) -> None:
        role = self.config.assessment
        results = await Execution().call(
            context.components[role].run,
            context.components["model"],
            run_id=context.output_name(f"{role}-{context.counters['stages']}"),
            seed=context.seed(f"assessment-{role}"),
            model_info=dict(context.counters),
        )
        store = context.components.get("store")
        if store is not None:
            save_results(store, results)
        aggregates = {name: summary.value for name, summary in results.aggregates.items()}
        context.publish(role, aggregates)


class Checkpoint:
    """Saves the run, so it can be resumed from here.

    It saves, as the store document ``"checkpoints/<stages>"``, the state of
    every component that exports it, the controller's, and the run
    context's: its counters, running time, published values and decisions.
    """

    requires = ("store",)

    async def run(self, context: RunContext) -> None:
        states = {}
        for role, component in context.components.items():
            export = getattr(component, "export_state", None)
            if callable(export):
                states[role] = await Execution().call(export)
        export = getattr(context.controller, "export_state", None)
        document = {
            "context": context.export_state(),
            "decisions": context.decisions,
            "controller": await Execution().call(export) if callable(export) else None,
            "components": states,
        }
        name = context.output_name(f"checkpoints/{context.counters['stages']}")
        context.components["store"].save_document(name, document)


stages.register("interact", Interact)
stages.register("learn_from_data", LearnFromData)
stages.register("learn_operation", LearnOperation)
stages.register("update_sampler", UpdateSampler)
stages.register("assess", Assess)
stages.register("checkpoint", Checkpoint)


# Helpers


async def learn(context: RunContext, operation: str, **payload: Any) -> Any:
    """Call the run's model's learning operation ``operation`` with ``payload``, count the update,
    and publish what it returns, its statistics, under the operation's name; then return them.

    The built-in stages learn through it, and a downstream project's stages can too, so every
    stage counts and publishes its updates alike. A model's ``learn`` may be ``async def``.
    Nothing is published when it returns ``None``.
    """
    stats = await Execution().call(context.components["model"].learn, operation, **payload)
    context.counters["updates"] += 1
    if stats is not None:
        context.publish(operation, stats)
    return stats


def _role(context: RunContext, role: str | None) -> Any:
    """The component with ``role``, or ``None`` when no role is set."""
    return None if role is None else context.components[role]


def _check_learns(context: RunContext, operation: str) -> None:
    model = context.components["model"]
    if not model_supports(model, operation):
        raise ValueError(f"the model, {name_of(model)}, does not support {operation!r}")


def _signals(step: RecordStep) -> dict[str, Any]:
    """A step's learning signals, by name and then by agent: rewards and the teacher's answers."""
    signals: dict[str, Any] = {}
    if step.transition is not None and step.transition.rewards is not None:
        signals["reward"] = dict(step.transition.rewards)
    for response in step.teacher_responses:
        if response.status == "answered":
            for kind, by_agent in response.contributions.items():
                signals.setdefault(kind, {}).update(by_agent)
    return signals


def _is_store(source: object) -> bool:
    return callable(getattr(source, "record_ids", None)) and callable(getattr(source, "read", None))


def _items(source: Any) -> list[Any]:
    """A store's finished records, except attempts at instances of a collection; or the items."""
    if not _is_store(source):
        return list(source)
    records: Iterable[Record] = (source.read(record_id) for record_id in source.record_ids())
    return [
        record
        for record in records
        if record.finished
        and (record.header.provenance is None or record.header.provenance.collection is None)
    ]
