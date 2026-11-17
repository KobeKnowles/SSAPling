"""Assessments: run a model over evaluation collections, measure each instance, aggregate.

An **assessment** is a registered component with its own settings. The
built-in ``"standard"`` assessment is driven entirely by its settings: which
collections, which measurements, which aggregations, and how each instance
is attempted. The same assessment runs standalone and inside Train's assess
stage.

    assessment = assessments.create(
        "standard",
        {"collections": ["validation"],
         "aggregations": {"accuracy": {"measurement": "score", "aggregator": "mean"}}},
        collections=collections, task=task,
    )
    results = await assessment.run(model, run_id="assess-1", seed=7)
    results.aggregates["accuracy"].value
"""

import asyncio
import dataclasses
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..core import Registry, setting, settings_to_dict, validate_settings
from ..core._declarations import check_identifier, check_int, keyed, name_of
from ..core._execution import Execution
from ..data import Collections, Drawn, Record, Store
from ..environments import supports as environment_supports
from ..interaction import EpisodeSettings, run_episode, solve
from ..interaction.episode import TeacherContext
from ..models import supports as model_supports
from .aggregation import AggregationSettings, Summary, aggregate, aggregators
from .measurement import InstanceResult, MeasuredValue, Trial, measurements

__all__ = [
    "Assessment",
    "AssessmentResults",
    "AssessmentSettings",
    "StandardAssessment",
    "assessments",
    "load_results",
    "save_results",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessmentResults:
    """What an assessment found.

    Attributes:
        run_id: The run's id; each record's id is ``"<run_id>:<instance id>"``.
        instances: What was found for each instance, in the order assessed.
        aggregates: Each aggregation's summary, keyed by its name.
        provenance: How the results were made, as plain values: the
            assessment's settings, the seed, the model information the
            caller gave, whether a teacher observed or intervened, and how
            many lanes ran.
    """

    run_id: str
    instances: tuple[InstanceResult, ...]
    aggregates: Mapping[str, Summary]
    provenance: Mapping[str, Any]


class Assessment(Protocol):
    """What every assessment provides: a run over a model, giving results."""

    async def run(
        self,
        model: object,
        *,
        run_id: str,
        seed: int,
        model_info: Mapping[str, Any] | None = None,
    ) -> AssessmentResults:
        """Assess ``model`` and return the results."""
        ...


@dataclass(frozen=True)
class AssessmentSettings:
    """Settings for the standard assessment."""

    collections: list[str] = setting(
        min_length=1, description="The collections assessed, served for evaluation."
    )
    measurements: list[str] = setting(
        default_factory=lambda: ["score"],
        min_length=1,
        description="The registered measurements taken of each instance.",
    )
    aggregations: dict[str, AggregationSettings] = setting(
        default_factory=dict, description="The summaries to compute, keyed by name."
    )
    episode: EpisodeSettings = setting(
        default_factory=EpisodeSettings,
        description=(
            "How each instance is attempted: timing, max_steps (a limit per instance), "
            "whether a teacher intervenes, threads."
        ),
    )
    max_instances: int | None = setting(
        None, ge=1, description="Assess at most this many instances, in collection order."
    )
    learning_operation: str | None = setting(
        None,
        min_length=1,
        description=(
            "For learning during evaluation: the learning operation called with each "
            "instance's record. None: the model does not learn."
        ),
    )
    stop_on_failure: bool = setting(
        False, description="Stop when a measurement raises, instead of recording it as failed."
    )

    def validate(self) -> None:
        unmeasured = [
            f"{name!r} ({aggregation.measurement!r})"
            for name, aggregation in self.aggregations.items()
            if aggregation.measurement not in self.measurements
        ]
        if unmeasured:
            raise ValueError(
                f"aggregations {', '.join(unmeasured)} name measurements that are not taken; "
                f"add them to measurements: {self.measurements}"
            )


class StandardAssessment:
    """Runs a model over collections, measures each instance, and aggregates, as set.

    Each instance is attempted with ``run_episode`` when there is an
    environment, and with ``solve`` (static solving) when there is none.

    Args:
        config: The settings.
        collections: Where the collections are; each is requested with
            ``purpose="evaluation"``, so held-out collections are served.
        task: The task, for the measurements, such as ``"score"``.
        env: Optional. The environment, or a list of environments: one per
            lane, run at once. Without one, instances are solved statically.
            The assessment never closes them.
        teacher: Optional. A teacher the episode loop consults. It observes
            unless ``episode.intervene`` is set.
        teacher_context: Optional. As for ``run_episode``.
        outcome: Optional. Turns a record into what the task scores. By
            default: the last prediction (static) or the final observation
            (with an environment), of the one participant, or keyed by
            agent when there are several.
        store: Optional. Where the records are written.

    Raises:
        ConfigError: Invalid settings.
        UnknownComponentError: A measurement or aggregator that is not
            registered.
    """

    Settings = AssessmentSettings

    def __init__(
        self,
        config: AssessmentSettings | Mapping[str, Any],
        *,
        collections: Collections,
        task: object,
        env: object = None,
        teacher: object = None,
        teacher_context: TeacherContext | None = None,
        outcome: Callable[[Record], Any] | None = None,
        store: Store | None = None,
    ) -> None:
        self.config = validate_settings(AssessmentSettings, config)
        self.measures = {name: measurements.get(name) for name in self.config.measurements}
        for aggregation in self.config.aggregations.values():
            aggregators.get(aggregation.aggregator)
        self.collections, self.task, self.store = collections, task, store
        self.envs = [] if env is None else list(env) if isinstance(env, Sequence) else [env]
        self.teacher, self.teacher_context = teacher, teacher_context
        self.outcome = outcome or _last_outcome

    async def run(
        self,
        model: object,
        *,
        run_id: str,
        seed: int,
        model_info: Mapping[str, Any] | None = None,
    ) -> AssessmentResults:
        """Assess ``model`` over the collections and return the results.

        The model is told it is being evaluated (``set_evaluating(True)``,
        then ``False`` at the end), and no learning operation is called
        unless ``learning_operation`` is set.

        Args:
            model: The model, or a mapping from agent id to model, as for
                ``run_episode``.
            run_id: Identifies this run; each record's id is
                ``"<run_id>:<instance id>"``.
            seed: Each instance's reset seed is derived from it and the
                instance's id, so it does not depend on the order or the
                lanes. It is passed only to an environment with seeded reset.
            model_info: Optional. Plain values about the model, kept in the
                results, such as ``{"checkpoint": "ckpt-12",
                "episodes_trained": 41200}``.

        Raises:
            UnknownCollectionError: A collection that does not exist.
            Exception: A model's or environment's own error, as itself;
                the failed record is kept.
        """
        check_identifier(run_id, "a run id")
        check_int(seed, "seed")
        info = keyed(model_info or {}, "model_info", "names")
        models = list({id(each): each for each in _models(model)}.values())
        operation = self.config.learning_operation
        if operation is not None:
            for each in models:
                if not model_supports(each, operation):
                    raise ValueError(f"{name_of(each)} does not support {operation!r}")
        drawn = [
            (name, item)
            for name in self.config.collections
            for item in self.collections.get(name, purpose="evaluation")
        ][: self.config.max_instances]

        await _set_evaluating(models, True)
        try:
            instances = await self._attempt_all(model, models, drawn, run_id, seed)
        finally:
            await _set_evaluating(models, False)
        intervene = self.config.episode.intervene
        teacher = None if self.teacher is None else "intervene" if intervene else "observe"
        return AssessmentResults(
            run_id=run_id,
            instances=tuple(instances),
            aggregates=aggregate(instances, self.config.aggregations),
            provenance={
                "settings": settings_to_dict(self.config),
                "seed": seed,
                "model": info,
                "teacher": teacher,
                "lanes": max(len(self.envs), 1),
            },
        )

    async def _attempt_all(
        self,
        model: object,
        models: list[object],
        drawn: list[tuple[str, Drawn]],
        run_id: str,
        seed: int,
    ) -> list[InstanceResult]:
        """Attempt every instance, in as many lanes as there are environments."""
        results: dict[int, InstanceResult] = {}
        pending = iter(enumerate(drawn))  # shared by the lanes: each takes the next instance
        lanes = self.envs or [None]

        async def lane(env: object, key: int | None) -> None:
            for position, (collection, item) in pending:
                results[position] = await self._attempt(
                    model, models, env, key, collection, item, run_id, seed
                )

        many = len(lanes) > 1
        tasks = [
            asyncio.ensure_future(lane(env, i if many else None)) for i, env in enumerate(lanes)
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        return [results[position] for position in range(len(drawn))]

    async def _attempt(
        self,
        model: object,
        models: list[object],
        env: object,
        lane: int | None,
        collection: str,
        drawn: Drawn,
        run_id: str,
        seed: int,
    ) -> InstanceResult:
        """Attempt one instance, measure it, and let the model learn from it if set."""
        instance, record_id = drawn.instance, f"{run_id}:{drawn.instance.id}"
        if env is None:
            record = await solve(
                model,
                instance,
                record_id=record_id,
                lane=lane,
                provenance=drawn.provenance,
                store=self.store,
                settings=self.config.episode,
            )
        else:
            seeded = environment_supports(env, "seeded_reset")
            record = await run_episode(
                env,
                model,
                instance,
                record_id=record_id,
                seed=_instance_seed(seed, instance.id) if seeded else None,
                teacher=self.teacher,
                teacher_context=self.teacher_context,
                lane=lane,
                provenance=drawn.provenance,
                store=self.store,
                settings=self.config.episode,
            )
        assert record.summary is not None
        if record.summary.status == "failed":  # nothing reliable to measure
            failed = MeasuredValue.failed(f"the attempt failed: {record.summary.detail}")
            values = dict.fromkeys(self.measures, failed)
        else:
            trial = Trial(instance, record, self.outcome(record), self.task)
            values = {
                name: self._measure(measure, trial) for name, measure in self.measures.items()
            }
        if self.config.learning_operation is not None:
            for each in models:
                learn = each.learn  # type: ignore[attr-defined]
                await Execution().call(learn, self.config.learning_operation, record=record)
        return InstanceResult(
            instance_id=instance.id,
            collection=collection,
            descriptors=instance.descriptors,
            record_id=record_id,
            status=record.summary.status,
            values=values,
        )

    def _measure(self, measure: Callable[[Trial], Any], trial: Trial) -> MeasuredValue:
        try:
            return MeasuredValue.of(measure(trial))
        except Exception as error:
            if self.config.stop_on_failure:
                raise
            return MeasuredValue.failed(f"{type(error).__name__}: {error}")


assessments: Registry[Callable[..., Assessment]] = Registry("assessment")
"""Registered assessments. Also available as ``ssapling.registries.assessments``.

``"standard"`` is built in.
"""
assessments.register("standard", StandardAssessment)


# Saving and loading results


def save_results(store: Store, results: AssessmentResults) -> None:
    """Save ``results`` in ``store``, as the document ``"results/<run id>"``."""
    store.save_document(f"results/{results.run_id}", dataclasses.asdict(results))


def load_results(store: Store, run_id: str) -> AssessmentResults:
    """The results saved by ``save_results`` for ``run_id``.

    Raises:
        UnknownDocumentError: The store holds no results for ``run_id``.
    """
    document = store.load_document(f"results/{run_id}")
    instances = tuple(
        InstanceResult(
            **{
                **doc,
                "values": {name: MeasuredValue(**value) for name, value in doc["values"].items()},
            }
        )
        for doc in document["instances"]
    )
    return AssessmentResults(
        run_id=document["run_id"],
        instances=instances,
        aggregates={name: _summary(doc) for name, doc in document["aggregates"].items()},
        provenance=document["provenance"],
    )


# Helpers


def _last_outcome(record: Record) -> Any:
    """What the task scores by default: the last prediction, or the final observation."""
    if not record.steps:
        return None
    last = record.steps[-1]
    if record.header.static:
        values = last.predictions or {}
    else:
        values = last.transition.observations if last.transition is not None else {}
    participants = record.header.participants
    return values.get(participants[0]) if len(participants) == 1 else dict(values)


def _instance_seed(seed: int, instance_id: str) -> int:
    """A reset seed for one instance, from the run's seed and the instance's id alone."""
    return random.Random(f"{seed}:{instance_id}").getrandbits(32)


def _models(model: object) -> list[object]:
    return list(model.values()) if isinstance(model, Mapping) else [model]


async def _set_evaluating(models: list[object], evaluating: bool) -> None:
    for each in models:
        method = getattr(each, "set_evaluating", None)
        if callable(method):
            await Execution().call(method, evaluating)


def _summary(doc: Mapping[str, Any]) -> Summary:
    return Summary(
        value=doc["value"],
        contributed=doc["contributed"],
        excluded=doc["excluded"],
        groups={key: _summary(group) for key, group in doc["groups"].items()},
    )
