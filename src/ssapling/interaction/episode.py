"""Running attempts: the episode loop and static solving, each recorded step by step.

``run_episode`` resets an environment with an instance's inputs, then, step
by step: asks the model controlling each agent whose turn it is for an
action, consults the teacher if there is one, steps the environment, and
appends the step to the record. ``solve`` is the same without an
environment: it presents the instance's inputs to a model and records its
predictions. Train and Evaluate both run attempts through these, so an
attempt behaves the same in training and assessment.

How long the loop waits for a model's output is the **timing mode**,
measured on a **clock**. On the simulated clock, time is counted in calls to
the model's ``advance``, made by the loop, so runs are reproducible. On the
wall clock, it is counted in seconds.
"""

import asyncio
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from ..core import setting, validate_settings
from ..core._declarations import check_int, name_of
from ..core._execution import Execution
from ..data import (
    AgentTiming,
    Clock,
    LateOutput,
    Provenance,
    Record,
    RecordBuilder,
    RecordHeader,
    RecordStep,
    RecordSummary,
    Store,
    TeacherPlacement,
)
from ..environments import (
    EnvironmentCommunicationError,
    EnvironmentRunner,
    ResetResult,
    StepResult,
    agents,
    next_agents,
)
from ..models import NOTHING, as_buffered
from ..tasks import Instance
from ..teachers import TeacherRequest, TeacherResponse, TeacherRunner

__all__ = ["EpisodeSettings", "run_episode", "solve"]

Role = Literal["environment", "model", "teacher"]

TeacherContext = Callable[[Instance, Mapping[str, Any]], Mapping[str, Any]]
"""Builds a teacher request's context from the instance and the acting agents' observations."""

OnStep = Callable[[RecordStep], Any]
"""Called with each step of an attempt once it is recorded; it may be ``async def``."""

_POLL_INTERVAL = 0.001
"""Seconds between polls of a model that works on its own, on the wall clock."""


@dataclass(frozen=True)
class EpisodeSettings:
    """How an attempt is run. Plain values, so a run file can set them.

    ``solve`` uses the timing settings and ``threads``; the others are for
    episodes with an environment.
    """

    timing: Literal["turn_based", "deadline", "real_time"] = setting(
        "turn_based",
        description=(
            "How long a model has to act. turn_based: as long as it needs. deadline: up to "
            "budget, acting as soon as it has an output. real_time: exactly budget, a tick, "
            "acting at its end."
        ),
    )
    clock: Clock = setting(
        "simulated",
        description=(
            "simulated: time is counted in calls to the model's advance, made by the loop, so "
            "runs are reproducible. wall: time is counted in seconds."
        ),
    )
    budget: float | None = setting(
        None,
        gt=0,
        description=(
            "For deadline and real_time: the time a model has per action, in calls to advance "
            "on the simulated clock or seconds on the wall clock."
        ),
    )
    late_outputs: Literal["discard", "use"] = setting(
        "discard",
        description=(
            "What to do with an output that arrives after its deadline: discard it, or use it "
            "as the agent's next action."
        ),
    )
    default_actions: dict[str, Any] = setting(
        default_factory=dict,
        description=(
            "Each agent's action when its model gives none in time. An agent not listed is left "
            "out of the step, so the environment applies its own default."
        ),
    )
    max_steps: int | None = setting(
        None, ge=1, description="Cut the episode off after this many steps, as truncated."
    )
    intervene: bool = setting(
        False,
        description=(
            "Apply the teacher's interventions to the actions. Otherwise the teacher only "
            "observes: its responses are recorded, not applied."
        ),
    )
    teacher_operation: str = setting(
        "review_action", min_length=1, description="The operation asked of the teacher each step."
    )
    threads: list[Role] = setting(
        default_factory=list,
        description=(
            "The components whose synchronous calls run in a worker thread of their own, so "
            "they do not block the loop and can have a timeout: 'environment', 'model', "
            "'teacher'."
        ),
    )
    environment_timeout: float | None = setting(
        None, gt=0, description="Seconds to wait for the environment's reset or step."
    )
    teacher_timeout: float | None = setting(
        None, gt=0, description="Seconds to wait for the teacher; then its response is cancelled."
    )

    def validate(self) -> None:
        if (self.timing == "turn_based") != (self.budget is None):
            raise ValueError("deadline and real_time timing need a budget; turn_based has none")
        if self.clock == "simulated" and self.budget is not None and self.budget % 1:
            raise ValueError(
                f"on the simulated clock the budget counts calls to advance, so it must be a "
                f"whole number, not {self.budget}"
            )


async def run_episode(
    env: object,
    model: object,
    instance: Instance,
    *,
    record_id: str,
    seed: int | None = None,
    teacher: object | None = None,
    teacher_context: TeacherContext | None = None,
    teacher_placement: TeacherPlacement | None = None,
    lane: Hashable | None = None,
    provenance: Provenance | None = None,
    store: Store | None = None,
    settings: EpisodeSettings | Mapping[str, Any] | None = None,
    on_step: OnStep | None = None,
) -> Record:
    """Run one episode of ``env`` on ``instance`` and return its record.

    Example::

        record = await run_episode(env, model, instance, record_id="ep-1", seed=7)
        record = await run_episode(
            env, {"player_1": alice, "player_2": bob}, instance, record_id="ep-2",
            teacher=tutor, store=store, settings={"timing": "deadline", "budget": 3},
        )

    The environment is reset with ``options=instance.inputs`` (nothing, if
    the inputs are ``None``), so the instance's reference never reaches it.
    The models' and the teacher's episode state is reset too. Then, until
    the episode is over:

    1. The agents whose turn it is (``environments.next_agents``) each give
       an action: their model perceives the agent's latest observation
       (``None`` if it has had none), and the loop waits for its output, as
       the timing mode allows. With no output in time, the agent's default
       action is used, or, without one, it is left out of the step.
    2. If there is a teacher, it is asked to review the proposals. Its
       interventions replace actions only with ``intervene``.
    3. The environment steps, and the step is appended to the record. What
       was executed, and any teacher responses, come from the environment's
       report when it gives them, and otherwise from the loop.

    The summary is the environment's: terminated or truncated. With
    ``max_steps``, an episode still going after that many steps is
    truncated, with ``"step limit"`` as the detail.

    Args:
        env: The environment.
        model: The model controlling every participant, or a mapping from
            each participant's agent id to its model. A model that controls
            several participants is given each one's id as ``key``.
        instance: What the episode attempts.
        record_id: The record's id.
        seed: Optional. Passed to the environment's reset.
        teacher: Optional. A teacher the loop consults at each step, so its
            placement is ``"coordinator"``.
        teacher_context: Optional. Builds each request's context from the
            instance and the acting agents' observations. By default the
            context is ``{"instance": instance, "observations": ...}``.
        teacher_placement: For the record's header, where a teacher the loop
            does not consult sits: ``"environment"`` (the environment owns
            it) or ``"participant"`` (it is one of the models).
        lane: Optional. For several episodes at once that share a model:
            each episode's lane, given to the model as its ``key`` (with the
            agent id, as ``(lane, agent)``, when it controls several
            participants), so each keeps its own episode state.
        provenance: Optional. Where the instance came from, for the header.
        store: Optional. Where the record is written, step by step.
        settings: ``EpisodeSettings``, or a mapping of them.
        on_step: Optional. Called with each step once it is recorded, before
            the next one, for example to learn online. It may be ``async
            def``.

    Returns:
        The record. When the environment cannot be reached during a step,
        the record says so: that step has no transition, and the summary is
        ``failed``.

    Raises:
        Exception: Whatever else goes wrong, such as a model's error or the
            environment's own, as itself. A record already begun is first
            finished as ``failed``, with the error as the reason, so it is
            kept.
        ValueError: A participant with no model; a teacher with a placement
            other than ``"coordinator"``; or a timeout for a synchronous
            component that does not run in a thread, which could never
            interrupt it. Raised before any record is begun. Also an agent
            named to act that is not a participant.
        TypeError: A model without ``advance`` on the simulated clock.
    """
    settings = validate_settings(EpisodeSettings, settings)
    if teacher is not None and teacher_placement not in (None, "coordinator"):
        raise ValueError(
            f"the loop consults a teacher only as 'coordinator', not {teacher_placement!r}; a "
            "teacher in the environment or controlling a participant is not passed as teacher"
        )
    attempt = _Attempt(
        settings, lane, on_step, env=env, teacher=teacher, teacher_context=teacher_context
    )
    try:
        return await attempt.interact(
            model,
            instance,
            record_id=record_id,
            seed=seed,
            placement="coordinator" if teacher is not None else teacher_placement,
            provenance=provenance,
            store=store,
        )
    finally:
        attempt.close()


async def solve(
    model: object,
    instance: Instance,
    *,
    record_id: str,
    attempts: int = 1,
    participant: str = "solver",
    lane: Hashable | None = None,
    provenance: Provenance | None = None,
    store: Store | None = None,
    settings: EpisodeSettings | Mapping[str, Any] | None = None,
    on_step: OnStep | None = None,
) -> Record:
    """Present ``instance``'s inputs to ``model``, with no environment, and record its predictions.

    Example::

        record = await solve(model, puzzle, record_id="p-1", attempts=2)
        record.steps[0].predictions["solver"]

    Each attempt is one step: the model perceives the whole of
    ``instance.inputs``, never the reference, and the loop waits for its
    prediction, as the timing mode allows. A prediction that does not come
    in time is left out of the step, and the step's timing says so. The
    summary is ``terminated``.

    Args:
        model: The model.
        instance: What is solved.
        record_id: The record's id.
        attempts: How many times the inputs are presented: one step each.
        participant: The model's agent id in the record.
        lane: Optional. As for ``run_episode``.
        provenance: Optional. Where the instance came from, for the header.
        store: Optional. Where the record is written, step by step.
        settings: ``EpisodeSettings``, or a mapping of them. The timing
            settings and ``threads`` apply.
        on_step: Optional. As for ``run_episode``.

    Raises:
        As for ``run_episode``.
    """
    settings = validate_settings(EpisodeSettings, settings)
    check_int(attempts, "attempts", minimum=1)
    attempt = _Attempt(settings, lane, on_step)
    try:
        return await attempt.present(
            model,
            instance,
            record_id=record_id,
            attempts=attempts,
            participant=participant,
            provenance=provenance,
            store=store,
        )
    finally:
        attempt.close()


class _Attempt:
    """What one attempt's steps share: the components, how each is called, and its timing."""

    def __init__(
        self,
        settings: EpisodeSettings,
        lane: Hashable | None,
        on_step: OnStep | None,
        *,
        env: object = None,
        teacher: object = None,
        teacher_context: TeacherContext | None = None,
    ) -> None:
        self.settings, self.lane, self.on_step = settings, lane, on_step
        self.env, self.teacher = env, teacher
        self.context = teacher_context or _default_context
        self.threads = {role: ThreadPoolExecutor(max_workers=1) for role in settings.threads}

        def placement(role: Role) -> dict[str, Any]:
            if role in self.threads:
                return {"placement": "thread", "executor": self.threads[role]}
            return {}

        self.environment = EnvironmentRunner(**placement("environment"))
        self.teachers = TeacherRunner(**placement("teacher"))
        self.teacher_calls = Execution(**placement("teacher"))
        self.model_calls = Execution(**placement("model"))  # advance and reset_episode
        self.direct = Execution()  # perceive and poll, which return at once
        self.controllers: dict[str, tuple[Any, Hashable | None]] = {}  # agent: model, key
        self.owed: Counter[str] = Counter()  # each agent's observations not yet answered
        self.advancing: dict[str, asyncio.Task[Any]] = {}  # wall clock: advance calls running
        # The teacher is first asked mid-episode, so a timeout that could never
        # interrupt it is refused now. The environment's is refused by its reset.
        respond = getattr(teacher, "respond", None)
        if settings.teacher_timeout is not None and callable(respond):
            _check_interruptible(self.teacher_calls, settings.teacher_timeout, respond, "teacher")

    def close(self) -> None:
        for task in self.advancing.values():
            task.cancel()
        for thread in self.threads.values():  # a call past its time may still be running
            thread.shutdown(wait=False, cancel_futures=True)

    # The two kinds of attempt

    async def interact(
        self,
        model: object,
        instance: Instance,
        *,
        record_id: str,
        seed: int | None,
        placement: TeacherPlacement | None,
        provenance: Provenance | None,
        store: Store | None,
    ) -> Record:
        reset = await self.environment.reset(
            self.env, seed=seed, options=instance.inputs, timeout=self.settings.environment_timeout
        )
        participants = agents(self.env, reset)
        self._control(model, participants)
        header = RecordHeader(
            id=record_id,
            instance_id=instance.id,
            participants=participants,
            reset=reset,
            provenance=provenance,
            seeds={} if seed is None else {"reset": seed},
            clock=self.settings.clock,
            teacher_placement=placement,
        )
        async with self._recording(header, store) as writer:
            return await self._play(instance, reset, writer)

    async def present(
        self,
        model: object,
        instance: Instance,
        *,
        record_id: str,
        attempts: int,
        participant: str,
        provenance: Provenance | None,
        store: Store | None,
    ) -> Record:
        self._control(model, (participant,))
        header = RecordHeader(
            id=record_id,
            instance_id=instance.id,
            participants=(participant,),
            provenance=provenance,
            clock=self.settings.clock,
        )
        async with self._recording(header, store) as writer:
            for index in range(attempts):
                output, timing, report = await self._decide(participant, instance.inputs)
                await writer.append(
                    RecordStep(
                        index=index,
                        acting=(participant,),
                        presented={participant: instance.inputs},
                        predictions={} if output is NOTHING else {participant: output},
                        timing={participant: timing},
                        diagnostics={} if report is None else {participant: report},
                    )
                )
            return writer.finish(RecordSummary.terminated())

    @asynccontextmanager
    async def _recording(
        self, header: RecordHeader, store: Store | None
    ) -> AsyncIterator["_Writer"]:
        """Begin the record and reset episode state; if what follows fails, keep the record."""
        writer = _Writer(header, store, self.on_step)
        try:
            for controller, key in self.controllers.values():
                if callable(getattr(controller, "reset_episode", None)):
                    await _call(self.model_calls, controller.reset_episode, key)
            if callable(getattr(self.teacher, "reset", None)):
                await self.teacher_calls.call(self.teacher.reset)  # type: ignore[attr-defined]
            yield writer
        except BaseException as error:
            if writer.record is None:  # keep what happened, marked failed
                writer.finish(RecordSummary.failed(f"{type(error).__name__}: {error}"))
            raise

    async def _play(self, instance: Instance, reset: ResetResult, writer: "_Writer") -> Record:
        observations = dict(reset.observations)  # each agent's latest
        last: ResetResult | StepResult = reset
        index = 0
        while True:
            if self.settings.max_steps is not None and index == self.settings.max_steps:
                return writer.finish(RecordSummary.truncated("step limit"))
            acting = next_agents(self.env, last)
            unknown = [agent for agent in acting if agent not in self.controllers]
            if unknown:
                raise ValueError(f"{name_of(self.env)} names {unknown} to act: not participants")
            seen = {agent: observations.get(agent) for agent in acting}  # None if never observed
            decided = await asyncio.gather(*(self._decide(agent, seen[agent]) for agent in acting))
            proposals: dict[str, Any] = {}
            actions: dict[str, Any] = {}  # the proposals, and default actions where none came
            timing: dict[str, AgentTiming] = {}
            diagnostics: dict[str, Any] = {}
            for agent, (output, when, report) in zip(acting, decided, strict=True):
                timing[agent] = when
                if report is not None:
                    diagnostics[agent] = report
                if output is not NOTHING:
                    proposals[agent] = actions[agent] = output
                elif agent in self.settings.default_actions:
                    actions[agent] = self.settings.default_actions[agent]

            responses: tuple[TeacherResponse, ...] = ()
            if self.teacher is not None:
                request = TeacherRequest(
                    id=f"{writer.id}:{index}",
                    operation=self.settings.teacher_operation,
                    context=self.context(instance, seen),
                    proposals=proposals,
                    episode=writer.id,
                    step=index,
                )
                response = await self.teachers.request(
                    self.teacher,  # type: ignore[arg-type]
                    request,
                    timeout=self.settings.teacher_timeout,
                )
                responses = (response,)
                if self.settings.intervene:
                    actions = _intervene(actions, response)

            try:
                result = await self.environment.step(
                    self.env, actions, timeout=self.settings.environment_timeout
                )
            except EnvironmentCommunicationError as error:
                await writer.append(
                    RecordStep(
                        index=index,
                        acting=acting,
                        proposals=proposals,
                        teacher_responses=responses,
                        timing=timing,
                        diagnostics=diagnostics,
                    )
                )
                return writer.finish(RecordSummary.failed(str(error)))
            await writer.append(
                RecordStep(
                    index=index,
                    acting=acting,
                    proposals=proposals,
                    teacher_responses=responses + tuple(result.teacher_responses or ()),
                    executed_actions=(
                        actions if result.executed_actions is None else result.executed_actions
                    ),
                    transition=result,
                    timing=timing,
                    diagnostics=diagnostics,
                )
            )
            if result.episode_over:
                ended = (
                    RecordSummary.terminated
                    if result.episode_terminated
                    else RecordSummary.truncated
                )
                return writer.finish(ended())
            observations.update(result.observations)
            last = result
            index += 1

    # Models and timing

    def _control(self, model: object, participants: tuple[str, ...]) -> None:
        """Find each participant's model, seen through buffers, and the key it is called with."""
        chosen: Mapping[str, Any] = (
            model if isinstance(model, Mapping) else dict.fromkeys(participants, model)
        )
        missing = [agent for agent in participants if agent not in chosen]
        if missing:
            names = ", ".join(repr(agent) for agent in missing)
            raise ValueError(f"no model controls {names}; give a model for every participant")
        buffered = {id(each): as_buffered(each) for each in chosen.values()}
        for controller in buffered.values():
            advance = getattr(controller, "advance", None)
            if not callable(advance):
                if self.settings.clock == "simulated":
                    raise TypeError(
                        f"{name_of(controller)} has perceive and poll but no advance, so it "
                        "cannot be driven on the simulated clock; use the wall clock"
                    )
            elif self.settings.clock == "wall" and self.settings.budget is not None:
                _check_interruptible(self.model_calls, self.settings.budget, advance, "model")

        serving = Counter(id(chosen[agent]) for agent in participants)

        def key(agent: str) -> Hashable | None:
            shared = serving[id(chosen[agent])] > 1
            if self.lane is None:
                return agent if shared else None
            return (self.lane, agent) if shared else self.lane

        self.controllers = {
            agent: (buffered[id(chosen[agent])], key(agent)) for agent in participants
        }

    async def _decide(self, agent: str, observation: Any) -> tuple[Any, AgentTiming, Any]:
        """Give ``agent``'s model the observation, and wait for its output as timing allows.

        Returns the output, or ``NOTHING`` when none came in time; the
        agent's timing; and the model's report about its decision, from its
        ``diagnostics`` method, or ``None`` when it has none.
        """
        output, timing = await self._wait(agent, observation)
        model, key = self.controllers[agent]
        diagnostics = getattr(model, "diagnostics", None)
        report = None if diagnostics is None else await _call(self.direct, diagnostics, key)
        return output, timing, report

    async def _wait(self, agent: str, observation: Any) -> tuple[Any, AgentTiming]:
        model, key = self.controllers[agent]
        await _call(self.direct, model.perceive, key, observation)
        self.owed[agent] += 1
        clock = _Clock(self.settings)
        late: LateOutput | None = None
        while True:
            output = await _call(self.direct, model.poll, key)
            if output is not NOTHING:
                self.owed[agent] -= 1
                if self.owed[agent] > 0:  # it answers an earlier observation
                    late = "used" if self.settings.late_outputs == "use" else "discarded"
                    if late == "discarded":
                        continue
                if self.settings.timing == "real_time":  # act at the end of the tick
                    await self._wait_out(agent, clock)
                return output, AgentTiming(elapsed=clock.elapsed(), late=late)
            if clock.done():
                return NOTHING, AgentTiming(elapsed=clock.elapsed(), defaulted=True, late=late)
            await self._advance(agent, clock)

    async def _advance(self, agent: str, clock: "_Clock") -> None:
        """Let ``agent``'s model work for one call to ``advance``, or until time runs out.

        On the simulated clock, the call is one unit of time. On the wall
        clock, a call still running when time runs out keeps running, and
        the agent's next wait continues it, so its output arrives late.
        """
        model, key = self.controllers[agent]
        advance = getattr(model, "advance", None)
        if not clock.wall:
            await _call(self.model_calls, advance, key)  # type: ignore[arg-type]
            clock.steps += 1
        elif advance is None:  # the model works on its own: look again shortly
            remaining = clock.remaining()
            await asyncio.sleep(
                _POLL_INTERVAL if remaining is None else min(_POLL_INTERVAL, remaining)
            )
        else:
            task = self.advancing.get(agent) or asyncio.create_task(
                _call(self.model_calls, advance, key)
            )
            self.advancing[agent] = task
            done, _ = await asyncio.wait({task}, timeout=clock.remaining())
            if done:
                del self.advancing[agent]
                task.result()  # its error, if it raised one

    async def _wait_out(self, agent: str, clock: "_Clock") -> None:
        """Let the rest of ``agent``'s tick pass; on the simulated clock its model advances."""
        while not clock.done():
            if clock.wall:
                await asyncio.sleep(clock.remaining() or 0)
            else:
                await self._advance(agent, clock)


class _Clock:
    """Measures one wait: calls to ``advance`` on the simulated clock, seconds on the wall clock."""

    def __init__(self, settings: EpisodeSettings) -> None:
        self.wall = settings.clock == "wall"
        self.budget = settings.budget
        self.start = time.monotonic()
        self.steps = 0

    def elapsed(self) -> float:
        return time.monotonic() - self.start if self.wall else self.steps

    def remaining(self) -> float | None:
        return None if self.budget is None else max(0.0, self.budget - self.elapsed())

    def done(self) -> bool:
        return self.remaining() == 0


class _Writer:
    """Builds the record, writes it to the store, if there is one, and passes on each step."""

    def __init__(self, header: RecordHeader, store: Store | None, on_step: OnStep | None) -> None:
        self.id = header.id
        self.builder = RecordBuilder(header)
        self.store, self.on_step = store, on_step
        self.record: Record | None = None  # once finished
        if store is not None:
            store.begin(header)

    async def append(self, step: RecordStep) -> None:
        self.builder.append(step)
        if self.store is not None:
            self.store.append(self.id, step)
        if self.on_step is not None:
            await Execution().call(self.on_step, step)

    def finish(self, summary: RecordSummary) -> Record:
        self.record = self.builder.finish(summary)
        if self.store is not None:
            self.store.finish(self.id, summary)
        return self.record


async def _call(
    execution: Execution, method: Callable[..., Any], key: Hashable | None, *args: Any
) -> Any:
    """Call a model's method, passing ``key`` only when there is one."""
    if key is None:
        return await execution.call(method, *args)
    return await execution.call(method, *args, key=key)


def _intervene(actions: Mapping[str, Any], response: TeacherResponse) -> dict[str, Any]:
    """``actions`` with the interventions of an answered response applied."""
    if response.status != "answered":
        return dict(actions)
    interventions = response.contributions.get("intervention", {})
    return {agent: interventions.get(agent, action) for agent, action in actions.items()}


def _default_context(instance: Instance, observations: Mapping[str, Any]) -> dict[str, Any]:
    return {"instance": instance, "observations": dict(observations)}


def _check_interruptible(
    execution: Execution, timeout: float, method: Callable[..., Any], role: Role
) -> None:
    """Refuse a time limit that could never interrupt ``method``, before the attempt begins."""
    try:
        execution.check_timeout(timeout, method)
    except ValueError as error:
        error.add_note(f"In the episode loop, list {role!r} in settings.threads.")
        raise
